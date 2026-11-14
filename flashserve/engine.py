"""
Model Inference Engine.

Separates execution into two distinct stages:
  1. Prefill: Computes full-prompt attention and scatters KV activations into paged blocks.
  2. Decode:  Generates one token per sequence per iteration using gathered KV blocks.

Supports FP16, INT8, and INT4 precision modes via bitsandbytes quantization.
"""

from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from flashserve.config import EngineConfig
from flashserve.kv_cache import PagedKVCache
from flashserve.quantization import QuantizationMode, QuantizationConfig, get_quantization_kwargs, get_model_memory_footprint
from flashserve.sampling import sample_token
from flashserve.sequence import Sequence


class InferenceEngine:
    """Manages model weights and executes prefill & batched decode steps with PagedKVCache."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype, torch.float16)

        # ── Quantization Configuration ──────────────────────────────
        self.quant_mode = QuantizationMode(config.quantization)
        self.quant_config = QuantizationConfig(mode=self.quant_mode)

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            local_files_only=config.local_files_only,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load Model with quantization support
        load_kwargs = get_quantization_kwargs(self.quant_config)
        load_kwargs["trust_remote_code"] = True
        if config.local_files_only:
            load_kwargs["local_files_only"] = True

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **load_kwargs,
        )
        self.model.eval()

        # Extract model hyperparameters
        model_config = self.model.config
        self.num_layers = model_config.num_hidden_layers
        self.num_kv_heads = getattr(
            model_config, "num_key_value_heads", model_config.num_attention_heads
        )
        self.num_heads = model_config.num_attention_heads
        self.head_dim = getattr(
            model_config, "head_dim", model_config.hidden_size // self.num_heads
        )

        # Initialize Paged KV Cache Manager
        self.kv_cache = PagedKVCache(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            block_size=config.block_size,
            num_blocks=config.num_gpu_blocks,
            device=self.device,
            dtype=self.dtype,
        )

    def _extract_layer_kv(self, past_key_values, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Extracts (K, V) tensor slices for a given layer regardless of cache implementation."""
        if hasattr(past_key_values, "layers"):
            layer = past_key_values.layers[layer_idx]
            return layer.keys, layer.values
        elif hasattr(past_key_values, "key_cache"):
            return past_key_values.key_cache[layer_idx], past_key_values.value_cache[layer_idx]
        else:
            return past_key_values[layer_idx][0], past_key_values[layer_idx][1]

    @torch.inference_mode()
    def prefill(self, sequence: Sequence) -> int:
        """
        Executes prefill for a sequence's prompt.
        Allocates physical blocks, scatters KV activations, and returns the first sampled token.
        """
        prompt_len = len(sequence.prompt_token_ids)
        if prompt_len == 0:
            raise ValueError(f"Sequence {sequence.seq_id} has empty prompt.")

        prompt_tensor = torch.tensor(
            [sequence.prompt_token_ids], device=self.device, dtype=torch.long
        )

        # Allocate physical blocks in the paged pool
        num_blocks_needed = self.kv_cache.get_num_required_blocks(prompt_len)
        self.kv_cache.allocate_blocks(sequence.seq_id, num_blocks_needed)

        # Forward pass on full prompt
        outputs = self.model(
            input_ids=prompt_tensor,
            use_cache=True,
            return_dict=True,
        )

        # Scatter activations into paged KV blocks
        for layer_idx in range(self.num_layers):
            k, v = self._extract_layer_kv(outputs.past_key_values, layer_idx)
            # Remove batch dimension: [1, heads, seq, dim] -> [heads, seq, dim]
            self.kv_cache.scatter_prefill(
                seq_id=sequence.seq_id,
                layer_idx=layer_idx,
                key_states=k.squeeze(0),
                value_states=v.squeeze(0),
            )

        sequence.current_length = prompt_len

        # Sample the initial token
        logits = outputs.logits[0, -1, :]
        first_token_id = sample_token(
            logits=logits,
            temperature=sequence.temperature,
            top_p=sequence.top_p,
            top_k=sequence.top_k,
        )
        return first_token_id

    @torch.inference_mode()
    def decode_single(self, sequence: Sequence, input_token_id: int) -> int:
        """
        Executes a single-sequence decode step using paged KV blocks.
        """
        cache = DynamicCache()
        for layer_idx in range(self.num_layers):
            k, v = self.kv_cache.gather_kv(
                seq_id=sequence.seq_id,
                layer_idx=layer_idx,
                seq_len=sequence.current_length,
            )
            cache.update(k, v, layer_idx)

        input_tensor = torch.tensor(
            [[input_token_id]], device=self.device, dtype=torch.long
        )

        outputs = self.model(
            input_ids=input_tensor,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )

        # Scatter the new token's KV activation into paged storage
        new_position = sequence.current_length
        for layer_idx in range(self.num_layers):
            k, v = self._extract_layer_kv(outputs.past_key_values, layer_idx)
            # Last token: [1, heads, -1:, dim]
            new_k = k[:, :, -1:, :]
            new_v = v[:, :, -1:, :]
            self.kv_cache.scatter_decode_token(
                seq_id=sequence.seq_id,
                layer_idx=layer_idx,
                position=new_position,
                key_state=new_k.squeeze(0),
                value_state=new_v.squeeze(0),
            )

        sequence.current_length += 1

        logits = outputs.logits[0, -1, :]
        next_token_id = sample_token(
            logits=logits,
            temperature=sequence.temperature,
            top_p=sequence.top_p,
            top_k=sequence.top_k,
        )
        return next_token_id

    @torch.inference_mode()
    def decode_batch(
        self,
        sequences: list[Sequence],
        input_token_ids: list[int],
    ) -> list[int]:
        """
        Executes batched decode across all active sequences in parallel.
        Left-pads gathered KV caches and applies attention masks.
        """
        if len(sequences) == 1:
            return [self.decode_single(sequences[0], input_token_ids[0])]

        batch_size = len(sequences)
        seq_ids = [s.seq_id for s in sequences]
        seq_lens = [s.current_length for s in sequences]
        max_len = max(seq_lens)

        # Gather and left-pad KV caches for batch
        past_kv_list = self.kv_cache.gather_kv_for_batch(seq_ids, seq_lens)
        cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(past_kv_list):
            cache.update(k, v, layer_idx)

        # Input token tensors: [batch_size, 1]
        input_tensor = torch.tensor(
            [[tid] for tid in input_token_ids],
            device=self.device,
            dtype=torch.long,
        )

        # Build attention mask: [batch_size, max_len + 1]
        attention_mask = torch.zeros(
            batch_size, max_len + 1, device=self.device, dtype=torch.long
        )
        for i, slen in enumerate(seq_lens):
            pad_len = max_len - slen
            attention_mask[i, pad_len:] = 1

        position_ids = torch.tensor(
            [[slen] for slen in seq_lens],
            device=self.device,
            dtype=torch.long,
        )

        outputs = self.model(
            input_ids=input_tensor,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )

        # Scatter newly generated token KV activations for each sequence
        for layer_idx in range(self.num_layers):
            k, v = self._extract_layer_kv(outputs.past_key_values, layer_idx)
            for i, seq in enumerate(sequences):
                new_k = k[i : i + 1, :, -1:, :]
                new_v = v[i : i + 1, :, -1:, :]
                self.kv_cache.scatter_decode_token(
                    seq_id=seq.seq_id,
                    layer_idx=layer_idx,
                    position=seq.current_length,
                    key_state=new_k.squeeze(0),
                    value_state=new_v.squeeze(0),
                )

        next_token_ids: list[int] = []
        for i, seq in enumerate(sequences):
            seq.current_length += 1
            logits = outputs.logits[i, -1, :]
            token_id = sample_token(
                logits=logits,
                temperature=seq.temperature,
                top_p=seq.top_p,
                top_k=seq.top_k,
            )
            next_token_ids.append(token_id)

        return next_token_ids

    def free_sequence(self, seq_id: int):
        """Release all allocated physical memory for sequence."""
        self.kv_cache.free_sequence(seq_id)
