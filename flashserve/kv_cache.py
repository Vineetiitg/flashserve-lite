"""
Paged KV Cache Manager.

Implements vLLM-inspired virtual memory paging for Key-Value attention tensors.
Memory is divided into fixed-size physical blocks of tokens. Sequences hold a
logical block table mapping sequence token indices to physical pool blocks.
"""

from typing import Optional
import torch


class PagedKVCache:
    """
    Pre-allocates and manages a contiguous GPU memory pool partitioned into blocks.
    Provides scatter (write) and gather (read) operations for transformer KV caches.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        num_blocks: int = 8000,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype: torch.dtype = torch.float16,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.device = device
        self.dtype = dtype

        # Pre-allocate contiguous pool: [num_layers, 2 (K/V), num_blocks, block_size, num_kv_heads, head_dim]
        self.kv_pool = torch.zeros(
            (num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )

        # Free block stack (LIFO for cache-locality)
        self.free_blocks: list[int] = list(range(num_blocks))
        self.num_free_blocks: int = num_blocks

        # Block tables: seq_id -> list of physical block IDs
        self.block_tables: dict[int, list[int]] = {}

    def get_num_required_blocks(self, seq_len: int) -> int:
        """Returns the number of blocks needed to store `seq_len` tokens."""
        if seq_len <= 0:
            return 1
        return (seq_len + self.block_size - 1) // self.block_size

    def allocate_blocks(self, seq_id: int, num_blocks_needed: int) -> list[int]:
        """Allocate `num_blocks_needed` physical blocks for a new sequence."""
        if num_blocks_needed > self.num_free_blocks:
            raise RuntimeError(
                f"Out of KV Cache Memory: Requested {num_blocks_needed} blocks, "
                f"only {self.num_free_blocks} available out of {self.num_blocks} total."
            )

        allocated: list[int] = []
        for _ in range(num_blocks_needed):
            block_id = self.free_blocks.pop()
            allocated.append(block_id)
            self.num_free_blocks -= 1

        self.block_tables[seq_id] = allocated
        return allocated

    def allocate_one_block(self, seq_id: int) -> int:
        """Append a single newly allocated block to an existing sequence's block table."""
        if self.num_free_blocks <= 0:
            raise RuntimeError(
                f"Out of KV Cache Memory: Cannot allocate next block for seq {seq_id}."
            )

        block_id = self.free_blocks.pop()
        self.num_free_blocks -= 1
        if seq_id not in self.block_tables:
            self.block_tables[seq_id] = []
        self.block_tables[seq_id].append(block_id)
        return block_id

    def free_sequence(self, seq_id: int):
        """Reclaim all physical blocks assigned to a finished sequence."""
        if seq_id in self.block_tables:
            blocks = self.block_tables.pop(seq_id)
            self.free_blocks.extend(blocks)
            self.num_free_blocks += len(blocks)

    def scatter_prefill(
        self,
        seq_id: int,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ):
        """
        Store full prompt KV activations into allocated physical blocks.
        
        Args:
            seq_id: Identifier for the sequence
            layer_idx: Transformer layer index [0, num_layers - 1]
            key_states: [num_kv_heads, prompt_len, head_dim]
            value_states: [num_kv_heads, prompt_len, head_dim]
        """
        seq_len = key_states.shape[1]
        blocks = self.block_tables[seq_id]

        for block_idx, block_id in enumerate(blocks):
            start = block_idx * self.block_size
            end = min(start + self.block_size, seq_len)
            tokens_in_block = end - start
            if tokens_in_block <= 0:
                break

            # Permute from [num_kv_heads, tokens, head_dim] to [tokens, num_kv_heads, head_dim]
            k_slice = key_states[:, start:end, :].permute(1, 0, 2)
            v_slice = value_states[:, start:end, :].permute(1, 0, 2)

            self.kv_pool[layer_idx, 0, block_id, :tokens_in_block] = k_slice
            self.kv_pool[layer_idx, 1, block_id, :tokens_in_block] = v_slice

    def scatter_decode_token(
        self,
        seq_id: int,
        layer_idx: int,
        position: int,
        key_state: torch.Tensor,
        value_state: torch.Tensor,
    ):
        """
        Store a single newly generated token's KV activation into the paged pool.
        
        Args:
            seq_id: Identifier for the sequence
            layer_idx: Transformer layer index
            position: Global token position (0-indexed)
            key_state: [num_kv_heads, 1, head_dim] or [num_kv_heads, head_dim]
            value_state: [num_kv_heads, 1, head_dim] or [num_kv_heads, head_dim]
        """
        block_idx = position // self.block_size
        block_offset = position % self.block_size

        if seq_id not in self.block_tables or block_idx >= len(self.block_tables[seq_id]):
            self.allocate_one_block(seq_id)

        physical_block = self.block_tables[seq_id][block_idx]

        # Ensure shape is [num_kv_heads, head_dim]
        if key_state.ndim == 3:
            k_vec = key_state.squeeze(1)
            v_vec = value_state.squeeze(1)
        else:
            k_vec = key_state
            v_vec = value_state

        self.kv_pool[layer_idx, 0, physical_block, block_offset] = k_vec
        self.kv_pool[layer_idx, 1, physical_block, block_offset] = v_vec

    def gather_kv(
        self,
        seq_id: int,
        layer_idx: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gather a sequence's KV activations across its physical blocks into contiguous tensors.
        
        Returns:
            key: [1, num_kv_heads, seq_len, head_dim]
            value: [1, num_kv_heads, seq_len, head_dim]
        """
        blocks = self.block_tables.get(seq_id, [])
        key_chunks: list[torch.Tensor] = []
        value_chunks: list[torch.Tensor] = []
        remaining = seq_len

        for block_id in blocks:
            tokens_in_block = min(self.block_size, remaining)
            if tokens_in_block <= 0:
                break

            # Slices have shape [tokens_in_block, num_kv_heads, head_dim]
            k_block = self.kv_pool[layer_idx, 0, block_id, :tokens_in_block]
            v_block = self.kv_pool[layer_idx, 1, block_id, :tokens_in_block]

            key_chunks.append(k_block)
            value_chunks.append(v_block)
            remaining -= tokens_in_block

            if remaining <= 0:
                break

        if not key_chunks:
            empty_k = torch.empty((1, self.num_kv_heads, 0, self.head_dim), device=self.device, dtype=self.dtype)
            empty_v = torch.empty((1, self.num_kv_heads, 0, self.head_dim), device=self.device, dtype=self.dtype)
            return empty_k, empty_v

        # Concat along token dim (0) -> [seq_len, num_kv_heads, head_dim]
        # Then permute to [1, num_kv_heads, seq_len, head_dim]
        k_cat = torch.cat(key_chunks, dim=0).permute(1, 0, 2).unsqueeze(0)
        v_cat = torch.cat(value_chunks, dim=0).permute(1, 0, 2).unsqueeze(0)

        return k_cat, v_cat

    def gather_kv_for_batch(
        self,
        seq_ids: list[int],
        seq_lens: list[int],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Gather and left-pad KV caches for a batch of sequences across all layers.
        
        Returns:
            List of (key, value) tuples per layer, each shaped:
            [batch_size, num_kv_heads, max_seq_len, head_dim]
        """
        batch_size = len(seq_ids)
        max_len = max(seq_lens)
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] = []

        for layer_idx in range(self.num_layers):
            layer_keys: list[torch.Tensor] = []
            layer_values: list[torch.Tensor] = []

            for sid, slen in zip(seq_ids, seq_lens):
                k, v = self.gather_kv(sid, layer_idx, slen)  # [1, heads, slen, dim]

                pad_len = max_len - slen
                if pad_len > 0:
                    # Left-pad the sequence length dimension (dim=2 in [batch, heads, seq, dim])
                    # Pad format is (pad_dim3_left, pad_dim3_right, pad_dim2_left, pad_dim2_right, ...)
                    k = torch.nn.functional.pad(k, (0, 0, pad_len, 0))
                    v = torch.nn.functional.pad(v, (0, 0, pad_len, 0))

                layer_keys.append(k)
                layer_values.append(v)

            stacked_k = torch.cat(layer_keys, dim=0)
            stacked_v = torch.cat(layer_values, dim=0)
            past_key_values.append((stacked_k, stacked_v))

        return past_key_values

    def get_usage_stats(self) -> dict:
        """Returns physical memory allocation and block pool utilization telemetry."""
        used_blocks = self.num_blocks - self.num_free_blocks
        total_bytes = self.kv_pool.nelement() * self.kv_pool.element_size()
        used_bytes = (used_blocks / self.num_blocks) * total_bytes if self.num_blocks > 0 else 0
        return {
            "total_blocks": self.num_blocks,
            "used_blocks": used_blocks,
            "free_blocks": self.num_free_blocks,
            "utilization_pct": round(100.0 * used_blocks / self.num_blocks, 2) if self.num_blocks > 0 else 0.0,
            "pool_size_mb": round(total_bytes / (1024 * 1024), 2),
            "used_mb": round(used_bytes / (1024 * 1024), 2),
            "active_sequences": len(self.block_tables),
        }
