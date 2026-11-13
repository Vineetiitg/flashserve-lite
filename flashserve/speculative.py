"""
Speculative Decoding Engine.

Implements the speculative decoding algorithm (Leviathan et al., 2023) where a fast
draft model proposes K candidate tokens and a larger target model verifies them in a
single forward pass, achieving up to Kx speedup without changing the output distribution.

Algorithm Overview:
  1. Draft model autoregressively generates K candidate tokens.
  2. Target model evaluates all K+1 positions in ONE forward pass.
  3. Accept/reject loop: for each draft token, accept if target agrees (probability check).
  4. On first rejection, sample from an adjusted distribution.
  5. If all K tokens accepted, bonus-sample one more from target's K+1 position.
  6. Guarantees identical output distribution to target-only decoding.

References:
  - "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al., 2023)
  - "Accelerating Large Language Model Decoding with Speculative Sampling" (Chen et al., 2023)
"""

import time
from dataclasses import dataclass, field
from typing import Generator, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding.
    
    Attributes:
        draft_model_name: HuggingFace model ID for the fast draft model.
        num_speculative_tokens: Number of tokens (K) the draft model proposes per step.
        draft_device: Device for draft model (defaults to same as target).
        draft_dtype: Precision for draft model.
    """
    draft_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    num_speculative_tokens: int = 4
    draft_device: str = "cuda"
    draft_dtype: str = "float16"


@dataclass
class SpeculativeMetrics:
    """Tracks speculative decoding performance statistics."""
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    total_generated_tokens: int = 0
    total_target_calls: int = 0
    total_draft_calls: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens

    @property
    def tokens_per_target_call(self) -> float:
        if self.total_target_calls == 0:
            return 0.0
        return self.total_generated_tokens / self.total_target_calls

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def tokens_per_second(self) -> float:
        elapsed = self.elapsed_seconds
        if elapsed <= 0:
            return 0.0
        return self.total_generated_tokens / elapsed

    def summary(self) -> dict:
        return {
            "total_generated": self.total_generated_tokens,
            "acceptance_rate": round(self.acceptance_rate * 100, 1),
            "tokens_per_target_call": round(self.tokens_per_target_call, 2),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "total_target_calls": self.total_target_calls,
            "total_draft_calls": self.total_draft_calls,
        }


class SpeculativeDecoder:
    """
    Speculative decoding engine using a draft model to accelerate a target model.
    
    The draft model generates K candidate tokens quickly, then the target model
    verifies all K candidates in a single forward pass. Accepted tokens are emitted
    without additional target model calls, giving up to K+1 tokens per target call.
    """

    def __init__(
        self,
        target_model: AutoModelForCausalLM,
        target_tokenizer: AutoTokenizer,
        config: SpeculativeConfig,
        target_device: str = "cuda",
    ):
        self.target_model = target_model
        self.tokenizer = target_tokenizer
        self.config = config
        self.K = config.num_speculative_tokens
        self.target_device = torch.device(target_device)
        self.draft_device = torch.device(config.draft_device)
        self.draft_dtype = getattr(torch, config.draft_dtype, torch.float16)

        # Load the draft model (lightweight, fast)
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            config.draft_model_name,
            dtype=self.draft_dtype,
            device_map=self.draft_device,
            trust_remote_code=True,
        )
        self.draft_model.eval()

        # Get EOS token
        self.eos_token_id = target_tokenizer.eos_token_id or 151645

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> Generator[dict, None, None]:
        """
        Generate tokens using speculative decoding with streaming output.
        
        Yields dicts with:
          - "tokens": list of accepted token strings
          - "accepted": number of draft tokens accepted this step
          - "drafted": number of draft tokens proposed this step
          - "metrics": running performance metrics
          
        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 = greedy).
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.target_device)
        metrics = SpeculativeMetrics()
        generated_ids = []

        while len(generated_ids) < max_new_tokens:
            remaining = max_new_tokens - len(generated_ids)
            K = min(self.K, remaining)

            if K == 0:
                break

            # ─── Step 1: Draft model proposes K candidate tokens ───
            draft_tokens, draft_probs = self._draft_propose(input_ids, K, temperature)
            metrics.total_draft_tokens += len(draft_tokens)
            metrics.total_draft_calls += K

            # ─── Step 2: Target model verifies all K tokens in one pass ───
            target_probs = self._target_verify(input_ids, draft_tokens)
            metrics.total_target_calls += 1

            # ─── Step 3: Accept/Reject loop ───
            accepted_tokens, bonus_token = self._accept_reject(
                draft_tokens, draft_probs, target_probs, temperature
            )
            metrics.total_accepted_tokens += len([t for t in accepted_tokens if t in draft_tokens])

            all_new_tokens = accepted_tokens
            if bonus_token is not None:
                all_new_tokens.append(bonus_token)

            # Check for EOS in accepted tokens
            eos_idx = None
            for i, tid in enumerate(all_new_tokens):
                if tid == self.eos_token_id:
                    eos_idx = i
                    break

            if eos_idx is not None:
                all_new_tokens = all_new_tokens[:eos_idx]

            if not all_new_tokens and eos_idx is not None:
                break

            generated_ids.extend(all_new_tokens)
            metrics.total_generated_tokens += len(all_new_tokens)

            # Update input_ids for next iteration
            new_ids = torch.tensor([all_new_tokens], device=self.target_device, dtype=torch.long)
            input_ids = torch.cat([input_ids, new_ids], dim=1)

            # Decode and yield
            token_texts = [self.tokenizer.decode([tid], skip_special_tokens=True) for tid in all_new_tokens]
            yield {
                "tokens": token_texts,
                "accepted": len(accepted_tokens),
                "drafted": K,
                "metrics": metrics.summary(),
            }

            if eos_idx is not None:
                break

        # Final metrics
        yield {
            "tokens": [],
            "accepted": 0,
            "drafted": 0,
            "metrics": metrics.summary(),
            "finished": True,
        }

    def _draft_propose(
        self,
        prefix_ids: torch.Tensor,
        K: int,
        temperature: float,
    ) -> tuple[list[int], list[torch.Tensor]]:
        """
        Autoregressively generate K candidate tokens with the draft model.
        
        Returns:
            draft_tokens: List of K proposed token IDs.
            draft_probs: List of K probability distributions (softmax over vocab).
        """
        draft_tokens = []
        draft_probs = []
        current_ids = prefix_ids.clone()

        for _ in range(K):
            outputs = self.draft_model(
                input_ids=current_ids,
                use_cache=False,  # Simplified: no KV cache for draft
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :]  # [1, vocab_size]

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)

            if temperature > 0:
                token_id = torch.multinomial(probs[0], num_samples=1).item()
            else:
                token_id = torch.argmax(logits, dim=-1).item()

            draft_tokens.append(token_id)
            draft_probs.append(probs[0].clone())  # [vocab_size]

            # Extend input for next draft step
            current_ids = torch.cat(
                [current_ids, torch.tensor([[token_id]], device=current_ids.device)],
                dim=1,
            )

            if token_id == self.eos_token_id:
                break

        return draft_tokens, draft_probs

    def _target_verify(
        self,
        prefix_ids: torch.Tensor,
        draft_tokens: list[int],
    ) -> list[torch.Tensor]:
        """
        Run the target model on [prefix + draft_tokens] in a single forward pass.
        Extract probability distributions at each draft token position.
        
        Returns:
            List of K+1 probability distributions from the target model.
        """
        # Build full input: prefix + draft tokens
        draft_tensor = torch.tensor([draft_tokens], device=self.target_device, dtype=torch.long)
        full_input = torch.cat([prefix_ids, draft_tensor], dim=1)

        outputs = self.target_model(
            input_ids=full_input,
            use_cache=False,
            return_dict=True,
        )

        # Extract logits at positions corresponding to draft tokens
        # If prefix has length P and draft has K tokens:
        #   - Position P-1 predicts first draft token
        #   - Position P predicts second draft token
        #   - ...
        #   - Position P+K-1 predicts the bonus token
        prefix_len = prefix_ids.shape[1]
        target_probs = []
        for i in range(len(draft_tokens) + 1):
            pos = prefix_len - 1 + i
            if pos < full_input.shape[1]:
                logits = outputs.logits[:, pos, :]  # [1, vocab_size]
                probs = torch.softmax(logits, dim=-1)
                target_probs.append(probs[0].clone())

        return target_probs

    def _accept_reject(
        self,
        draft_tokens: list[int],
        draft_probs: list[torch.Tensor],
        target_probs: list[torch.Tensor],
        temperature: float,
    ) -> tuple[list[int], Optional[int]]:
        """
        Standard speculative decoding acceptance/rejection.
        
        For greedy decoding (temperature=0): Accept if target's argmax matches draft.
        For stochastic decoding: Accept with probability min(1, p_target/p_draft).
        
        Returns:
            accepted_tokens: List of accepted token IDs.
            bonus_token: Extra token from target if all K were accepted, else None.
        """
        accepted_tokens = []

        for i, draft_token in enumerate(draft_tokens):
            if i >= len(target_probs):
                break

            if temperature <= 0:
                # Greedy mode: accept if target's top-1 matches draft
                target_top = torch.argmax(target_probs[i]).item()
                if target_top == draft_token:
                    accepted_tokens.append(draft_token)
                else:
                    # Reject: use target's choice instead
                    accepted_tokens.append(target_top)
                    return accepted_tokens, None
            else:
                # Stochastic mode: accept with probability min(1, p_target/p_draft)
                p_target = target_probs[i][draft_token].item()
                p_draft = draft_probs[i][draft_token].item()

                if p_draft <= 0:
                    accepted_tokens.append(draft_token)
                    continue

                accept_prob = min(1.0, p_target / p_draft)
                r = torch.rand(1).item()

                if r < accept_prob:
                    accepted_tokens.append(draft_token)
                else:
                    # Reject: sample from adjusted distribution
                    adjusted = torch.clamp(target_probs[i] - draft_probs[i], min=0)
                    adjusted_sum = adjusted.sum()
                    if adjusted_sum > 0:
                        adjusted = adjusted / adjusted_sum
                        corrected_token = torch.multinomial(adjusted, num_samples=1).item()
                    else:
                        corrected_token = torch.argmax(target_probs[i]).item()
                    accepted_tokens.append(corrected_token)
                    return accepted_tokens, None

        # All K tokens accepted — get bonus token from target's last position
        bonus_token = None
        if len(accepted_tokens) == len(draft_tokens) and len(target_probs) > len(draft_tokens):
            bonus_probs = target_probs[len(draft_tokens)]
            if temperature <= 0:
                bonus_token = torch.argmax(bonus_probs).item()
            else:
                bonus_token = torch.multinomial(bonus_probs, num_samples=1).item()

        return accepted_tokens, bonus_token
