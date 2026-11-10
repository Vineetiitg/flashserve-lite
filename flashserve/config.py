"""
FlashServe-Lite Engine Configuration.

Central configuration for model loading, KV cache sizing, scheduler limits,
quantization mode, speculative decoding, and server parameters.
"""

from dataclasses import dataclass, field
import torch


@dataclass
class EngineConfig:
    """Complete engine configuration with sensible defaults for RTX 3050 6GB."""

    # ── Model ──────────────────────────────────────────────────────────
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: str = "float16"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    local_files_only: bool = False

    # ── Quantization ───────────────────────────────────────────────────
    quantization: str = "fp16"             # "fp16", "int8", or "int4"

    # ── Paged KV Cache ─────────────────────────────────────────────────
    block_size: int = 16
    num_gpu_blocks: int = 8000

    # ── Scheduler ──────────────────────────────────────────────────────
    max_batch_size: int = 16
    max_seq_len: int = 1024
    max_queue_size: int = 128

    # ── Sampling Defaults ──────────────────────────────────────────────
    temperature: float = 0.0
    top_k: int = 50
    top_p: float = 0.9
    max_new_tokens: int = 256

    # ── Speculative Decoding ───────────────────────────────────────────
    enable_speculative: bool = False
    draft_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    num_speculative_tokens: int = 4

    # ── Server ─────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
