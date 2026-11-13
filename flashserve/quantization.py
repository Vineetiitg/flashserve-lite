"""
Quantization utilities for INT8 and INT4 model loading via bitsandbytes.

Supports three precision modes:
  - fp16:  Standard half-precision inference (default)
  - int8:  8-bit quantization via LLM.int8() — ~50% memory reduction
  - int4:  4-bit quantization via NF4/FP4   — ~75% memory reduction

Both quantized modes use bitsandbytes under the hood and require a CUDA GPU.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch


class QuantizationMode(Enum):
    """Supported model precision modes."""
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"


@dataclass
class QuantizationConfig:
    """Configuration for model quantization.
    
    Attributes:
        mode: Quantization mode (fp16, int8, or int4).
        compute_dtype: Dtype for computation in quantized layers (used by int4).
        use_double_quant: Whether to use double quantization for int4 (saves ~0.4 bits/param).
        bnb_4bit_quant_type: Quantization type for 4-bit — 'nf4' (default) or 'fp4'.
    """
    mode: QuantizationMode = QuantizationMode.FP16
    compute_dtype: torch.dtype = torch.float16
    use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"

    @property
    def is_quantized(self) -> bool:
        return self.mode != QuantizationMode.FP16


def get_quantization_kwargs(quant_config: QuantizationConfig) -> dict:
    """
    Build the keyword arguments to pass to AutoModelForCausalLM.from_pretrained()
    based on the requested quantization mode.
    
    Returns:
        dict of kwargs for model loading.
    """
    if quant_config.mode == QuantizationMode.FP16:
        return {
            "dtype": torch.float16,
            "device_map": "auto",
        }

    # Lazy-import to avoid hard dependency if not using quantization
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        raise ImportError(
            "bitsandbytes and accelerate are required for quantized inference. "
            "Install with: pip install bitsandbytes accelerate"
        )

    if quant_config.mode == QuantizationMode.INT8:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        return {
            "quantization_config": bnb_config,
            "device_map": "auto",
        }

    if quant_config.mode == QuantizationMode.INT4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=quant_config.compute_dtype,
            bnb_4bit_use_double_quant=quant_config.use_double_quant,
            bnb_4bit_quant_type=quant_config.bnb_4bit_quant_type,
        )
        return {
            "quantization_config": bnb_config,
            "device_map": "auto",
        }

    raise ValueError(f"Unsupported quantization mode: {quant_config.mode}")


def estimate_model_memory(
    num_params_millions: float,
    mode: QuantizationMode,
) -> float:
    """
    Estimate GPU memory in MB for a given model size and quantization mode.
    
    Args:
        num_params_millions: Number of model parameters in millions.
        mode: Quantization precision mode.
        
    Returns:
        Estimated memory usage in MB.
    """
    bytes_per_param = {
        QuantizationMode.FP16: 2.0,
        QuantizationMode.INT8: 1.0,
        QuantizationMode.INT4: 0.5,
    }
    bpp = bytes_per_param.get(mode, 2.0)
    # Add ~20% overhead for optimizer states, activations, etc.
    return num_params_millions * 1e6 * bpp / (1024 * 1024) * 1.2


def get_model_memory_footprint(model) -> dict:
    """
    Measure actual GPU memory used by a loaded model.
    
    Returns:
        Dict with total_mb, parameter details, and device info.
    """
    total_bytes = 0
    param_count = 0
    for param in model.parameters():
        param_count += param.numel()
        if hasattr(param, "quant_state"):
            # bitsandbytes quantized parameter
            total_bytes += param.numel() * param.quant_state.blocksize
        else:
            total_bytes += param.numel() * param.element_size()

    return {
        "total_parameters": param_count,
        "total_mb": round(total_bytes / (1024 * 1024), 2),
        "gpu_allocated_mb": round(
            torch.cuda.memory_allocated() / (1024 * 1024), 2
        ) if torch.cuda.is_available() else 0.0,
        "gpu_reserved_mb": round(
            torch.cuda.memory_reserved() / (1024 * 1024), 2
        ) if torch.cuda.is_available() else 0.0,
    }
