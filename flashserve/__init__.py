"""
FlashServe-Lite: High-Throughput Mini LLM Serving Engine
with Paged KV Cache, Continuous Batching, Quantization, and Speculative Decoding.
"""

# Windows Application Control compatibility for torch dynamo polyfills
try:
    import torch.utils._pytree as _torch_pytree
    _torch_pytree._cxx_pytree_dynamo_traceable = False
except Exception:
    pass

from flashserve.config import EngineConfig
from flashserve.sequence import Sequence, SequenceStatus
from flashserve.kv_cache import PagedKVCache
from flashserve.quantization import QuantizationMode, QuantizationConfig
from flashserve.engine import InferenceEngine
from flashserve.scheduler import ContinuousBatchingScheduler
from flashserve.speculative import SpeculativeDecoder, SpeculativeConfig

__version__ = "0.2.0"
__all__ = [
    "EngineConfig",
    "Sequence",
    "SequenceStatus",
    "PagedKVCache",
    "QuantizationMode",
    "QuantizationConfig",
    "InferenceEngine",
    "ContinuousBatchingScheduler",
    "SpeculativeDecoder",
    "SpeculativeConfig",
]
