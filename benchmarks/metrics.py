"""
Latency, throughput, and GPU memory metrics measurement utilities.
"""

from dataclasses import dataclass, field
import numpy as np
import torch


@dataclass
class RequestMetrics:
    """Metrics recorded for a single generation request."""
    prompt_tokens: int = 0
    generated_tokens: int = 0
    time_to_first_token: float = 0.0   # seconds
    total_time: float = 0.0            # seconds
    token_timestamps: list[float] = field(default_factory=list)

    @property
    def inter_token_latencies(self) -> list[float]:
        """Inter-token latency values (in milliseconds) between consecutive tokens."""
        if len(self.token_timestamps) < 2:
            return []
        return [
            (self.token_timestamps[i] - self.token_timestamps[i - 1]) * 1000.0
            for i in range(1, len(self.token_timestamps))
        ]


def compute_summary(all_metrics: list[RequestMetrics], wall_clock_seconds: float) -> dict:
    """Aggregates per-request metrics into engine-wide summary statistics."""
    if not all_metrics:
        return {
            "num_requests": 0,
            "total_tokens_generated": 0,
            "wall_clock_seconds": 0.0,
            "throughput_tokens_per_sec": 0.0,
            "ttft_p50_ms": 0.0,
            "ttft_p95_ms": 0.0,
            "itl_p50_ms": 0.0,
            "itl_p95_ms": 0.0,
            "gpu_peak_memory_mb": 0.0,
        }

    ttft_values = [m.time_to_first_token * 1000.0 for m in all_metrics if m.time_to_first_token > 0]
    total_generated = sum(m.generated_tokens for m in all_metrics)

    all_itl: list[float] = []
    for m in all_metrics:
        all_itl.extend(m.inter_token_latencies)

    gpu_peak_mb = 0.0
    if torch.cuda.is_available():
        gpu_peak_mb = round(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), 2)

    return {
        "num_requests": len(all_metrics),
        "total_tokens_generated": total_generated,
        "wall_clock_seconds": round(wall_clock_seconds, 2),
        "throughput_tokens_per_sec": (
            round(total_generated / wall_clock_seconds, 2) if wall_clock_seconds > 0 else 0.0
        ),
        "ttft_p50_ms": round(float(np.percentile(ttft_values, 50)), 2) if ttft_values else 0.0,
        "ttft_p95_ms": round(float(np.percentile(ttft_values, 95)), 2) if ttft_values else 0.0,
        "itl_p50_ms": round(float(np.percentile(all_itl, 50)), 2) if all_itl else 0.0,
        "itl_p95_ms": round(float(np.percentile(all_itl, 95)), 2) if all_itl else 0.0,
        "gpu_peak_memory_mb": gpu_peak_mb,
    }
