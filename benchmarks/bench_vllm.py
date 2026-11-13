"""
Benchmark: vLLM Production Inference Engine Comparison.
Detects if vLLM is installed; if not available (e.g. native Windows), provides clean fallback.
"""

import time
from benchmarks.workload import generate_workload
from benchmarks.metrics import RequestMetrics, compute_summary


def run_vllm_benchmark(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    num_requests: int = 16,
    max_new_tokens: int = 64,
) -> dict:
    """Executes vLLM benchmark if installed on environment."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise RuntimeError(
            "vLLM is not installed in the current environment (vLLM officially targets Linux). "
            "FlashServe-Lite provides Python-level paged inference natively on Windows."
        )

    llm = LLM(
        model=model_name,
        dtype="float16",
        gpu_memory_utilization=0.85,
        max_model_len=1024,
    )
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
    )

    prompts = generate_workload(num_requests)
    all_metrics: list[RequestMetrics] = []

    t_start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    t_end = time.time()
    wall_clock = t_end - t_start

    for output in outputs:
        metrics = RequestMetrics(
            prompt_tokens=len(output.prompt_token_ids),
            generated_tokens=len(output.outputs[0].token_ids),
            total_time=wall_clock / len(outputs),
        )
        all_metrics.append(metrics)

    return compute_summary(all_metrics, wall_clock)


if __name__ == "__main__":
    try:
        res = run_vllm_benchmark(num_requests=4, max_new_tokens=64)
        print(res)
    except Exception as e:
        print(f"vLLM check: {e}")
