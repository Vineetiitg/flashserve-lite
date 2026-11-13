"""
Benchmark: FlashServe-Lite High-Throughput Serving Engine.
Submits concurrent async requests and instruments exact per-token streaming timestamps.
"""

import asyncio
import time
import torch

from flashserve.config import EngineConfig
from flashserve.engine import InferenceEngine
from flashserve.scheduler import ContinuousBatchingScheduler
from benchmarks.workload import generate_workload
from benchmarks.metrics import RequestMetrics, compute_summary


async def run_flashserve_benchmark(
    num_requests: int = 16,
    max_new_tokens: int = 64,
    max_batch_size: int = 8,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
) -> dict:
    """Executes asynchronous concurrent benchmark on FlashServe-Lite engine."""
    config = EngineConfig(
        model_name=model_name,
        max_batch_size=max_batch_size,
        max_new_tokens=max_new_tokens,
        local_files_only=True,
    )
    engine = InferenceEngine(config)
    scheduler = ContinuousBatchingScheduler(engine, config)

    prompts = generate_workload(num_requests)
    all_metrics: list[RequestMetrics] = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Launch background scheduler loop
    loop_task = asyncio.create_task(scheduler.run_loop())

    async def client_worker(prompt: str) -> RequestMetrics:
        metrics = RequestMetrics()
        t_req_start = time.time()
        first_token_seen = False

        seq_id, q = scheduler.add_request(
            prompt_text=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
        metrics.prompt_tokens = len(engine.tokenizer.encode(prompt))

        while True:
            token = await q.get()
            if token is None:
                break

            t_now = time.time()
            metrics.token_timestamps.append(t_now)
            metrics.generated_tokens += 1

            if not first_token_seen:
                metrics.time_to_first_token = t_now - t_req_start
                first_token_seen = True

        metrics.total_time = time.time() - t_req_start
        return metrics

    t_bench_start = time.time()
    # Fire all concurrent client requests
    results = await asyncio.gather(*(client_worker(p) for p in prompts))
    all_metrics.extend(results)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_clock = time.time() - t_bench_start

    # Terminate background scheduler
    scheduler.is_running = False
    loop_task.cancel()
    try:
        await loop_task
    except asyncio.CancelledError:
        pass

    return compute_summary(all_metrics, wall_clock)


if __name__ == "__main__":
    import json
    print("Running FlashServe-Lite benchmark (4 requests, 64 tokens, batch_size=4)...")
    res = asyncio.run(run_flashserve_benchmark(num_requests=4, max_new_tokens=64, max_batch_size=4))
    print(json.dumps(res, indent=2))
