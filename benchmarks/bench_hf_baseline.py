"""
Benchmark: Naive HuggingFace model.generate() Baseline.
Processes requests sequentially with default contiguous KV cache allocation.
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from benchmarks.workload import generate_workload
from benchmarks.metrics import RequestMetrics, compute_summary


def run_hf_baseline(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    num_requests: int = 16,
    max_new_tokens: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """Executes sequential HuggingFace generate() benchmark."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map=device,
        local_files_only=True,
    )
    model.eval()

    prompts = generate_workload(num_requests)
    all_metrics: list[RequestMetrics] = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t_bench_start = time.time()

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        t_req_start = time.time()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.0,
                use_cache=True,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_req_end = time.time()

        gen_len = outputs.shape[1] - prompt_len
        metrics = RequestMetrics(
            prompt_tokens=prompt_len,
            generated_tokens=gen_len,
            time_to_first_token=0.0,
            total_time=t_req_end - t_req_start,
        )
        all_metrics.append(metrics)

    wall_clock = time.time() - t_bench_start
    return compute_summary(all_metrics, wall_clock)


if __name__ == "__main__":
    import json
    print("Running HuggingFace baseline (4 requests, 64 tokens)...")
    res = run_hf_baseline(num_requests=4, max_new_tokens=64)
    print(json.dumps(res, indent=2))
