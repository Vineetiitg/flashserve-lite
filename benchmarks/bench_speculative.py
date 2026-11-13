"""
Speculative Decoding Benchmark.

Measures the speedup from speculative decoding by comparing:
  1. Standard autoregressive decoding (target model only)
  2. Speculative decoding (draft model proposes, target model verifies)

Reports acceptance rate, tokens per target call, and effective speedup.
"""

import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flashserve.speculative import SpeculativeDecoder, SpeculativeConfig
from benchmarks.workload import PROMPTS


def run_standard_generation(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 64,
) -> dict:
    """Baseline: standard autoregressive generation."""
    total_tokens = 0
    t_start = time.time()

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.0,
            )
            total_tokens += output_ids.shape[1] - input_ids.shape[1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t_start

    return {
        "total_tokens": total_tokens,
        "elapsed_sec": round(elapsed, 2),
        "tokens_per_sec": round(total_tokens / elapsed, 2) if elapsed > 0 else 0,
    }


def run_speculative_generation(
    decoder: SpeculativeDecoder,
    prompts: list[str],
    max_new_tokens: int = 64,
) -> dict:
    """Speculative decoding generation."""
    total_tokens = 0
    total_accepted = 0
    total_drafted = 0
    total_target_calls = 0
    t_start = time.time()

    for prompt in prompts:
        for result in decoder.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0):
            if result.get("finished"):
                metrics = result["metrics"]
                total_accepted += metrics.get("total_accepted", 0)
                total_target_calls += metrics.get("total_target_calls", 0)
                break
            total_tokens += len(result["tokens"])
            total_drafted += result["drafted"]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t_start

    return {
        "total_tokens": total_tokens,
        "elapsed_sec": round(elapsed, 2),
        "tokens_per_sec": round(total_tokens / elapsed, 2) if elapsed > 0 else 0,
        "acceptance_rate": round(total_accepted / max(total_drafted, 1) * 100, 1),
        "tokens_per_target_call": round(total_tokens / max(total_target_calls, 1), 2),
        "total_target_calls": total_target_calls,
    }


def run_speculative_benchmark(
    target_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    draft_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens: int = 64,
    num_speculative_tokens_list: list[int] = None,
) -> dict:
    """
    Run speculative decoding benchmark with different K values.
    
    Note: When draft == target (same model), acceptance rate will be ~100% for
    greedy decoding, but there's no speedup since both models are the same speed.
    Use a smaller draft model for actual speedup.
    """
    if num_speculative_tokens_list is None:
        num_speculative_tokens_list = [2, 4, 6, 8]

    prompts = PROMPTS[:3]
    results = {}

    print(f"Target model: {target_model_name}")
    print(f"Draft model:  {draft_model_name}")
    print(f"Prompts:      {len(prompts)}")
    print(f"Max tokens:   {max_new_tokens}")

    # Load target model
    print("\nLoading target model...")
    tokenizer = AutoTokenizer.from_pretrained(target_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    target_model.eval()

    # Baseline: standard generation
    print("\n--- Standard (autoregressive) ---")
    baseline = run_standard_generation(target_model, tokenizer, prompts, max_new_tokens)
    results["standard"] = baseline
    print(f"  {baseline['tokens_per_sec']:.2f} tok/s | {baseline['total_tokens']} tokens | {baseline['elapsed_sec']:.2f}s")

    # Speculative decoding with different K values
    for K in num_speculative_tokens_list:
        print(f"\n--- Speculative (K={K}) ---")
        try:
            spec_config = SpeculativeConfig(
                draft_model_name=draft_model_name,
                num_speculative_tokens=K,
            )
            decoder = SpeculativeDecoder(
                target_model=target_model,
                target_tokenizer=tokenizer,
                config=spec_config,
            )

            spec_result = run_speculative_generation(decoder, prompts, max_new_tokens)
            results[f"speculative_K{K}"] = spec_result

            speedup = spec_result["tokens_per_sec"] / baseline["tokens_per_sec"] if baseline["tokens_per_sec"] > 0 else 0
            print(f"  {spec_result['tokens_per_sec']:.2f} tok/s | "
                  f"Acceptance: {spec_result['acceptance_rate']}% | "
                  f"Tokens/call: {spec_result['tokens_per_target_call']:.2f} | "
                  f"Speedup: {speedup:.2f}x")

            # Clean up draft model to free memory
            del decoder

        except Exception as e:
            print(f"  ERROR: {e}")
            results[f"speculative_K{K}"] = {"status": "error", "error": str(e)}

    return results


if __name__ == "__main__":
    print("FlashServe-Lite: Speculative Decoding Benchmark")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    results = run_speculative_benchmark()

    os.makedirs("plots", exist_ok=True)
    with open("plots/speculative_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to plots/speculative_results.json")
