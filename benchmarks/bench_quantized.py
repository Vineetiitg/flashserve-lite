"""
Quantization Benchmark.

Compares inference throughput and GPU memory across FP16, INT8, and INT4 modes
for the same model and prompt workload.
"""

import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flashserve.quantization import QuantizationMode, QuantizationConfig, get_quantization_kwargs
from benchmarks.workload import PROMPTS


def run_quantization_benchmark(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens: int = 64,
    modes: list[str] = None,
) -> dict:
    """
    Benchmark model loading time, inference speed, and memory for each quantization mode.
    
    Returns dict mapping mode -> {load_time, tokens_per_sec, gpu_memory_mb, ...}
    """
    if modes is None:
        modes = ["fp16", "int8", "int4"]

    results = {}
    prompts = PROMPTS[:3]  # Use 3 prompts for quick comparison

    for mode_str in modes:
        print(f"\n{'='*60}")
        print(f"  Benchmarking: {mode_str.upper()}")
        print(f"{'='*60}")

        try:
            mode = QuantizationMode(mode_str)
            quant_config = QuantizationConfig(mode=mode)
            load_kwargs = get_quantization_kwargs(quant_config)
            load_kwargs["trust_remote_code"] = True

            # Clear GPU memory from previous run
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Measure model loading time
            t_load_start = time.time()
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            model.eval()
            load_time = time.time() - t_load_start

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Measure GPU memory after loading
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                mem_after_load = torch.cuda.memory_allocated() / (1024 * 1024)
            else:
                mem_after_load = 0

            # Run inference on prompts
            total_tokens = 0
            t_infer_start = time.time()

            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt")
                input_ids = inputs["input_ids"]
                if torch.cuda.is_available():
                    input_ids = input_ids.to("cuda")

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        repetition_penalty=1.0,
                    )
                    generated = output_ids.shape[1] - input_ids.shape[1]
                    total_tokens += generated

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_time = time.time() - t_infer_start

            # Peak memory
            if torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
            else:
                peak_mem = 0

            tokens_per_sec = total_tokens / infer_time if infer_time > 0 else 0

            results[mode_str] = {
                "load_time_sec": round(load_time, 2),
                "tokens_per_sec": round(tokens_per_sec, 2),
                "total_tokens": total_tokens,
                "inference_time_sec": round(infer_time, 2),
                "model_memory_mb": round(mem_after_load, 2),
                "peak_memory_mb": round(peak_mem, 2),
                "status": "success",
            }

            print(f"  Load time:     {load_time:.2f}s")
            print(f"  Throughput:    {tokens_per_sec:.2f} tok/s")
            print(f"  Model memory:  {mem_after_load:.0f} MB")
            print(f"  Peak memory:   {peak_mem:.0f} MB")
            print(f"  Total tokens:  {total_tokens}")

            # Clean up to free GPU memory for next mode
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            results[mode_str] = {
                "status": "error",
                "error": str(e),
            }

    return results


if __name__ == "__main__":
    print("FlashServe-Lite: Quantization Benchmark")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    results = run_quantization_benchmark()

    # Save results
    os.makedirs("plots", exist_ok=True)
    with open("plots/quantization_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for mode, data in results.items():
        if data.get("status") == "success":
            print(f"  {mode.upper():>5}: {data['tokens_per_sec']:>7.2f} tok/s | "
                  f"{data['peak_memory_mb']:>7.0f} MB peak | "
                  f"{data['load_time_sec']:>5.2f}s load")
        else:
            print(f"  {mode.upper():>5}: FAILED — {data.get('error', 'unknown')}")
