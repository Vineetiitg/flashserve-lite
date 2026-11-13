"""
Master Benchmark Orchestrator & Chart Generation.

Runs HuggingFace baseline and FlashServe-Lite engine across concurrency levels.
Saves raw JSON telemetry and renders publication-ready comparison charts into plots/.
"""

import asyncio
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from benchmarks.bench_hf_baseline import run_hf_baseline
from benchmarks.bench_flashserve import run_flashserve_benchmark


CONCURRENCY_LEVELS = [1, 2, 4, 8]
MAX_NEW_TOKENS = 64
OUTPUT_DIR = "plots"


def generate_charts(results: dict, concurrency: list[int], output_dir: str):
    """Renders high-res comparison charts."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Throughput Comparison Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    hf_throughput = [r["throughput_tokens_per_sec"] for r in results["hf"]]
    fs_throughput = [r["throughput_tokens_per_sec"] for r in results["flashserve"]]

    ax.plot(concurrency, hf_throughput, marker="o", color="#ef4444", linewidth=2.5, label="HuggingFace generate() [Sequential]")
    ax.plot(concurrency, fs_throughput, marker="s", color="#3b82f6", linewidth=2.5, label="FlashServe-Lite [Continuous Batching + Paged KV]")

    ax.set_title("Inference Serving Throughput vs Concurrency (Qwen2.5-0.5B FP16)", fontsize=12, pad=12)
    ax.set_xlabel("Concurrent Client Streams", fontsize=11)
    ax.set_ylabel("Aggregate Throughput (Tokens / Sec)", fontsize=11)
    ax.set_xticks(concurrency)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="#f8fafc", loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "throughput_vs_concurrency.png"), dpi=200)
    plt.close()

    # 2. Latency Distribution Chart (P50 and P95 Inter-Token Latency)
    fig, ax = plt.subplots(figsize=(8, 5))
    fs_p50 = [r["itl_p50_ms"] for r in results["flashserve"]]
    fs_p95 = [r["itl_p95_ms"] for r in results["flashserve"]]

    ax.plot(concurrency, fs_p50, marker="o", color="#2563eb", linewidth=2, label="FlashServe-Lite Inter-Token Latency p50 (ms)")
    ax.plot(concurrency, fs_p95, marker="^", color="#f59e0b", linewidth=2, linestyle="--", label="FlashServe-Lite Inter-Token Latency p95 (ms)")

    ax.set_title("FlashServe-Lite Latency Percentiles vs Concurrency", fontsize=12, pad=12)
    ax.set_xlabel("Concurrent Client Streams", fontsize=11)
    ax.set_ylabel("Latency (ms / token)", fontsize=11)
    ax.set_xticks(concurrency)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="#f8fafc", loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_vs_concurrency.png"), dpi=200)
    plt.close()

    # 3. Peak GPU Memory Usage
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_width = 0.35
    x_indices = list(range(len(concurrency)))

    hf_mem = [r["gpu_peak_memory_mb"] for r in results["hf"]]
    fs_mem = [r["gpu_peak_memory_mb"] for r in results["flashserve"]]

    ax.bar([x - bar_width / 2 for x in x_indices], hf_mem, width=bar_width, color="#f87171", label="HuggingFace Peak VRAM (MB)")
    ax.bar([x + bar_width / 2 for x in x_indices], fs_mem, width=bar_width, color="#60a5fa", label="FlashServe-Lite Peak VRAM (MB)")

    ax.set_title("Peak GPU Memory Footprint on RTX 3050 6GB", fontsize=12, pad=12)
    ax.set_xlabel("Concurrent Client Streams", fontsize=11)
    ax.set_ylabel("Peak VRAM Allocated (MB)", fontsize=11)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(concurrency)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(frameon=True, facecolor="#f8fafc", loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_vs_concurrency.png"), dpi=200)
    plt.close()

    # 4. Multi-Panel Comprehensive Summary Dashboard
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: Throughput
    axes[0].plot(concurrency, hf_throughput, marker="o", color="#ef4444", label="HuggingFace")
    axes[0].plot(concurrency, fs_throughput, marker="s", color="#3b82f6", label="FlashServe-Lite")
    axes[0].set_title("Throughput Scaling")
    axes[0].set_xlabel("Concurrency")
    axes[0].set_ylabel("Tokens / Sec")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Panel B: FlashServe ITL
    axes[1].plot(concurrency, fs_p50, marker="o", color="#2563eb", label="p50 ITL")
    axes[1].plot(concurrency, fs_p95, marker="^", color="#f59e0b", linestyle="--", label="p95 ITL")
    axes[1].set_title("Inter-Token Latency (ITL)")
    axes[1].set_xlabel("Concurrency")
    axes[1].set_ylabel("ms")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    # Panel C: Memory
    axes[2].bar([x - bar_width / 2 for x in x_indices], hf_mem, width=bar_width, color="#f87171", label="HuggingFace")
    axes[2].bar([x + bar_width / 2 for x in x_indices], fs_mem, width=bar_width, color="#60a5fa", label="FlashServe-Lite")
    axes[2].set_title("Peak VRAM Footprint")
    axes[2].set_xlabel("Concurrency")
    axes[2].set_ylabel("MB")
    axes[2].set_xticks(x_indices)
    axes[2].set_xticklabels(concurrency)
    axes[2].grid(True, linestyle="--", alpha=0.5, axis="y")
    axes[2].legend()

    plt.suptitle("FlashServe-Lite vs HuggingFace Benchmark Suite (Qwen2.5-0.5B)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_dashboard.png"), dpi=200)
    plt.close()


def main():
    print("=" * 70)
    print("  FlashServe-Lite Master Benchmark Suite")
    print(f"  Target Concurrency: {CONCURRENCY_LEVELS} | Max New Tokens: {MAX_NEW_TOKENS}")
    print("=" * 70)

    results = {"hf": [], "flashserve": []}

    for n in CONCURRENCY_LEVELS:
        print(f"\n[>>> Running Concurrency Level = {n} <<<]")

        # 1. HuggingFace baseline
        print(f"  [1/2] Benchmarking HuggingFace generate() (N={n})...")
        hf_res = run_hf_baseline(num_requests=n, max_new_tokens=MAX_NEW_TOKENS)
        results["hf"].append(hf_res)
        print(f"        HF Throughput:  {hf_res['throughput_tokens_per_sec']} tok/s | Peak VRAM: {hf_res['gpu_peak_memory_mb']} MB")

        # 2. FlashServe-Lite
        print(f"  [2/2] Benchmarking FlashServe-Lite (N={n})...")
        fs_res = asyncio.run(
            run_flashserve_benchmark(
                num_requests=n,
                max_new_tokens=MAX_NEW_TOKENS,
                max_batch_size=max(n, 4),
            )
        )
        results["flashserve"].append(fs_res)
        print(f"        FS Throughput:  {fs_res['throughput_tokens_per_sec']} tok/s | TTFT p50: {fs_res['ttft_p50_ms']} ms | Peak VRAM: {fs_res['gpu_peak_memory_mb']} MB")

    # Save JSON data
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR, "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw benchmark data to: {json_path}")

    # Render charts
    print("Generating benchmark figures in plots/...")
    generate_charts(results, CONCURRENCY_LEVELS, OUTPUT_DIR)
    print("Charts successfully saved to plots/ directory.")


if __name__ == "__main__":
    main()
