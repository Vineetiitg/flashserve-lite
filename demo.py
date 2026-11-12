#!/usr/bin/env python3
"""
FlashServe-Lite Quick Demo.

Launches the inference engine, generates a completion, and prints tokens as they stream.
No server required — runs the engine directly in-process.

Usage:
    python demo.py
    python demo.py --prompt "Explain quantum computing"
    python demo.py --quantization int8
    python demo.py --max-tokens 128 --temperature 0.7
"""

import argparse
import asyncio
import sys
import time

import torch

from flashserve.config import EngineConfig
from flashserve.engine import InferenceEngine
from flashserve.scheduler import ContinuousBatchingScheduler


def print_banner():
    print()
    print("  ⚡ FlashServe-Lite — Mini LLM Inference Engine")
    print("  ─────────────────────────────────────────────────")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"  VRAM: {mem:.1f} GB")
    else:
        print("  Device: CPU (no CUDA GPU detected)")
    print()


async def run_demo(
    prompt: str,
    max_tokens: int,
    temperature: float,
    quantization: str,
):
    """Run a single generation request through the engine."""
    config = EngineConfig(
        quantization=quantization,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )

    print(f"  Model:        {config.model_name}")
    print(f"  Quantization: {quantization.upper()}")
    print(f"  Max tokens:   {max_tokens}")
    print(f"  Temperature:  {temperature}")
    print()

    # Initialize engine
    print("  Loading model...", end=" ", flush=True)
    t_load = time.time()
    engine = InferenceEngine(config)
    print(f"done ({time.time() - t_load:.1f}s)")

    if torch.cuda.is_available():
        mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"  GPU memory:   {mem_mb:.0f} MB")
    print()

    # Start scheduler
    scheduler = ContinuousBatchingScheduler(engine, config)
    loop_task = asyncio.create_task(scheduler.run_loop())

    # Submit request
    print(f"  Prompt: {prompt}")
    print(f"  {'─' * 56}")
    print("  Output: ", end="", flush=True)

    seq_id, queue = scheduler.add_request(
        prompt_text=prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )

    # Stream tokens
    token_count = 0
    t_start = time.time()
    t_first_token = None

    while True:
        token_text = await queue.get()
        if token_text is None:
            break
        if t_first_token is None:
            t_first_token = time.time()
        print(token_text, end="", flush=True)
        token_count += 1

    t_end = time.time()
    print()
    print(f"  {'─' * 56}")

    # Stats
    elapsed = t_end - t_start
    ttft = (t_first_token - t_start) if t_first_token else 0
    tok_per_sec = token_count / elapsed if elapsed > 0 else 0

    print(f"  Tokens:       {token_count}")
    print(f"  TTFT:         {ttft*1000:.0f} ms")
    print(f"  Throughput:   {tok_per_sec:.2f} tok/s")
    print(f"  Total time:   {elapsed:.2f}s")
    print()

    # Cleanup
    scheduler.is_running = False
    loop_task.cancel()
    try:
        await loop_task
    except asyncio.CancelledError:
        pass


def main():
    parser = argparse.ArgumentParser(description="FlashServe-Lite Quick Demo")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain how a paged KV cache works in LLM inference in 3 sentences.",
        help="Input prompt for generation",
    )
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 = greedy)")
    parser.add_argument("--quantization", type=str, default="fp16", choices=["fp16", "int8", "int4"], help="Quantization mode")
    args = parser.parse_args()

    print_banner()
    asyncio.run(run_demo(args.prompt, args.max_tokens, args.temperature, args.quantization))


if __name__ == "__main__":
    main()
