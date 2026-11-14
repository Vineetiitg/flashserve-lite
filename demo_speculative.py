#!/usr/bin/env python3
"""
FlashServe-Lite: Speculative Decoding Interactive Demo.

Visualizes the Draft-and-Verify cycle step-by-step:
  - Draft model proposes K candidate tokens
  - Target model verifies all K in a single forward pass
  - Accepted tokens are printed live with acceptance stats
"""

import argparse
import sys
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Compatibility patch
try:
    import torch.utils._pytree as _torch_pytree
    _torch_pytree._cxx_pytree_dynamo_traceable = False
except Exception:
    pass

from flashserve.speculative import SpeculativeDecoder, SpeculativeConfig


def main():
    parser = argparse.ArgumentParser(description="FlashServe-Lite Speculative Decoding Demo")
    parser.add_argument("--target-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Target (large) model ID (e.g. Qwen/Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--draft-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Draft (small) model ID")
    parser.add_argument("--prompt", type=str, default="Artificial intelligence and machine learning are", help="Input prompt")
    parser.add_argument("--k", type=int, default=4, help="Number of speculative tokens (K)")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max tokens to generate")
    args = parser.parse_args()

    print()
    print("  🔮 FlashServe-Lite — Speculative Decoding Visualizer")
    print("  ─────────────────────────────────────────────────────────────")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device:       {device.upper()}")
    print(f"  Lookahead K:  {args.k} candidate tokens per step")
    print(f"  Target Model: {args.target_model}")
    print(f"  Draft Model:  {args.draft_model}")
    print()

    print("  Loading models into GPU...", end=" ", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )
    target_model.eval()

    config = SpeculativeConfig(
        draft_model_name=args.draft_model,
        num_speculative_tokens=args.k,
        draft_device=device,
        draft_dtype="float16" if device == "cuda" else "float32",
    )
    decoder = SpeculativeDecoder(target_model, tokenizer, config, target_device=device)
    print(f"ready ({time.time() - t0:.1f}s)")
    print()

    print(f"  Prompt: \"{args.prompt}\"")
    print("  ─────────────────────────────────────────────────────────────")
    print("  Step-by-Step Draft & Verify Execution:")
    print()

    step_num = 1
    total_tokens = 0
    total_accepted = 0
    total_drafted = 0
    total_target_calls = 0

    t_start = time.time()
    for res in decoder.generate(args.prompt, max_new_tokens=args.max_tokens, temperature=0.0):
        if res.get("finished"):
            break

        tokens = res["tokens"]
        accepted_count = res["accepted"]
        drafted_count = res["drafted"]

        total_tokens += len(tokens)
        total_accepted += accepted_count
        total_drafted += drafted_count
        total_target_calls += 1

        token_str = "".join(tokens)
        status_badge = f"✓ {accepted_count}/{drafted_count} accepted"
        print(f"   [Step {step_num:>2}] Target Call #{total_target_calls} | {status_badge:>18} | Emitted: {token_str!r}")
        step_num += 1

    elapsed = time.time() - t_start
    print()
    print("  ─────────────────────────────────────────────────────────────")
    print("  📊 Speculative Decoding Summary:")
    print(f"     • Total Generated Tokens:    {total_tokens}")
    print(f"     • Target Model GPU Calls:    {total_target_calls}")
    print(f"     • Tokens / Target Call:      {total_tokens / max(total_target_calls, 1):.2f}x (effective speedup)")
    print(f"     • Draft Acceptance Rate:     {(total_accepted / max(total_drafted, 1)) * 100:.1f}%")
    print(f"     • Total Time:                {elapsed:.2f}s ({total_tokens / max(elapsed, 0.001):.1f} tok/s)")
    print()


if __name__ == "__main__":
    main()
