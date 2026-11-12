"""
Realistic workload prompt generator with varied context lengths.
"""

PROMPTS = [
    # Short prompts (~10-20 tokens)
    "Explain what a neural network is in one sentence.",
    "What is the capital of France and what is it known for?",
    "Write a concise haiku about low-latency systems programming.",
    "Define what time-to-first-token means in machine learning serving.",

    # Medium prompts (~30-60 tokens)
    "Explain the core difference between TCP and UDP network transport protocols. Detail when an engineer should choose each protocol, citing video streaming vs banking as examples.",
    "Write a clean Python function that performs iterative binary search on a sorted list. Include comprehensive docstrings, type annotations, and edge case handling.",
    "Describe how virtual memory paging works in modern operating systems. Explain the roles of page tables, translation lookaside buffers, and page faults.",

    # Long prompts (~80-150 tokens)
    "You are a principal infrastructure architect conducting a technical design review. The team proposes building an LLM inference serving layer using dynamic KV-cache memory allocation and continuous batching. Critique this architecture with respect to internal memory fragmentation, GPU memory bandwidth saturation, scheduling preemption, and latency tail percentiles (p95/p99). Provide concrete recommendations.",
    "Analyze the trade-offs between dense and sparse attention mechanisms in modern transformer architectures. How do grouped-query attention (GQA) and multi-head latent attention (MLA) reduce KV-cache memory footprints during high-concurrency decoding phases? Contrast their memory bandwidth demands on consumer GPUs.",
]


def generate_workload(num_requests: int) -> list[str]:
    """Generates a reproducible sequence of prompts cycling through short, medium, and long contexts."""
    return [PROMPTS[i % len(PROMPTS)] for i in range(num_requests)]
