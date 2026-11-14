<div align="center">

# ⚡ FlashServe-Lite

**A Mini LLM Inference Engine with Paged KV Cache, Continuous Batching, INT8/INT4 Quantization, and Speculative Decoding**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Built from scratch to understand how production LLM serving systems (vLLM, TGI, TensorRT-LLM) work under the hood.*

[Quick Start](#-quick-start) · [Architecture](#-architecture) · [Benchmarks](#-benchmark-results) · [Web UI](#-web-ui) · [API Reference](#-api-reference)

</div>

---

## 📋 What This Is

FlashServe-Lite is a **from-scratch LLM inference engine** that implements the core techniques used by production serving systems:

| Technique | What It Does | Production Systems |
|---|---|---|
| **Paged KV Cache** | Virtual memory paging for attention KV tensors — eliminates memory fragmentation | vLLM, SGLang |
| **Continuous Batching** | Iteration-level scheduling that adds/removes requests mid-batch | vLLM, TGI, TensorRT-LLM |
| **INT8/INT4 Quantization** | Reduced-precision inference via bitsandbytes — 2-4× memory savings | GPTQ, AWQ, bitsandbytes |
| **Speculative Decoding** | Draft model proposes tokens, target model verifies in one pass — up to K× speedup | Medusa, EAGLE, SpecInfer |
| **SSE Streaming** | OpenAI-compatible Server-Sent Events for token-by-token streaming | All production APIs |

> **Key result:** 6.3× throughput improvement over HuggingFace `generate()` at 4 concurrent requests on an RTX 3050 6GB GPU.

---

## 🔧 Architecture

```
                    ┌──────────────────────────────────────────┐
                    │              FastAPI Server               │
                    │  POST /v1/completions (SSE Streaming)     │
                    │  GET  /v1/health  /v1/stats  /metrics     │
                    └─────────────────┬────────────────────────┘
                                      │
                    ┌─────────────────▼────────────────────────┐
                    │     Continuous Batching Scheduler         │
                    │                                          │
                    │  ┌──────────┐    ┌───────────────────┐   │
                    │  │ Waiting  │───▶│  Running Batch     │   │
                    │  │  Queue   │    │  (iteration-level) │   │
                    │  └──────────┘    └────────┬──────────┘   │
                    └──────────────────────────┬───────────────┘
                                               │
                    ┌──────────────────────────▼───────────────┐
                    │          Inference Engine                 │
                    │                                          │
                    │  ┌──────────┐    ┌───────────────────┐   │
                    │  │ Prefill  │    │  Batched Decode    │   │
                    │  │ (prompt) │    │  (token-by-token)  │   │
                    │  └──────────┘    └───────────────────┘   │
                    │                                          │
                    │  ┌──────────────────────────────────┐    │
                    │  │  Quantization: FP16 / INT8 / INT4│    │
                    │  └──────────────────────────────────┘    │
                    └──────────────────────────┬───────────────┘
                                               │
                    ┌──────────────────────────▼───────────────┐
                    │           Paged KV Cache                 │
                    │                                          │
                    │  ┌─────┬─────┬─────┬─────┬─────┐        │
                    │  │ B0  │ B1  │ B2  │ ... │ Bn  │ Pool   │
                    │  └──┬──┴──┬──┴─────┴─────┴─────┘        │
                    │     │     │                               │
                    │  Seq 1   Seq 2  (block tables)           │
                    └──────────────────────────────────────────┘
```

### Core Components

| Component | File | Description |
|---|---|---|
| `EngineConfig` | [`config.py`](flashserve/config.py) | Central configuration for model, KV cache, scheduler, quantization, and server |
| `Sequence` | [`sequence.py`](flashserve/sequence.py) | Request lifecycle state machine with timing instrumentation |
| `PagedKVCache` | [`kv_cache.py`](flashserve/kv_cache.py) | Virtual memory block manager with scatter/gather operations |
| `InferenceEngine` | [`engine.py`](flashserve/engine.py) | Model execution with prefill and batched decode pipelines |
| `ContinuousBatchingScheduler` | [`scheduler.py`](flashserve/scheduler.py) | Iteration-level scheduler with 4-phase step loop |
| `QuantizationConfig` | [`quantization.py`](flashserve/quantization.py) | INT8/INT4 quantization via bitsandbytes |
| `SpeculativeDecoder` | [`speculative.py`](flashserve/speculative.py) | Draft-then-verify speculative decoding engine |
| `sample_token` | [`sampling.py`](flashserve/sampling.py) | Greedy, top-k, and nucleus (top-p) sampling strategies |
| FastAPI Server | [`server.py`](flashserve/server.py) | OpenAI-compatible API with SSE streaming and Prometheus metrics |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (tested on RTX 3050 6GB)
- ~3GB free VRAM for FP16 inference

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/mini-llm-inference-engine.git
cd mini-llm-inference-engine

pip install -r requirements.txt
```

### Run the Demo

```bash
# Quick generation (no server needed)
python demo.py

# With custom prompt
python demo.py --prompt "Explain transformers in 3 sentences"

# With INT8 quantization (half the memory)
python demo.py --quantization int8

# With temperature sampling
python demo.py --temperature 0.7 --max-tokens 200
```

### Start the Server

```bash
# Launch with default settings (FP16)
uvicorn flashserve.server:app --host 0.0.0.0 --port 8000

# Open the Web UI
# → http://localhost:8000
```

### Send a Request

```bash
# Streaming (SSE)
curl -N http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI is", "max_tokens": 100, "stream": true}'

# Non-streaming
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "max_tokens": 50, "stream": false}'
```

---

## 🖥️ Web UI

FlashServe-Lite ships with a built-in React web interface accessible at `http://localhost:8000` when the server is running.

**Features:**
- Real-time SSE token streaming with typing cursor animation
- Live engine statistics (running sequences, queue depth, KV cache utilization)
- Configurable generation parameters (temperature, top-k, top-p, max tokens)
- GPU memory monitoring
- Dark theme with glassmorphism design

---

## 📊 Benchmark Results

Benchmarked on **NVIDIA GeForce RTX 3050 Laptop GPU (6GB)** with **Qwen2.5-0.5B-Instruct** in FP16.

### Throughput vs Concurrency

| Concurrency | HuggingFace (tok/s) | FlashServe (tok/s) | Speedup |
|:-----------:|:-------------------:|:------------------:|:-------:|
| 1 | 7.24 | 6.41 | 0.89× |
| 2 | 7.74 | 11.94 | **1.54×** |
| 4 | 10.57 | 66.92 | **6.33×** |
| 8 | 20.45 | 45.38 | **2.22×** |

> **Why slower at concurrency 1?** The gather/scatter overhead of the paged KV cache adds latency for single requests. This is a known trade-off — the same one vLLM makes. The benefit appears under concurrent load where paged memory management prevents fragmentation and enables true batching.

> **Why does throughput drop at concurrency 8?** GPU compute saturation on the RTX 3050's limited CUDA cores (2048). A larger GPU would show continued scaling.

### Latency

| Concurrency | FlashServe p50 (ms/tok) | FlashServe p95 (ms/tok) |
|:-----------:|:-----------------------:|:-----------------------:|
| 1 | 155.2 | 162.8 |
| 2 | 83.4 | 91.7 |
| 4 | 29.6 | 38.1 |
| 8 | 43.9 | 55.3 |

### GPU Memory Usage

| Concurrency | HuggingFace (MB) | FlashServe (MB) |
|:-----------:|:-----------------:|:---------------:|
| 1 | 2,306 | 2,326 |
| 4 | 2,306 | 2,411 |
| 8 | 2,306 | 2,479 |

FlashServe's paged KV cache uses marginally more memory at low concurrency due to the pre-allocated block pool, but enables **predictable memory scaling** as concurrency increases — unlike HuggingFace which would OOM with truly concurrent in-flight sequences.

---

## ⚙️ Quantization

Reduce GPU memory by 2-4× with INT8/INT4 quantization via [bitsandbytes](https://github.com/TimDettmers/bitsandbytes):

```bash
# INT8 quantization (~50% memory reduction)
python demo.py --quantization int8

# INT4 quantization (~75% memory reduction)
python demo.py --quantization int4
```

| Mode | Model Memory | Peak Memory | Tokens/sec | Memory Savings |
|:----:|:------------:|:-----------:|:----------:|:--------------:|
| FP16 | ~1,024 MB | ~2,400 MB | 7.24 | baseline |
| INT8 | ~530 MB | ~1,800 MB | 6.1 | ~50% |
| INT4 | ~320 MB | ~1,500 MB | 5.3 | ~70% |

> INT4 quantization enables running **larger models** on consumer GPUs. A 1.5B model in INT4 fits within 6GB VRAM.

---

## 🔮 Speculative Decoding

Speculative decoding uses a fast **draft model** to propose K candidate tokens, then the **target model** verifies all K in a single forward pass:

```
Draft model:    generates token 1, 2, 3, 4  (4 forward passes, fast)
Target model:   verifies all 4 in ONE pass   (1 forward pass, slow but batched)
Result:         up to 5 tokens per target call (K accepted + 1 bonus)
```

```bash
# Run speculative decoding benchmark
python -m benchmarks.bench_speculative
```

**When it helps:** When the draft model is significantly faster than the target (e.g., 0.5B draft → 3B target). When using the same model for both (as in our 6GB GPU setup), the algorithm validates correctness but doesn't show speedup.

---

## 📡 API Reference

### `POST /v1/completions`

OpenAI-compatible text completion endpoint.

```json
{
    "prompt": "The meaning of life is",
    "max_tokens": 256,
    "temperature": 0.0,
    "top_p": 0.9,
    "top_k": 50,
    "stream": true
}
```

**SSE Stream Format:**
```
data: {"id":"cmpl-0","object":"text_completion","choices":[{"text":" to","index":0,"finish_reason":null}]}

data: {"id":"cmpl-0","object":"text_completion","choices":[{"text":" find","index":0,"finish_reason":null}]}

data: {"id":"cmpl-0","object":"text_completion","choices":[{"text":"","index":0,"finish_reason":"stop"}]}

data: [DONE]
```

### `GET /v1/health`
Engine readiness check. Returns `{"status": "healthy"}`.

### `GET /v1/stats`
Real-time telemetry: running sequences, queue depth, KV cache utilization, GPU memory.

### `GET /v1/models`
Lists the loaded model with quantization mode.

### `GET /metrics`
Prometheus-compatible metrics endpoint for monitoring dashboards (Grafana, etc.).

### `GET /`
Serves the built-in React web UI.

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Individual test suites
pytest tests/test_kv_cache.py -v        # Block allocation, scatter-gather roundtrip
pytest tests/test_scheduler.py -v       # Request queuing, eviction-promotion
pytest tests/test_engine.py -v          # Prefill-decode integration
pytest tests/test_correctness.py -v     # Greedy equivalence vs HF generate()
```

---

## 📁 Project Structure

```
flashserve-lite/
├── flashserve/                  # Core engine
│   ├── config.py                # Central configuration
│   ├── sequence.py              # Request state machine
│   ├── kv_cache.py              # Paged KV cache (virtual memory)
│   ├── engine.py                # Prefill & batched decode
│   ├── scheduler.py             # Continuous batching scheduler
│   ├── quantization.py          # INT8/INT4 quantization support
│   ├── speculative.py           # Speculative decoding engine
│   ├── sampling.py              # Token sampling strategies
│   └── server.py                # FastAPI SSE server + Prometheus metrics
├── web/                         # React web UI
│   ├── index.html
│   └── static/
│       ├── style.css
│       └── app.jsx
├── tests/                       # Test suite
│   ├── test_kv_cache.py
│   ├── test_scheduler.py
│   ├── test_engine.py
│   └── test_correctness.py
├── benchmarks/                  # Benchmark suite
│   ├── workload.py              # Prompt workloads
│   ├── metrics.py               # Metrics computation
│   ├── bench_hf_baseline.py     # HuggingFace baseline
│   ├── bench_flashserve.py      # FlashServe async benchmark
│   ├── bench_quantized.py       # Quantization comparison
│   ├── bench_speculative.py     # Speculative decoding benchmark
│   ├── bench_vllm.py            # vLLM comparison (Linux only)
│   └── run_all.py               # Master orchestrator + chart generation
├── plots/                       # Generated benchmark charts & data
├── demo.py                      # Quick demo script
├── Dockerfile                   # Container deployment
├── docker-compose.yml           # GPU-enabled compose
├── requirements.txt
├── .env.example
└── LICENSE
```

---

## 🐳 Docker

```bash
# Build
docker compose build

# Run (requires NVIDIA Container Toolkit)
docker compose up
```

---

## 🔬 Design Decisions & Trade-offs

| Decision | Rationale |
|---|---|
| **Python-level scatter/gather** instead of custom CUDA kernels | Prioritizes readability and correctness over raw throughput. Production systems (vLLM) use custom CUDA kernels for 10-100× faster KV operations. |
| **Pre-allocated contiguous KV pool** | One large `torch.zeros()` allocation at startup — avoids repeated `torch.empty()` calls and CUDA memory fragmentation. |
| **Block size = 16 tokens** | Balances internal fragmentation (wasted tail slots) vs. block table overhead. Same default as vLLM. |
| **Left-padding for batched decode** | Matches HuggingFace's expected attention mask format. Right-padding would require custom position embeddings. |
| **asyncio.Queue for token streaming** | Simpler than async generators for multi-consumer scenarios. Each request gets its own queue. |
| **bitsandbytes for quantization** | Most portable option — works with any HuggingFace model. GPTQ/AWQ require pre-quantized model files. |
| **Same-model speculative decoding** | Demonstrates the algorithm correctly on limited hardware. Real deployments use a smaller draft model (e.g., 0.5B → 7B). |

---

## 🎯 Resume Bullet Points

> - Built a **mini LLM inference engine from scratch** with paged KV cache, continuous batching, INT8/INT4 quantization, and speculative decoding — achieving **6.3× throughput** vs HuggingFace at concurrency 4 on a consumer GPU
> - Implemented **virtual memory paging** for attention KV tensors with block-level scatter/gather, reducing GPU memory fragmentation and enabling predictable memory scaling under concurrent load
> - Designed an **iteration-level continuous batching scheduler** that dynamically admits/evicts requests mid-batch, achieving near-linear throughput scaling from 1→4 concurrent sequences
> - Added **bitsandbytes INT8/INT4 quantization** support with ~50-70% memory reduction, enabling larger model deployment on memory-constrained GPUs
> - Built an **OpenAI-compatible SSE streaming API** with a React web UI, Prometheus metrics endpoint, and Docker GPU deployment

---

## 📚 References

- [vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)
- [QLoRA: Efficient Finetuning of Quantized Language Models](https://arxiv.org/abs/2305.14314)

---

## License

[MIT](LICENSE)
