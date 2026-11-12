"""
FastAPI Server with OpenAI-Compatible Completions API, SSE Streaming,
Prometheus-style metrics, and an embedded React Web UI.
"""

import asyncio
from contextlib import asynccontextmanager
import json
import os
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from pydantic import BaseModel, Field

from flashserve.config import EngineConfig
from flashserve.engine import InferenceEngine
from flashserve.scheduler import ContinuousBatchingScheduler


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0, le=1000)
    stream: bool = Field(default=True)


# ── Global State ───────────────────────────────────────────────────
scheduler: Optional[ContinuousBatchingScheduler] = None
scheduler_task: Optional[asyncio.Task] = None
engine_config: Optional[EngineConfig] = None

# Prometheus-style counters
_metrics = {
    "requests_total": 0,
    "tokens_generated_total": 0,
    "errors_total": 0,
    "start_time": time.time(),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: Initialize engine and start continuous batching loop."""
    global scheduler, scheduler_task, engine_config
    engine_config = EngineConfig()
    engine = InferenceEngine(engine_config)
    scheduler = ContinuousBatchingScheduler(engine, engine_config)

    scheduler_task = asyncio.create_task(scheduler.run_loop())
    yield

    if scheduler:
        scheduler.is_running = False
    if scheduler_task:
        scheduler_task.cancel()
        try:
            await scheduler_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="FlashServe-Lite Engine",
    description="High-Throughput Mini LLM Serving Gateway with Paged KV Cache and Continuous Batching",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount Web UI static files ──────────────────────────────────────
WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
STATIC_DIR = os.path.join(WEB_DIR, "static")

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Web UI Route ───────────────────────────────────────────────────
@app.get("/")
async def serve_ui():
    """Serve the FlashServe-Lite web UI."""
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(
            index_path,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    return JSONResponse(
        content={"message": "FlashServe-Lite API is running. Web UI not found."},
        status_code=200,
    )


# ── Completions Endpoint ───────────────────────────────────────────
@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """OpenAI-compatible completions endpoint supporting SSE token streaming."""
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")

    _metrics["requests_total"] += 1

    seq_id, output_queue = scheduler.add_request(
        prompt_text=request.prompt,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
    )

    created_timestamp = int(time.time())

    # Streaming mode
    if request.stream:
        async def event_generator():
            try:
                while True:
                    token_text = await output_queue.get()
                    if token_text is None:
                        # End of sequence
                        final_chunk = {
                            "id": f"cmpl-{seq_id}",
                            "object": "text_completion",
                            "created": created_timestamp,
                            "choices": [
                                {
                                    "text": "",
                                    "index": 0,
                                    "logprobs": None,
                                    "finish_reason": "stop",
                                }
                            ],
                        }
                        yield f"data: {json.dumps(final_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        break

                    _metrics["tokens_generated_total"] += 1
                    chunk = {
                        "id": f"cmpl-{seq_id}",
                        "object": "text_completion",
                        "created": created_timestamp,
                        "choices": [
                            {
                                "text": token_text,
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
            finally:
                scheduler.output_queues.pop(seq_id, None)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming mode
    accumulated_tokens: list[str] = []
    try:
        while True:
            token = await output_queue.get()
            if token is None:
                break
            accumulated_tokens.append(token)
            _metrics["tokens_generated_total"] += 1
    finally:
        scheduler.output_queues.pop(seq_id, None)

    full_text = "".join(accumulated_tokens)
    return JSONResponse(
        content={
            "id": f"cmpl-{seq_id}",
            "object": "text_completion",
            "created": created_timestamp,
            "choices": [
                {
                    "text": full_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "completion_tokens": len(accumulated_tokens),
            },
        }
    )


# ── Model Info Endpoint ────────────────────────────────────────────
@app.get("/v1/models")
async def list_models():
    """Returns loaded model information."""
    if engine_config is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    return {
        "data": [
            {
                "id": engine_config.model_name,
                "object": "model",
                "owned_by": "flashserve-lite",
                "quantization": engine_config.quantization,
                "speculative_decoding": engine_config.enable_speculative,
            }
        ]
    }


# ── Health Check ───────────────────────────────────────────────────
@app.get("/v1/health")
async def health_check():
    """Engine readiness check."""
    return {
        "status": "healthy",
        "engine": "flashserve-lite",
        "version": "0.2.0",
        "quantization": engine_config.quantization if engine_config else "unknown",
    }


# ── Stats Endpoint ─────────────────────────────────────────────────
@app.get("/v1/stats")
async def engine_stats():
    """Real-time engine and memory telemetry."""
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    stats = scheduler.get_stats()
    if engine_config:
        stats["model"] = engine_config.model_name
        stats["dtype"] = engine_config.dtype
        stats["device"] = engine_config.device
        stats["quantization"] = getattr(engine_config, "quantization", "fp16")
    return stats


# ── Prometheus-Style Metrics ───────────────────────────────────────
@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus-compatible metrics endpoint for monitoring and observability."""
    import torch

    uptime = time.time() - _metrics["start_time"]
    gpu_mem_allocated = 0
    gpu_mem_reserved = 0
    gpu_utilization = 0

    if torch.cuda.is_available():
        gpu_mem_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        gpu_mem_reserved = torch.cuda.memory_reserved() / (1024 * 1024)

    scheduler_stats = scheduler.get_stats() if scheduler else {}
    kv_stats = scheduler_stats.get("kv_cache", {})

    lines = [
        "# HELP flashserve_requests_total Total number of completion requests received",
        "# TYPE flashserve_requests_total counter",
        f'flashserve_requests_total {_metrics["requests_total"]}',
        "",
        "# HELP flashserve_tokens_generated_total Total tokens generated across all requests",
        "# TYPE flashserve_tokens_generated_total counter",
        f'flashserve_tokens_generated_total {_metrics["tokens_generated_total"]}',
        "",
        "# HELP flashserve_errors_total Total number of errors",
        "# TYPE flashserve_errors_total counter",
        f'flashserve_errors_total {_metrics["errors_total"]}',
        "",
        "# HELP flashserve_uptime_seconds Engine uptime in seconds",
        "# TYPE flashserve_uptime_seconds gauge",
        f"flashserve_uptime_seconds {uptime:.2f}",
        "",
        "# HELP flashserve_running_sequences Current number of actively decoding sequences",
        "# TYPE flashserve_running_sequences gauge",
        f'flashserve_running_sequences {scheduler_stats.get("running_sequences", 0)}',
        "",
        "# HELP flashserve_waiting_requests Current number of queued requests",
        "# TYPE flashserve_waiting_requests gauge",
        f'flashserve_waiting_requests {scheduler_stats.get("waiting_requests", 0)}',
        "",
        "# HELP flashserve_kv_cache_utilization KV cache block utilization percentage",
        "# TYPE flashserve_kv_cache_utilization gauge",
        f'flashserve_kv_cache_utilization {kv_stats.get("utilization_pct", 0)}',
        "",
        "# HELP flashserve_gpu_memory_allocated_mb GPU memory allocated in MB",
        "# TYPE flashserve_gpu_memory_allocated_mb gauge",
        f"flashserve_gpu_memory_allocated_mb {gpu_mem_allocated:.2f}",
        "",
        "# HELP flashserve_gpu_memory_reserved_mb GPU memory reserved in MB",
        "# TYPE flashserve_gpu_memory_reserved_mb gauge",
        f"flashserve_gpu_memory_reserved_mb {gpu_mem_reserved:.2f}",
    ]

    return StreamingResponse(
        iter(["\n".join(lines) + "\n"]),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
