"""
Continuous Batching Scheduler.

Implements iteration-level scheduling:
  - Dynamically evicts completed sequences on every iteration step.
  - Promotes waiting requests and executes prefill as soon as KV cache slots become available.
  - Runs batched decode across all active sequences simultaneously.
  - Dispatches generated token events to asynchronous streaming queues.
"""

import asyncio
import time
from typing import Optional

from flashserve.config import EngineConfig
from flashserve.engine import InferenceEngine
from flashserve.sequence import Sequence, SequenceStatus


class ContinuousBatchingScheduler:
    """Orchestrates request life-cycles and iteration-level batch execution."""

    def __init__(self, engine: InferenceEngine, config: EngineConfig):
        self.engine = engine
        self.config = config

        # Request management queues
        self.waiting_queue: list[Sequence] = []
        self.running_batch: dict[int, Sequence] = {}  # seq_id -> Sequence
        self.next_seq_id: int = 0
        self.total_completed: int = 0

        # Asynchronous token output queues: seq_id -> asyncio.Queue
        self.output_queues: dict[int, asyncio.Queue] = {}

        # Tracking input token for next decode step: seq_id -> token_id
        self.next_input_tokens: dict[int, int] = {}

        # Execution flags
        self.is_running: bool = False
        self._step_event = asyncio.Event()

    def add_request(
        self,
        prompt_text: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> tuple[int, asyncio.Queue]:
        """
        Submits a new generation request into the engine pipeline.
        Returns unique sequence id and an asyncio.Queue streaming token strings.
        """
        token_ids = self.engine.tokenizer.encode(prompt_text, add_special_tokens=True)
        if len(token_ids) == 0:
            token_ids = [self.engine.tokenizer.eos_token_id or 151645]

        seq_id = self.next_seq_id
        self.next_seq_id += 1

        eos_id = self.engine.tokenizer.eos_token_id
        if eos_id is None:
            eos_id = 151645

        sequence = Sequence(
            seq_id=seq_id,
            prompt_token_ids=token_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            eos_token_id=eos_id,
        )

        output_queue: asyncio.Queue = asyncio.Queue()
        self.output_queues[seq_id] = output_queue
        self.waiting_queue.append(sequence)

        # Notify scheduler loop
        self._step_event.set()
        return seq_id, output_queue

    async def run_loop(self):
        """Continuous execution loop processing iterations."""
        self.is_running = True

        while self.is_running:
            if not self.running_batch and not self.waiting_queue:
                self._step_event.clear()
                try:
                    await asyncio.wait_for(self._step_event.wait(), timeout=0.1)
                except asyncio.TimeoutError:
                    pass
                continue

            await self._step()
            # Yield control to event loop to allow socket/SSE flushes
            await asyncio.sleep(0.0001)

    async def _step(self):
        """Single scheduler iteration: Evict -> Promote & Prefill -> Batched Decode -> Dispatch."""
        
        # ==========================================================
        # PHASE 1: Evict Completed Sequences
        # ==========================================================
        finished_ids = [
            sid for sid, seq in self.running_batch.items() if seq.is_finished
        ]
        for sid in finished_ids:
            seq = self.running_batch.pop(sid)
            seq.status = SequenceStatus.FINISHED
            seq.finish_time = time.time()
            self.total_completed += 1

            # Reclaim GPU KV Cache memory
            self.engine.free_sequence(sid)

            # Signal stream completion via None sentinel
            if sid in self.output_queues:
                await self.output_queues[sid].put(None)

            self.next_input_tokens.pop(sid, None)

        # ==========================================================
        # PHASE 2: Promote Waiting Sequences & Execute Prefill
        # ==========================================================
        while (
            self.waiting_queue
            and len(self.running_batch) < self.config.max_batch_size
        ):
            candidate = self.waiting_queue[0]
            needed_blocks = self.engine.kv_cache.get_num_required_blocks(
                candidate.prompt_length + candidate.max_new_tokens
            )

            if needed_blocks > self.engine.kv_cache.num_free_blocks:
                # KV memory pressure: wait until running sequences finish
                break

            sequence = self.waiting_queue.pop(0)
            sequence.status = SequenceStatus.RUNNING

            # Execute Prefill
            first_token_id = self.engine.prefill(sequence)
            sequence.first_token_time = time.time()
            sequence.generated_token_ids.append(first_token_id)

            token_text = self.engine.tokenizer.decode(
                [first_token_id], skip_special_tokens=True
            )
            if sequence.seq_id in self.output_queues:
                await self.output_queues[sequence.seq_id].put(token_text)

            self.next_input_tokens[sequence.seq_id] = first_token_id
            self.running_batch[sequence.seq_id] = sequence

        # ==========================================================
        # PHASE 3: Batched Decode
        # ==========================================================
        if not self.running_batch:
            return

        active_sequences = list(self.running_batch.values())
        active_input_tokens = [
            self.next_input_tokens[seq.seq_id] for seq in active_sequences
        ]

        # Batched forward pass on GPU
        next_tokens = self.engine.decode_batch(
            active_sequences, active_input_tokens
        )

        # ==========================================================
        # PHASE 4: Dispatch Outputs
        # ==========================================================
        for seq, next_token_id in zip(active_sequences, next_tokens):
            seq.generated_token_ids.append(next_token_id)
            self.next_input_tokens[seq.seq_id] = next_token_id

            token_text = self.engine.tokenizer.decode(
                [next_token_id], skip_special_tokens=True
            )
            if seq.seq_id in self.output_queues:
                await self.output_queues[seq.seq_id].put(token_text)

    def get_stats(self) -> dict:
        """Returns live scheduler telemetry and memory utilization."""
        return {
            "running_sequences": len(self.running_batch),
            "waiting_requests": len(self.waiting_queue),
            "total_completed": self.total_completed,
            "total_submitted": self.next_seq_id,
            "kv_cache": self.engine.kv_cache.get_usage_stats(),
        }
