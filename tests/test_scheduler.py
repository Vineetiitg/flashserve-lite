"""
Unit tests for the ContinuousBatchingScheduler queue mechanics and sequence lifecycle.
"""

import asyncio
import pytest
from unittest.mock import MagicMock
from flashserve.config import EngineConfig
from flashserve.scheduler import ContinuousBatchingScheduler
from flashserve.sequence import SequenceStatus


@pytest.mark.asyncio
async def test_scheduler_add_request():
    # Mock engine
    mock_engine = MagicMock()
    mock_engine.tokenizer.encode.return_value = [10, 20, 30]
    mock_engine.tokenizer.eos_token_id = 151645

    config = EngineConfig(max_batch_size=4)
    scheduler = ContinuousBatchingScheduler(mock_engine, config)

    seq_id1, q1 = scheduler.add_request("Hello world", max_new_tokens=10)
    assert seq_id1 == 0
    assert len(scheduler.waiting_queue) == 1
    assert scheduler.waiting_queue[0].seq_id == 0
    assert scheduler.waiting_queue[0].status == SequenceStatus.WAITING

    seq_id2, q2 = scheduler.add_request("Second prompt", max_new_tokens=20)
    assert seq_id2 == 1
    assert len(scheduler.waiting_queue) == 2


@pytest.mark.asyncio
async def test_scheduler_eviction_and_promotion():
    mock_engine = MagicMock()
    mock_engine.tokenizer.encode.return_value = [10, 20]
    mock_engine.tokenizer.decode.return_value = " token"
    mock_engine.tokenizer.eos_token_id = 999

    # KV cache mock
    mock_engine.kv_cache.get_num_required_blocks.return_value = 2
    mock_engine.kv_cache.num_free_blocks = 100
    mock_engine.prefill.return_value = 100
    mock_engine.decode_batch.return_value = [999]  # EOS token

    config = EngineConfig(max_batch_size=2)
    scheduler = ContinuousBatchingScheduler(mock_engine, config)

    # Add 1 request with max_new_tokens=1
    seq_id, q = scheduler.add_request("Test", max_new_tokens=1)

    # Step 1: Promotion & Prefill
    await scheduler._step()
    assert len(scheduler.running_batch) == 1
    assert scheduler.running_batch[seq_id].status == SequenceStatus.RUNNING

    # Step 2: Decode -> produces token 999 (EOS), finished flag becomes true
    await scheduler._step()

    # Step 3: Eviction
    await scheduler._step()
    assert len(scheduler.running_batch) == 0
    assert scheduler.total_completed == 1

    # Queue should receive prefill token, decode token, then None (sentinel)
    tok1 = await q.get()
    assert tok1 == " token"
    tok2 = await q.get()
    assert tok2 == " token"
    end_sentinel = await q.get()
    assert end_sentinel is None
