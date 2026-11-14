"""
Integration test for InferenceEngine prefill and decode execution.
"""

import pytest
import torch
from flashserve.config import EngineConfig
from flashserve.engine import InferenceEngine
from flashserve.sequence import Sequence


@pytest.fixture(scope="module")
def engine():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for engine integration test")
    config = EngineConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        num_gpu_blocks=500,
        local_files_only=True,
    )
    return InferenceEngine(config)


def test_engine_prefill_and_decode(engine):
    prompt = "The primary color of the sky is"
    token_ids = engine.tokenizer.encode(prompt)
    seq = Sequence(
        seq_id=1,
        prompt_token_ids=token_ids,
        max_new_tokens=10,
        temperature=0.0,
    )

    # 1. Prefill
    first_token = engine.prefill(seq)
    assert isinstance(first_token, int)
    assert first_token >= 0
    assert seq.current_length == len(token_ids)
    assert 1 in engine.kv_cache.block_tables

    # 2. Decode 5 steps
    curr_token = first_token
    for _ in range(5):
        next_tok = engine.decode_single(seq, curr_token)
        assert isinstance(next_tok, int)
        curr_token = next_tok

    assert seq.current_length == len(token_ids) + 5

    # 3. Cleanup
    engine.free_sequence(1)
    assert 1 not in engine.kv_cache.block_tables
