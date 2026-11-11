"""
Unit tests for PagedKVCache block allocation, deallocation, scatter, and gather.
"""

import pytest
import torch
from flashserve.kv_cache import PagedKVCache


def test_block_allocation_and_free():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = PagedKVCache(
        num_layers=4,
        num_kv_heads=2,
        head_dim=64,
        block_size=16,
        num_blocks=100,
        device=device,
        dtype=torch.float32,
    )

    assert cache.num_free_blocks == 100
    assert len(cache.free_blocks) == 100

    # Allocate 5 blocks for seq 1
    blocks_seq1 = cache.allocate_blocks(seq_id=1, num_blocks_needed=5)
    assert len(blocks_seq1) == 5
    assert cache.num_free_blocks == 95
    assert cache.block_tables[1] == blocks_seq1

    # Allocate 3 blocks for seq 2
    blocks_seq2 = cache.allocate_blocks(seq_id=2, num_blocks_needed=3)
    assert len(blocks_seq2) == 3
    assert cache.num_free_blocks == 92

    # Append 1 block to seq 1
    new_blk = cache.allocate_one_block(seq_id=1)
    assert cache.num_free_blocks == 91
    assert len(cache.block_tables[1]) == 6
    assert cache.block_tables[1][-1] == new_blk

    # Free seq 1
    cache.free_sequence(seq_id=1)
    assert cache.num_free_blocks == 97
    assert 1 not in cache.block_tables

    # Free seq 2
    cache.free_sequence(seq_id=2)
    assert cache.num_free_blocks == 100
    assert 2 not in cache.block_tables


def test_out_of_memory_error():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = PagedKVCache(
        num_layers=2,
        num_kv_heads=2,
        head_dim=32,
        block_size=8,
        num_blocks=10,
        device=device,
        dtype=torch.float32,
    )

    cache.allocate_blocks(seq_id=10, num_blocks_needed=8)
    assert cache.num_free_blocks == 2

    # Attempting to allocate 5 blocks when only 2 remain should raise RuntimeError
    with pytest.raises(RuntimeError):
        cache.allocate_blocks(seq_id=20, num_blocks_needed=5)


def test_scatter_gather_roundtrip():
    """Verify that scattering KV activations into pages and gathering them back produces exact equality."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_layers = 2
    num_kv_heads = 2
    head_dim = 64
    block_size = 16
    prompt_len = 37  # Non-multiple of block_size (needs 3 blocks)

    cache = PagedKVCache(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        num_blocks=20,
        device=device,
        dtype=torch.float32,
    )

    needed_blocks = cache.get_num_required_blocks(prompt_len)
    assert needed_blocks == 3
    cache.allocate_blocks(seq_id=1, num_blocks_needed=needed_blocks)

    # Generate synthetic K and V states
    torch.manual_seed(42)
    original_keys = [
        torch.randn(num_kv_heads, prompt_len, head_dim, device=device, dtype=torch.float32)
        for _ in range(num_layers)
    ]
    original_values = [
        torch.randn(num_kv_heads, prompt_len, head_dim, device=device, dtype=torch.float32)
        for _ in range(num_layers)
    ]

    # Scatter into pages
    for l_idx in range(num_layers):
        cache.scatter_prefill(
            seq_id=1,
            layer_idx=l_idx,
            key_states=original_keys[l_idx],
            value_states=original_values[l_idx],
        )

    # Gather back from pages
    for l_idx in range(num_layers):
        k_gathered, v_gathered = cache.gather_kv(seq_id=1, layer_idx=l_idx, seq_len=prompt_len)
        # Expected shape: [1, num_kv_heads, prompt_len, head_dim]
        assert k_gathered.shape == (1, num_kv_heads, prompt_len, head_dim)
        assert v_gathered.shape == (1, num_kv_heads, prompt_len, head_dim)

        k_orig = original_keys[l_idx].unsqueeze(0)
        v_orig = original_values[l_idx].unsqueeze(0)

        torch.testing.assert_close(k_gathered, k_orig)
        torch.testing.assert_close(v_gathered, v_orig)


def test_decode_token_scatter():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = PagedKVCache(
        num_layers=2,
        num_kv_heads=2,
        head_dim=64,
        block_size=16,
        num_blocks=20,
        device=device,
        dtype=torch.float32,
    )

    # Allocate initial 1 block for seq 42
    cache.allocate_blocks(seq_id=42, num_blocks_needed=1)

    # Write token at position 5
    k_token = torch.randn(2, 64, device=device, dtype=torch.float32)
    v_token = torch.randn(2, 64, device=device, dtype=torch.float32)

    cache.scatter_decode_token(seq_id=42, layer_idx=0, position=5, key_state=k_token, value_state=v_token)

    # Gather up to 6 tokens
    k_gat, v_gat = cache.gather_kv(seq_id=42, layer_idx=0, seq_len=6)
    # The 5th token should match exactly
    torch.testing.assert_close(k_gat[0, :, 5, :], k_token)
    torch.testing.assert_close(v_gat[0, :, 5, :], v_token)
