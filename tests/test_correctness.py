"""
Correctness Test: Compares FlashServe-Lite generated tokens against HuggingFace generate().
Both use greedy decoding (temperature=0.0). Proves that PagedKVCache is mathematically lossless.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flashserve.config import EngineConfig
from flashserve.engine import InferenceEngine
from flashserve.sequence import Sequence


@pytest.fixture(scope="module")
def setup_models():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for correctness benchmark test")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    config = EngineConfig(
        model_name=model_name,
        num_gpu_blocks=500,
        local_files_only=True,
    )
    flash_engine = InferenceEngine(config)

    hf_tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="cuda",
        local_files_only=True,
    )
    hf_model.eval()

    return flash_engine, hf_tokenizer, hf_model


def test_greedy_equivalence_with_huggingface(setup_models):
    flash_engine, hf_tokenizer, hf_model = setup_models
    test_prompt = "Explain why the Earth revolves around the Sun:"
    max_tokens = 15

    # 1. Generate with HuggingFace generate()
    hf_inputs = hf_tokenizer(test_prompt, return_tensors="pt").to("cuda")
    prompt_len = hf_inputs["input_ids"].shape[1]

    with torch.inference_mode():
        hf_output = hf_model.generate(
            **hf_inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            repetition_penalty=1.0,
            use_cache=True,
        )
    hf_generated_ids = hf_output[0, prompt_len:].tolist()

    # 2. Generate with FlashServe-Lite engine
    prompt_token_ids = flash_engine.tokenizer.encode(test_prompt)
    seq = Sequence(
        seq_id=99,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_tokens,
        temperature=0.0,
    )

    first_tok = flash_engine.prefill(seq)
    flash_generated_ids = [first_tok]

    curr_tok = first_tok
    for _ in range(max_tokens - 1):
        next_tok = flash_engine.decode_single(seq, curr_tok)
        flash_generated_ids.append(next_tok)
        curr_tok = next_tok

    flash_engine.free_sequence(99)

    # 3. Assert exact token-for-token equality
    print("\nHF Tokens:        ", hf_generated_ids)
    print("FlashServe Tokens:", flash_generated_ids)
    print("Decoded HF Text:        ", hf_tokenizer.decode(hf_generated_ids))
    print("Decoded FlashServe Text:", flash_engine.tokenizer.decode(flash_generated_ids))

    assert flash_generated_ids == hf_generated_ids, (
        f"Mismatch between FlashServe-Lite and HuggingFace outputs!\n"
        f"HF:         {hf_generated_ids}\n"
        f"FlashServe: {flash_generated_ids}"
    )
