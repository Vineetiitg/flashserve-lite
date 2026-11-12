"""
Sampling strategies for token generation (Greedy, Top-k, Top-p).
"""

import torch


def sample_token(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: float = 0.9,
    top_k: int = 50,
) -> int:
    """
    Sample a single token id from unnormalized logits.
    
    Args:
        logits: Tensor of shape [vocab_size] or [1, vocab_size]
        temperature: 0.0 = greedy argmax, >0.0 = stochastic temperature scaling
        top_p: Nucleus filtering threshold (cumulative probability)
        top_k: Top-k filtering threshold
    
    Returns:
        int token id
    """
    if logits.ndim > 1:
        logits = logits.squeeze(0)
    
    if temperature <= 0.0:
        return int(torch.argmax(logits, dim=-1).item())
    
    # Apply temperature
    logits = logits / temperature
    
    # Top-K filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float("Inf")
    
    # Top-p (nucleus) filtering
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float("Inf")
    
    probs = torch.softmax(logits, dim=-1)
    # Guard against NaN/inf
    if torch.isnan(probs).any():
        return int(torch.argmax(logits, dim=-1).item())
    
    next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token.item())
