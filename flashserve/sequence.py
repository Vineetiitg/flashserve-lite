"""
Sequence representation and lifecycle state tracking.
"""

from dataclasses import dataclass, field
from enum import Enum
import time


class SequenceStatus(Enum):
    WAITING = "waiting"       # In the waiting queue
    RUNNING = "running"       # Actively participating in decode iterations
    FINISHED = "finished"     # Completed generation (EOS or max_new_tokens reached)


@dataclass
class Sequence:
    """Represents a single request's generation state in the engine."""
    seq_id: int
    prompt_token_ids: list[int]
    
    # State tracking
    generated_token_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    
    # Paged KV Cache tracking
    block_ids: list[int] = field(default_factory=list)  # Physical block IDs allocated
    current_length: int = 0                              # Total tokens in KV cache (prompt + generated)
    
    # Timing & Latency instrumentation
    arrival_time: float = field(default_factory=time.time)
    first_token_time: float = 0.0                        # Timestamp when first token was emitted
    finish_time: float = 0.0                             # Timestamp when generation completed
    
    # Generation parameters
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 0.9
    top_k: int = 50
    eos_token_id: int = 151645                           # Default to Qwen2.5 EOS token id
    
    @property
    def num_generated(self) -> int:
        return len(self.generated_token_ids)
    
    @property
    def prompt_length(self) -> int:
        return len(self.prompt_token_ids)
    
    @property
    def total_length(self) -> int:
        return self.prompt_length + self.num_generated
    
    @property
    def is_finished(self) -> bool:
        if self.status == SequenceStatus.FINISHED:
            return True
        if self.num_generated >= self.max_new_tokens:
            return True
        if self.generated_token_ids and self.generated_token_ids[-1] == self.eos_token_id:
            return True
        return False
    
    @property
    def time_to_first_token(self) -> float:
        if self.first_token_time > 0:
            return self.first_token_time - self.arrival_time
        return 0.0
    
    @property
    def total_time(self) -> float:
        if self.finish_time > 0:
            return self.finish_time - self.arrival_time
        return 0.0
