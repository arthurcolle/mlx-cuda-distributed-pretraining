import mlx.core as mx
import numpy as np

# Add scheduling functions
def linear_schedule(start_value, end_value, steps):
    """Create a linear learning rate schedule.
    
    Args:
        start_value: Initial learning rate
        end_value: Final learning rate
        steps: Number of steps for the schedule
        
    Returns:
        A function that takes a step and returns the corresponding learning rate
    """
    def schedule(step):
        if step >= steps:
            return end_value
        return start_value + (end_value - start_value) * (step / steps)
    return schedule

def cosine_decay(start_value, steps, end_value=0.0):
    """Create a cosine decay learning rate schedule.
    
    Args:
        start_value: Initial learning rate
        steps: Number of steps for the schedule
        end_value: Final learning rate (default: 0.0)
        
    Returns:
        A function that takes a step and returns the corresponding learning rate
    """
    def schedule(step):
        if step >= steps:
            return end_value
        progress = step / steps
        cosine_decay = 0.5 * (1 + mx.cos(mx.pi * progress))
        return end_value + (start_value - end_value) * cosine_decay
    return schedule

def join_schedules(schedules, transition_steps):
    """Join multiple schedules at specified transition steps.
    
    Args:
        schedules: List of schedule functions
        transition_steps: List of steps at which to transition between schedules
        
    Returns:
        A function that takes a step and returns the corresponding learning rate
    """
    def schedule(step):
        for i, transition_step in enumerate(transition_steps):
            if step < transition_step:
                return schedules[i](step)
        return schedules[-1](step - transition_steps[-1])
    return schedule

def make_sampler(temp=1.0, min_p=None, top_p=None):
    """Create a sampler function for generating tokens.
    
    Args:
        temp: The temperature for sampling (default: 1.0)
        min_p: Minimum probability threshold (default: None)
        top_p: Top-p sampling threshold (default: None)
        
    Returns:
        A sampling function
    """
    if min_p:
        # MinP sampling
        def sampler(logits):
            probs = mx.softmax(logits * (1 / temp), axis=-1)
            sorted_indices = mx.argsort(-logits)
            sorted_probs = probs[sorted_indices]
            top_prob = probs[sorted_indices[0]]
            scaled_min_p = min_p * top_prob
            tokens_to_remove = sorted_probs < scaled_min_p
            tokens_to_remove[0] = False  # Keep at least one token
            selected_probs = mx.where(tokens_to_remove, 0, sorted_probs)
            selected_probs_sum = mx.sum(selected_probs)
            if selected_probs_sum > 0:
                selected_probs = selected_probs / selected_probs_sum
            sorted_token = mx.random.categorical(mx.log(selected_probs))
            return sorted_indices[sorted_token]
    elif top_p:
        # TopP sampling
        def sampler(logits):
            probs = mx.softmax(logits * (1 / temp), axis=-1)
            sorted_indices = mx.argsort(-probs)
            sorted_probs = probs[sorted_indices]
            cumulative_probs = mx.cumsum(sorted_probs)
            mask = cumulative_probs <= top_p
            # Keep at least one token
            if not mask.any():
                mask = mx.zeros_like(mask)
                mask = mask.at[0].set(True)
            # Zero out all tokens above the top_p threshold
            sorted_probs = mx.where(mask, sorted_probs, 0)
            sorted_probs_sum = mx.sum(sorted_probs)
            if sorted_probs_sum > 0:
                sorted_probs = sorted_probs / sorted_probs_sum
            sorted_token = mx.random.categorical(mx.log(sorted_probs))
            return sorted_indices[sorted_token]
    else:
        # Default temperature sampling
        def sampler(logits):
            return mx.random.categorical(logits * (1 / temp))
            
    return sampler

def make_logits_processors(repetition_penalty=1.0, repetition_context_size=0):
    """Create logits processors for text generation.
    
    Args:
        repetition_penalty: Penalty for repeated tokens (default: 1.0)
        repetition_context_size: Context size to consider for repetition penalty (default: 0)
        
    Returns:
        A list of logits processor functions
    """
    processors = []
    
    if repetition_penalty != 1.0 and repetition_context_size > 0:
        def repetition_processor(logits, tokens, idx):
            if idx < repetition_context_size:
                context = tokens[:idx]
            else:
                context = tokens[idx-repetition_context_size:idx]
            
            for t in context:
                t_idx = int(t)
                logits = logits.at[t_idx].set(logits[t_idx] / repetition_penalty)
            
            return logits
        
        processors.append(repetition_processor)
    
    return processors