# generate_lite.py
# Copyright © 2023-2024 Apple Inc.

import time
import logging
from contextlib import contextmanager
from typing import Any, Callable, Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache

##############################################################################
# Minimal Utilities (wired_limit, generation_stream, maybe_quantize_kv_cache)
##############################################################################

# A stream on the default device just for generation
generation_stream = mx.new_stream(mx.default_device())

class ModelNotFoundError(Exception):
    """Exception for missing model files."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

@contextmanager
def wired_limit(model: nn.Module, streams: Optional[List[mx.Stream]] = None):
    """
    A context manager to temporarily change the wired limit, synchronizing
    streams on exit to prevent overlapping changes in asynchronous contexts.
    
    For newer MLX versions that don't have set_wired_limit, this becomes a no-op.
    """
    model_bytes = 0
    # Recursively sum up all array nbytes in the model
    def _tree_reduce(m, acc=0):
        if isinstance(m, mx.array):
            return acc + m.nbytes
        if isinstance(m, nn.Module):
            for child in m.children():
                acc = _tree_reduce(child, acc)
        return acc
    model_bytes = _tree_reduce(model)

    # Check if we're on metal and if set_wired_limit exists
    is_metal = hasattr(mx, 'metal') and hasattr(mx.metal, 'device_info')
    has_wired_limit = hasattr(mx, 'set_wired_limit')
    
    # Only attempt to set the limit if both conditions are true
    old_limit = None
    if is_metal and has_wired_limit:
        max_rec_size = mx.metal.device_info()["max_recommended_working_set_size"]
        if model_bytes > 0.9 * max_rec_size:
            model_mb = model_bytes // 2**20
            max_rec_mb = max_rec_size // 2**20
            print(
                f"[WARNING] Generating with a model that requires {model_mb} MB, "
                f"close to the max recommended {max_rec_mb} MB. This can be slow."
            )
        old_limit = mx.set_wired_limit(max_rec_size)
    
    try:
        yield
    finally:
        if streams is not None:
            for s in streams:
                mx.synchronize(s)
        else:
            mx.synchronize()
        
        # Only reset the limit if we set it earlier
        if is_metal and has_wired_limit and old_limit is not None:
            mx.set_wired_limit(old_limit)

def maybe_quantize_kv_cache(prompt_cache, quantized_kv_start, kv_group_size, kv_bits):
    """
    If we've passed 'quantized_kv_start', convert the KV cache to a quantized
    variant (if not already), using the specified group size and bits.
    """
    if (
        kv_bits is not None
        and not isinstance(prompt_cache[0], nn.cache.QuantizedKVCache)
        and prompt_cache[0].offset > quantized_kv_start
    ):
        for i in range(len(prompt_cache)):
            if isinstance(prompt_cache[i], nn.cache.KVCache):
                prompt_cache[i] = prompt_cache[i].to_quantized(
                    group_size=kv_group_size, bits=kv_bits
                )


##############################################################################
# The Core Generator (generate_step)
##############################################################################

def generate_step(
    prompt: mx.array,
    model: nn.Module,
    *,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    max_kv_size: Optional[int] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 512,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    prompt_progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    A low-level generator producing token ids from 'prompt', using 'model'.
    Yields (token, logprobs) as we generate one token at a time.
    """
    y = prompt
    tokens = None  # for the optional logits processors

    # Create (or reuse) the key-value cache for generation
    if prompt_cache is None:
        prompt_cache = make_prompt_cache(model, max_kv_size=max_kv_size)
    elif len(prompt_cache) != len(model.layers):
        raise ValueError("Wrong number of layers in the prompt cache.")

    # Default to argmax if no sampler is provided
    if sampler is None:
        def default_sampler(logprobs):
            next_token = mx.argmax(logprobs, axis=-1)
            # Ensure we return a scalar for consistency
            if hasattr(next_token, 'item'):
                return next_token.item()
            return next_token
        sampler = default_sampler
    logits_processors = logits_processors or []
    prompt_progress_callback = prompt_progress_callback or (lambda *_: None)

    def _step(y_tok: mx.array):
        """One forward pass step: produce next-token logits, apply processors."""
        # Ensure y_tok is 1D before adding batch dimension
        if y_tok.ndim == 0:
            y_tok = y_tok[None]
        
        try:
            # Try to run the model with the token
            logits = model(y_tok[None], cache=prompt_cache)
        except ValueError as e:
            if "reshape" in str(e):
                # This is likely a dimension mismatch in the attention mechanism
                # Try to fix the model dimensions
                if hasattr(model, 'model') and hasattr(model.model, 'layers') and len(model.model.layers) > 0:
                    for layer in model.model.layers:
                        if hasattr(layer, 'self_attn'):
                            # Get the embedding dimension
                            if hasattr(layer, 'input_layernorm') and hasattr(layer.input_layernorm, 'weight'):
                                hidden_size = layer.input_layernorm.weight.shape[0]
                                # Adjust n_heads to be a divisor of hidden_size
                                for n_heads in [12, 16, 8, 4]:
                                    if hidden_size % n_heads == 0:
                                        head_dim = hidden_size // n_heads
                                        print(f"Adjusting attention: hidden_size={hidden_size}, n_heads={n_heads}, head_dim={head_dim}")
                                        layer.self_attn.n_heads = n_heads
                                        layer.self_attn.head_dim = head_dim
                                        break
                
                # Try again with adjusted dimensions
                logits = model(y_tok[None], cache=prompt_cache)
            else:
                # Re-raise if it's not a reshape error
                raise
        # logits shape: [1, seq_len=1, vocab_size]
        logits = logits[:, -1, :]  # take the last token
        if logits_processors:
            nonlocal tokens
            tokens = mx.concat([tokens, y_tok]) if tokens is not None else y_tok
            idx = tokens.size - 1  # Current position in token sequence
            for processor in logits_processors:
                logits = processor(tokens, logits, idx)

        maybe_quantize_kv_cache(prompt_cache, quantized_kv_start, kv_group_size, kv_bits)
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        next_token = sampler(logprobs)
        # Ensure next_token has a batch dimension (1D array)
        if next_token.ndim == 0:
            next_token = next_token[None]
        # Force evaluation of the token to ensure it's computed
        mx.eval(next_token)
        # Handle different logprobs shapes safely
        if logprobs.ndim > 1 and logprobs.shape[0] == 1:
            return next_token, logprobs.squeeze(0)
        else:
            return next_token, logprobs

    # Prefill stage: feed large chunks of the prompt to fill the cache
    total_prompt_tokens = y.size
    prompt_processed_tokens = 0
    while y.size > prefill_step_size:
        model(y[:prefill_step_size][None], cache=prompt_cache)
        maybe_quantize_kv_cache(prompt_cache, quantized_kv_start, kv_group_size, kv_bits)
        mx.eval([c.state for c in prompt_cache])
        prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
        prompt_processed_tokens += prefill_step_size
        y = y[prefill_step_size:]
        mx.clear_cache()

    # Process the remainder of the prompt in a single step
    y, logprobs = _step(y)
    mx.async_eval(y, logprobs)

    # Generate tokens up to max_tokens
    n = 0
    while True:
        if n != max_tokens:
            next_y, next_logprobs = _step(y)
            mx.async_eval(next_y, next_logprobs)
        if n == 0:
            mx.eval(y)
            prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
        if n == max_tokens:
            break
        # Output the current token & logprobs
        yield y.item(), logprobs
        if n % 256 == 0:
            mx.clear_cache()
        y, logprobs = next_y, next_logprobs
        n += 1


##############################################################################
# The "generate_lite" Function
##############################################################################

def generate_lite(
    model: nn.Module,
    prompt: mx.array,
    *,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    max_kv_size: Optional[int] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 512,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    prompt_progress_callback: Optional[Callable[[int, int], None]] = None,
    stop_tokens: Optional[List[int]] = None,
    verbose: bool = False,
):
    """
    A compact function that generates tokens from an mx.array prompt,
    without requiring any tokenizer. It supports:
      - caching (prompt_cache)
      - samplers
      - logits processors
      - custom stopping tokens
      - kv cache quantization
      - prefill steps
      - optional verbose logging

    Args:
        model (nn.Module): The model to use for generation.
        prompt (mx.array): The prompt tokens.
        max_tokens (int): Maximum new tokens to generate.
        sampler (Callable[[mx.array], mx.array], optional): Sampler for picking the next token
            from logprobs. Defaults to argmax if not provided.
        logits_processors (List[Callable[[mx.array, mx.array], mx.array]], optional):
            Functions to transform the logits at each step, e.g. repetition penalty.
        max_kv_size (int, optional): Maximum capacity of the KV cache.
        prompt_cache (Any, optional): Existing cache to reuse; updated in-place.
        prefill_step_size (int): How many tokens to feed at once for the prompt.
        kv_bits (int, optional): Bits for KV cache quantization.
        kv_group_size (int): Group size for KV quantization.
        quantized_kv_start (int): Step index at which to begin quantizing the KV cache.
        prompt_progress_callback (Callable[[int, int], None], optional):
            Callback that receives (#tokens_processed, total_prompt_tokens).
        stop_tokens (List[int], optional): If a generated token is in this list, generation stops.
        verbose (bool): Print basic debug info (timing, memory usage, etc.).

    Returns:
        mx.array: The concatenated tokens (original prompt + newly generated tokens).
    """
    if stop_tokens is None:
        stop_tokens = []
    lps = []
    with wired_limit(model, [generation_stream]):
        start_time = time.perf_counter()
        generated_tokens = []

        # Loop over generate_step
        for i, (token, logprobs) in enumerate(
            generate_step(
                prompt,
                model,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                max_kv_size=max_kv_size,
                prompt_cache=prompt_cache,
                prefill_step_size=prefill_step_size,
                kv_bits=kv_bits,
                kv_group_size=kv_group_size,
                quantized_kv_start=quantized_kv_start,
                prompt_progress_callback=prompt_progress_callback,
            )
        ):
            # On the first iteration, measure how long the prompt took
            if i == 0:
                prompt_time = time.perf_counter() - start_time
                prompt_tps = (prompt.size / prompt_time) if prompt_time > 0 else 0.0
                # Reset timer for generation
                start_time = time.perf_counter()
            generated_tokens.append(token)
            lps.append(logprobs[token])

            # Stop if we hit any user-defined stop token
            if token in stop_tokens:
                break

        # Final stats
        generation_time = time.perf_counter() - start_time
        generation_tps = (len(generated_tokens) / generation_time) if generation_time > 0 else 0.0

    # Print debug info if requested
    if verbose:
        print("=" * 10)
        if len(generated_tokens) == 0:
            print("No tokens generated for this prompt.")
        else:
            print(f"Prompt: {prompt.size} tokens, {prompt_tps:.3f} tokens/sec")
            print(
                f"Generation: {len(generated_tokens)} tokens, "
                f"{generation_tps:.3f} tokens/sec"
            )
            used_mem_gb = mx.get_peak_memory() / 1e9
            print(f"Peak memory: {used_mem_gb:.3f} GB")
    lps_avg = sum(lps)
    # Return the combined sequence: original prompt + newly generated tokens
    if generated_tokens:
        return mx.array(generated_tokens, dtype=prompt.dtype), lps_avg
    else:
        return mx.array([], dtype=prompt.dtype), lps_avg

def beam_search(model, input_tokens, max_tokens=512, verbose=False, n_beams=4, stop_tokens=None):
    """
    Perform beam search to generate text from the model.
    """
    # Repeat the input for each beam and initialize beam scores.
    beams = mx.repeat(mx.array([input_tokens]), n_beams, axis=0).tolist()
    beam_scores = [0] * n_beams
    finished_beams = []
    l_prefix = len(input_tokens)  # To later remove the input prefix from the output.

    for step in range(max_tokens):
        # Use the current number of beams instead of the constant n_beams.
        current_beam_count = len(beams)
        logits = model(mx.array(beams))[:, -1, :]  # Get logits for the last token in each beam.
        logprobs = nn.log_softmax(logits, axis=-1)   # Convert logits to log probabilities.
        # For each beam, pick the top n_beams candidate tokens.
        top_indices = mx.argsort(-logprobs, axis=-1)[:, :n_beams]
        #top_logprobs = logprobs[mx.arange(current_beam_count), top_indices]
        top_logprobs = mx.take_along_axis(logprobs, top_indices, axis=-1)
        top_indices = top_indices.tolist()
        top_logprobs = top_logprobs.tolist()

        # Build candidate extensions for each beam.
        beam_possibilities = []
        for beam_idx in range(current_beam_count):
            token_and_score = []
            for k in range(n_beams):
                token_and_score.append((top_indices[beam_idx][k], top_logprobs[beam_idx][k]))
            beam_possibilities.append(token_and_score)
        
        # Extend each current beam with every candidate token.
        new_beams = []
        for beam_idx in range(current_beam_count):
            base_beam = beams[beam_idx]
            base_beam_score = beam_scores[beam_idx]
            for token, logprob in beam_possibilities[beam_idx]:
                new_beam = base_beam + [token]
                mix = 1/(len(new_beam) - l_prefix)
                #new_score = base_beam_score * (1-mix) + logprob * (mix)
                new_score = base_beam_score + logprob
                new_beams.append((new_beam, new_score))
        
        # Sort and de-duplicate the candidate beams.
        seen_beams = set()
        dedup_new_beams = []
        for beam, score in new_beams:
            hash_beam = tuple(beam)
            if hash_beam not in seen_beams:
                seen_beams.add(hash_beam)
                dedup_new_beams.append((beam, score))
        new_beams = dedup_new_beams
        new_beams.sort(key=lambda x: x[1], reverse=True)
        # Select the top candidates while checking for stop tokens.
        chosen_beams = []
        while len(chosen_beams) < n_beams and new_beams:
            possible_beam, possible_score = new_beams.pop(0)
            if stop_tokens is not None and possible_beam[-1] in stop_tokens:
                if len(possible_beam) - l_prefix == 1: # its just an EOS token
                    possible_score = -float('inf')  # Penalize EOS to avoid it being chosen unless it's the only option.
                finished_beams.append((possible_beam[:-1], possible_score))
                n_beams -= 1  # Reduce the beam count since we finished one.
            else:
                chosen_beams.append((possible_beam, possible_score))
        
        # Update the beams and scores for the next iteration.
        beams = [beam for beam, score in chosen_beams]
        beam_scores = [score for beam, score in chosen_beams]
        
        # Exit early if no beams are left to extend.
        if len(beams) == 0:
            if verbose:
                print("All beams finished.")
            break

    # If no beams finished with a stop token, use the current beams.
    if not finished_beams:
        finished_beams = list(zip(beams, beam_scores))
    else:
        finished_beams.extend(
            [(beam, score) for beam, score in zip(beams, beam_scores) if len(beam) > l_prefix]
        )
    finished_beams.sort(key=lambda x: x[1], reverse=True)
    # Remove the input prefix from the output beams.
    finished_beams = [(beam[l_prefix:], score) for beam, score in finished_beams]
    return finished_beams
