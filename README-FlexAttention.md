# FlexAttention for MLX

FlexAttention is a flexible, programmable attention mechanism that provides nearly the same performance as FlashAttention while allowing for custom attention patterns.

## Features

- **Custom Score Modifications**: Apply relative position biases, ALiBi, or any other custom function to attention scores
- **Flexible Masking**: Implement sliding windows, prefix-LM, or any custom masking pattern
- **Block-Sparse Attention**: Skip computation on blocks of tokens that don't need attention
- **Similar API to PyTorch's FlexAttention**: Easy to use if you're familiar with PyTorch's implementation

## Usage Examples

### Basic Usage

```python
from models.attention.flex_attention import FlexAttention

# Initialize the attention module
attention = FlexAttention(
    hidden_size=768,
    num_heads=12,
    num_kv_heads=4,  # For grouped query attention (GQA)
    head_dim=64,
    dropout=0.1,
    use_bias=False
)

# Apply attention
output = attention(input_tensor)
```

### Custom Score Modifications

```python
# Define a relative position bias
def relative_position_bias(score, batch_idx, head_idx, query_idx, key_idx):
    # Apply relative position bias
    distance = query_idx - key_idx
    bias = 0.1 * mx.exp(-abs(distance) / 16)  # Decay with distance
    return score + bias

# Apply with custom scoring function
output = attention(input_tensor, score_mod_fn=relative_position_bias)
```

### Custom Masking

```python
# Define a sliding window + causal mask
def sliding_causal_mask(batch_idx, head_idx, query_idx, key_idx):
    # Causal: can only attend to past tokens
    is_causal = query_idx >= key_idx
    # Sliding window of 256 tokens
    is_in_window = query_idx - key_idx <= 256
    return is_causal and is_in_window

# Apply with custom mask
output = attention(input_tensor, mask_mod_fn=sliding_causal_mask)
```

### Precomputed Block Mask

```python
from models.attention.flex_attention import create_block_mask

# Create a block mask once
block_mask = create_block_mask(
    sliding_causal_mask,
    batch_size=1, 
    num_heads=12,
    q_len=1024,
    kv_len=1024,
    block_size=128
)

# Reuse the mask for multiple attention computations
output = attention(input_tensor, block_mask=block_mask)
```

### Standalone Function

```python
from models.attention.flex_attention import flex_attention

# Use the standalone function with Q, K, V tensors
output = flex_attention(
    q=query_tensor,  # [batch, seq, heads, head_dim] or [batch, heads, seq, head_dim]
    k=key_tensor,
    v=value_tensor,
    score_mod=relative_position_bias,
    mask_mod=sliding_causal_mask,
    dropout=0.1,
    block_size=128
)
```

## Performance Considerations

- The implementation is designed to minimize memory usage while maintaining flexibility
- For small sequence lengths, the overhead of block-sparse computation may not be worth it
- For large sequence lengths with sparse patterns (like sliding windows), the speedup can be significant

## Differences from PyTorch FlexAttention

- Uses MLX's functional approach instead of PyTorch's imperative style
- Implementation details are adapted to MLX's API and limitations
- Optimized for Apple Silicon hardware

## Implementation Details

The implementation involves:
1. Block-sparse attention computation
2. Custom score modification through callbacks
3. Custom mask patterns through callbacks
4. Support for grouped query attention (GQA)
5. Support for both module-based and functional API styles