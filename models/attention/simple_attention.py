"""
Simple Attention implementation for MLX
A basic implementation that avoids advanced operations
"""

from typing import Optional, Tuple, Callable

import mlx.core as mx
from mlx.nn import Linear, Module


class SimpleAttention(Module):
    """
    Simple Attention implementation for MLX
    
    This implementation provides a basic attention mechanism without
    any complex optimizations to ensure compatibility.
    
    Args:
        hidden_size: Size of hidden dimension
        num_heads: Number of attention heads
        num_kv_heads: Number of key/value heads (for MQA/GQA)
        head_dim: Dimension of each attention head
        dropout: Dropout probability (defaults to 0.0)
        use_bias: Whether to use bias in projection layers
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.dropout = dropout
        
        # Initialize projection matrices
        self.q_proj = Linear(hidden_size, self.num_heads * self.head_dim, bias=use_bias)
        self.k_proj = Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=use_bias)
        self.v_proj = Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=use_bias)
        self.o_proj = Linear(self.num_heads * self.head_dim, hidden_size, bias=use_bias)
        
        # Initialize scaling factor
        self.scale = self.head_dim ** -0.5
    
    def __call__(
        self, 
        x: mx.array, 
        mask: Optional[mx.array] = None,
        score_mod_fn: Optional[Callable[[mx.array, int, int, int, int], mx.array]] = None
    ) -> mx.array:
        """
        Forward pass for SimpleAttention
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            mask: Optional attention mask [batch_size, 1, seq_len, seq_len]
            score_mod_fn: Optional function to modify attention scores
                          Takes (score, batch_idx, head_idx, query_idx, key_idx) and returns modified score
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project inputs to queries, keys, and values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for attention computation
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Handle different number of KV heads (grouped query attention)
        if self.num_heads > self.num_kv_heads:
            # Each KV head serves multiple query heads
            repeat_factor = self.num_heads // self.num_kv_heads
            
            # Reshape to enable broadcasting
            k_expanded = mx.repeat(
                k.reshape(batch_size, seq_len, self.num_kv_heads, 1, self.head_dim),
                repeat_factor,
                axis=3
            )
            v_expanded = mx.repeat(
                v.reshape(batch_size, seq_len, self.num_kv_heads, 1, self.head_dim),
                repeat_factor,
                axis=3
            )
            
            # Reshape to standard format
            k = k_expanded.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            v = v_expanded.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Scale query
        q = q * self.scale
        
        # Transpose for batch matrix multiplication
        # [batch, seq_q, head, dim] -> [batch, head, seq_q, dim]
        q = q.transpose(0, 2, 1, 3)
        # [batch, seq_k, head, dim] -> [batch, head, dim, seq_k]
        k = k.transpose(0, 2, 3, 1)
        # [batch, seq_v, head, dim] -> [batch, head, seq_v, dim]
        v = v.transpose(0, 2, 1, 3)
        
        # [batch, head, seq_q, dim] @ [batch, head, dim, seq_k] -> [batch, head, seq_q, seq_k]
        scores = mx.matmul(q, k)
        
        # Apply score modification if provided
        if score_mod_fn is not None:
            # For each batch
            for b_idx in range(batch_size):
                # For each head
                for h_idx in range(self.num_heads):
                    # For each query position
                    for q_idx in range(seq_len):
                        # For each key position
                        for k_idx in range(seq_len):
                            # Apply the score modification
                            modified_score = score_mod_fn(
                                scores[b_idx, h_idx, q_idx, k_idx],
                                b_idx, h_idx, q_idx, k_idx
                            )
                            scores = mx.array_update(scores, modified_score, (b_idx, h_idx, q_idx, k_idx))
        
        # Apply mask if provided
        if mask is not None:
            # Make sure mask has the right shape for broadcasting
            if mask.ndim == 3:  # [batch, seq_q, seq_k]
                mask = mask[:, None, :, :]  # Add head dimension
            elif mask.ndim == 2:  # [seq_q, seq_k]
                mask = mask[None, None, :, :]  # Add batch and head dimensions
            
            # If mask has only one head dimension, broadcast it
            if mask.shape[1] == 1 and self.num_heads > 1:
                mask = mx.repeat(mask, self.num_heads, axis=1)
                
            scores = scores + mask
        
        # Apply softmax along the sequence dimension
        attention_weights = mx.softmax(scores, axis=-1)
        
        # Apply dropout if needed
        if self.dropout > 0.0:
            attention_weights = mx.dropout(attention_weights, self.dropout)
        
        # [batch, head, seq_q, seq_k] @ [batch, head, seq_v, dim] -> [batch, head, seq_q, dim]
        context = mx.matmul(attention_weights, v)
        
        # [batch, head, seq, dim] -> [batch, seq, head, dim]
        context = context.transpose(0, 2, 1, 3)
        
        # Reshape to concatenate all head outputs
        context = context.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        
        # Apply output projection
        output = self.o_proj(context)
        
        return output