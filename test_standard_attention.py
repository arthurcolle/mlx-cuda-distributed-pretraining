#!/usr/bin/env python
# Test standard attention implementation

import mlx.core as mx
import mlx.nn as nn

class StandardAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads=None,
        head_dim=None,
        dropout=0.0,
        use_bias=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.dropout = dropout
        
        # Initialize projection matrices
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=use_bias)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=use_bias)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=use_bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=use_bias)
        
        # Initialize scaling factor
        self.scale = self.head_dim ** -0.5
        
    def __call__(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project inputs to queries, keys, and values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Handle grouped query attention (GQA) or multi-query attention (MQA)
        if self.num_kv_heads < self.num_heads:
            # Repeat KV heads if num_kv_heads < num_heads
            k = mx.repeat(
                k.reshape(batch_size, seq_len, self.num_kv_heads, 1, self.head_dim),
                self.num_heads // self.num_kv_heads,
                axis=3
            ).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            
            v = mx.repeat(
                v.reshape(batch_size, seq_len, self.num_kv_heads, 1, self.head_dim),
                self.num_heads // self.num_kv_heads,
                axis=3
            ).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            
        # Transpose for attention computation
        q = q.transpose(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.transpose(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
        v = v.transpose(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Compute attention scores
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2))  # transpose last two dims
        scores = scores * self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask
            
        # Apply softmax
        attn_weights = mx.softmax(scores, axis=-1)
        
        # Apply dropout if needed
        if self.dropout > 0.0:
            attn_weights = nn.dropout(attn_weights, self.dropout)
            
        # Compute output
        context = mx.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Reshape back
        context = context.transpose(0, 2, 1, 3)  # [batch_size, seq_len, num_heads, head_dim]
        context = context.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        
        # Apply output projection
        output = self.o_proj(context)
        
        return output


def test_standard_attention():
    """Test the standard attention implementation"""
    batch_size = 2
    seq_len = 16
    hidden_size = 128
    head_dim = 32
    
    # Test case 1: Equal number of heads (vanilla attention)
    print("\nTest 1: Equal number of heads (4 query, 4 kv)")
    num_heads = 4
    num_kv_heads = 4
    attn = StandardAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim
    )
    
    # Create a sample input tensor
    x = mx.random.normal((batch_size, seq_len, hidden_size))
    
    # Forward pass
    output = attn(x)
    
    # Check shape
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: [{batch_size}, {seq_len}, {hidden_size}]")
    
    # Test case 2: MQA (Multi-Query Attention)
    print("\nTest 2: MQA (4 query, 1 kv)")
    num_heads = 4
    num_kv_heads = 1
    attn_mqa = StandardAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim
    )
    
    # Forward pass
    output_mqa = attn_mqa(x)
    
    # Check shape
    print(f"Output shape: {output_mqa.shape}")
    print(f"Expected shape: [{batch_size}, {seq_len}, {hidden_size}]")
    
    # Test case 3: GQA (Grouped-Query Attention)
    print("\nTest 3: GQA (4 query, 2 kv)")
    num_heads = 4
    num_kv_heads = 2
    attn_gqa = StandardAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim
    )
    
    # Forward pass
    output_gqa = attn_gqa(x)
    
    # Check shape
    print(f"Output shape: {output_gqa.shape}")
    print(f"Expected shape: [{batch_size}, {seq_len}, {hidden_size}]")
    
    # Test case 4: With attention mask
    print("\nTest 4: With causal attention mask")
    mask = mx.full((seq_len, seq_len), -float("inf"))
    mask = mx.triu(mask, k=1)
    # Add batch dimension
    attention_mask = mask[None, None, :, :]
    
    # Forward pass with mask
    output_masked = attn(x, attention_mask)
    
    # Check shape
    print(f"Output shape: {output_masked.shape}")
    print(f"Expected shape: [{batch_size}, {seq_len}, {hidden_size}]")
    
    return "All tests completed successfully"

if __name__ == "__main__":
    print("Testing StandardAttention implementation...")
    result = test_standard_attention()
    print(f"\nResult: {result}")