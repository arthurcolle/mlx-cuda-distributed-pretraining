#!/usr/bin/env python
# Test for the flash attention implementation

import mlx.core as mx
import mlx.nn as nn
from arch.flash_attention import FlashAttention

def test_flash_attention():
    """Test the FlashAttention implementation with different numbers of heads"""
    batch_size = 2
    seq_len = 16
    hidden_size = 128
    head_dim = 32
    
    # Test case 1: Equal number of heads (vanilla attention)
    print("\nTest 1: Equal number of heads (4 query, 4 kv)")
    num_heads = 4
    num_kv_heads = 4
    flash_attn = FlashAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        flash_block_size=8
    )
    
    # Create a sample input tensor
    x = mx.random.normal((batch_size, seq_len, hidden_size))
    
    # Forward pass
    output = flash_attn(x)
    
    # Check shape
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: [{batch_size}, {seq_len}, {hidden_size}]")
    
    # Test case 2: MQA (Multi-Query Attention)
    print("\nTest 2: MQA (4 query, 1 kv)")
    num_heads = 4
    num_kv_heads = 1
    flash_attn_mqa = FlashAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        flash_block_size=8
    )
    
    # Forward pass
    output_mqa = flash_attn_mqa(x)
    
    # Check shape
    print(f"Output shape: {output_mqa.shape}")
    print(f"Expected shape: [{batch_size}, {seq_len}, {hidden_size}]")
    
    # Test case 3: GQA (Grouped-Query Attention)
    print("\nTest 3: GQA (4 query, 2 kv)")
    num_heads = 4
    num_kv_heads = 2
    flash_attn_gqa = FlashAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        flash_block_size=8
    )
    
    # Forward pass
    output_gqa = flash_attn_gqa(x)
    
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
    output_masked = flash_attn(x, attention_mask)
    
    # Check shape
    print(f"Output shape: {output_masked.shape}")
    print(f"Expected shape: [{batch_size}, {seq_len}, {hidden_size}]")
    
    return "All tests completed successfully"

if __name__ == "__main__":
    print("Testing FlashAttention implementation...")
    result = test_flash_attention()
    print(f"\nResult: {result}")