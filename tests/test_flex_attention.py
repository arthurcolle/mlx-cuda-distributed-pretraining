"""
Test FlexAttention implementation for MLX
"""

import math
import unittest

import mlx.core as mx
import numpy as np

from arch import FlexAttention, flex_attention, create_block_mask


class TestFlexAttention(unittest.TestCase):
    def setUp(self):
        # Common test parameters
        self.batch_size = 2
        self.seq_len = 16
        self.hidden_size = 128
        self.num_heads = 4
        self.head_dim = self.hidden_size // self.num_heads
        
        # Create random input
        np.random.seed(42)
        self.inputs = mx.array(
            np.random.normal(0, 1, (self.batch_size, self.seq_len, self.hidden_size)).astype(np.float32)
        )
        
        # Initialize the attention module
        self.flex_attention = FlexAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout=0.0,
            use_bias=False,
        )
        
    def test_flex_attention_forward(self):
        """Test basic forward pass of FlexAttention"""
        output = self.flex_attention(self.inputs)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_size))
        
    def test_flex_attention_score_mod(self):
        """Test FlexAttention with score modification"""
        
        # Define a simple score modification function (e.g., for relative position bias)
        def rel_pos_bias(score, b, h, q_idx, kv_idx):
            # Simple relative position bias
            distance = q_idx - kv_idx
            bias = 0.1 * math.exp(-abs(distance) / 4)  # Decay with distance
            return score + mx.array(bias, dtype=mx.float32)
        
        # Run with score modification
        output_with_mod = self.flex_attention(self.inputs, score_mod_fn=rel_pos_bias)
        
        # Run without modification for comparison
        output_without_mod = self.flex_attention(self.inputs)
        
        # Outputs should be different when score modification is applied
        diff = mx.abs(output_with_mod - output_without_mod).sum()
        self.assertGreater(diff, 0.01)  # There should be a noticeable difference
        
    def test_flex_attention_mask_mod(self):
        """Test FlexAttention with mask modification"""
        
        # Define a simple mask function (e.g., for causal + sliding window)
        def sliding_causal_mask(b, h, q_idx, kv_idx):
            # Causal (can only attend to past)
            is_causal = q_idx >= kv_idx
            # Sliding window of 8 tokens
            is_in_window = q_idx - kv_idx <= 8
            return is_causal and is_in_window
        
        # Run with mask modification
        output_with_mask = self.flex_attention(self.inputs, mask_mod_fn=sliding_causal_mask)
        
        # Create a full causal mask for comparison
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx
        
        # Run with causal mask only
        output_with_causal = self.flex_attention(self.inputs, mask_mod_fn=causal_mask)
        
        # Outputs should be different when the sliding window is applied
        diff = mx.abs(output_with_mask - output_with_causal).sum()
        self.assertGreater(diff, 0.01)  # There should be a noticeable difference
        
    def test_block_mask_creation(self):
        """Test creating block masks"""
        
        # Define a simple mask function
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx
        
        # Create block mask
        block_size = 4
        mask = create_block_mask(
            causal_mask, 
            self.batch_size, 
            self.num_heads, 
            self.seq_len, 
            self.seq_len, 
            block_size
        )
        
        # Check mask shape: [batch, heads, q_blocks, kv_blocks]
        q_blocks = (self.seq_len + block_size - 1) // block_size
        kv_blocks = (self.seq_len + block_size - 1) // block_size
        self.assertEqual(mask.shape, (self.batch_size, self.num_heads, q_blocks, kv_blocks))
        
        # Verify that upper triangular blocks are masked (False)
        for b in range(self.batch_size):
            for h in range(self.num_heads):
                for q_block in range(q_blocks):
                    for kv_block in range(q_blocks):
                        if q_block < kv_block:
                            # Upper triangular blocks should be masked out (False)
                            self.assertFalse(mask[b, h, q_block, kv_block])
                        else:
                            # Diagonal and lower triangular blocks should be included (True)
                            self.assertTrue(mask[b, h, q_block, kv_block])
                            
    def test_standalone_flex_attention(self):
        """Test the standalone flex_attention function"""
        
        # Create Q, K, V tensors
        q = mx.array(np.random.normal(0, 1, (self.batch_size, self.seq_len, self.num_heads, self.head_dim)).astype(np.float32))
        k = mx.array(np.random.normal(0, 1, (self.batch_size, self.seq_len, self.num_heads, self.head_dim)).astype(np.float32))
        v = mx.array(np.random.normal(0, 1, (self.batch_size, self.seq_len, self.num_heads, self.head_dim)).astype(np.float32))
        
        # Define a rel position bias
        def rel_pos_bias(score, b, h, q_idx, kv_idx):
            distance = q_idx - kv_idx
            bias = 0.1 * math.exp(-abs(distance) / 4)
            return score + mx.array(bias, dtype=mx.float32)
        
        # Run standalone flex attention
        output = flex_attention(q, k, v, score_mod=rel_pos_bias)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.num_heads, self.head_dim))
        
        # Test with different format (batch, head, seq, dim)
        q_bhsd = q.transpose(0, 2, 1, 3)  # [batch, head, seq, dim]
        k_bhsd = k.transpose(0, 2, 1, 3)
        v_bhsd = v.transpose(0, 2, 1, 3)
        
        output_bhsd = flex_attention(q_bhsd, k_bhsd, v_bhsd, score_mod=rel_pos_bias)
        
        # Should return in the same format that was input
        self.assertEqual(output_bhsd.shape, (self.batch_size, self.num_heads, self.seq_len, self.head_dim))
        
    def test_gqa_support(self):
        """Test grouped query attention support"""
        # GQA with 4 query heads and 2 KV heads
        num_kv_heads = 2
        
        flex_attention_gqa = FlexAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=self.head_dim,
        )
        
        output = flex_attention_gqa(self.inputs)
        
        # Output shape should be the same regardless of GQA
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_size))
        
        # Try with standalone function too
        
        # Create Q, K, V tensors for GQA
        q = mx.array(np.random.normal(0, 1, (self.batch_size, self.seq_len, self.num_heads, self.head_dim)).astype(np.float32))
        k = mx.array(np.random.normal(0, 1, (self.batch_size, self.seq_len, num_kv_heads, self.head_dim)).astype(np.float32))
        v = mx.array(np.random.normal(0, 1, (self.batch_size, self.seq_len, num_kv_heads, self.head_dim)).astype(np.float32))
        
        # This should work with different number of heads for Q and KV
        output = flex_attention(q, k, v)
        
        # Output should have same number of heads as query
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.num_heads, self.head_dim))


if __name__ == "__main__":
    unittest.main()