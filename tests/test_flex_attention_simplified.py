"""
Test FlexAttention implementation for MLX (simplified version)
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
        self.seq_len = 8  # Smaller for quicker tests
        self.hidden_size = 64  # Smaller for quicker tests
        self.num_heads = 2
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
        
    def test_forward_pass(self):
        """Test basic forward pass"""
        output = self.flex_attention(self.inputs)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_size))
        
        # Test the standalone flex_attention function
        q = mx.array(
            np.random.normal(0, 1, (self.batch_size, self.seq_len, self.num_heads, self.head_dim)).astype(np.float32)
        )
        k = mx.array(
            np.random.normal(0, 1, (self.batch_size, self.seq_len, self.num_heads, self.head_dim)).astype(np.float32)
        )
        v = mx.array(
            np.random.normal(0, 1, (self.batch_size, self.seq_len, self.num_heads, self.head_dim)).astype(np.float32)
        )
        
        output = flex_attention(q, k, v)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.num_heads, self.head_dim))


if __name__ == "__main__":
    unittest.main()