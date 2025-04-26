#!/usr/bin/env python
# Test simple model loading for development

import mlx.core as mx
import mlx.nn as nn
import inspect

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union, Tuple

@dataclass
class SimpleModelArgs:
    model_type: str = "simple"
    vocab_size: int = 32000
    hidden_size: int = 128
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    intermediate_size: int = 512
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    max_position_embeddings: int = 128
    initializer_range: float = 0.02
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

class SimpleSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout_rate = dropout_rate
        
        # Single attention with simple projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        # For scaling attention scores
        self.scale = self.head_dim ** -0.5

    def __call__(
        self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        batch_size, seq_length, _ = hidden_states.shape
        
        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query = query.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Transpose to shape [batch_size, num_heads, seq_length, head_dim]
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)
        
        # Compute attention scores
        attention_scores = mx.matmul(query, key.transpose(0, 1, 3, 2)) * self.scale
        
        # Apply mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        # Apply softmax
        attention_probs = mx.softmax(attention_scores, axis=-1)
        
        # Apply dropout
        if self.dropout_rate > 0.0:
            attention_probs = nn.dropout(attention_probs, self.dropout_rate)
            
        # Compute attention output
        context = mx.matmul(attention_probs, value)
        
        # Transpose back to shape [batch_size, seq_length, hidden_size]
        context = context.transpose(0, 2, 1, 3)
        context = context.reshape(batch_size, seq_length, self.hidden_size)
        
        # Apply output projection
        output = self.o_proj(context)
        
        return output

class SimpleMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        self.dropout_rate = dropout_rate
        
    def __call__(self, x: mx.array) -> mx.array:
        gate = nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        
        # SwiGLU activation
        hidden_states = gate * up
        
        # Output projection
        output = self.down_proj(hidden_states)
        
        # Apply dropout
        if self.dropout_rate > 0.0:
            output = nn.dropout(output, self.dropout_rate)
            
        return output

class SimpleLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.self_attn = SimpleSelfAttention(
            hidden_size=hidden_size, 
            num_heads=num_heads, 
            dropout_rate=dropout_rate
        )
        self.mlp = SimpleMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout_rate=dropout_rate
        )
        self.input_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.post_attention_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        
    def __call__(
        self, 
        hidden_states: mx.array, 
        attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        # Self-attention with pre-normalization
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        # MLP with pre-normalization
        residual = hidden_states
        hidden_states = self.post_attention_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class SimpleModel(nn.Module):
    def __init__(self, config: SimpleModelArgs):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.layers = [
            SimpleLayer(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                dropout_rate=config.dropout_rate
            )
            for _ in range(config.num_hidden_layers)
        ]
        
        self.norm = nn.RMSNorm(config.hidden_size, eps=1e-6)
        
        # Create output projection (tied with input embeddings by default)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.lm_head.weight = self.embed_tokens.weight
        
    def __call__(
        self,
        inputs: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        batch_size, seq_length = inputs.shape
        
        # Create causal mask for attention if not provided
        if attention_mask is None:
            attention_mask = mx.full((seq_length, seq_length), -float("inf"))
            attention_mask = mx.triu(attention_mask, k=1)
            attention_mask = attention_mask.reshape(1, 1, seq_length, seq_length)
        
        # Get token embeddings
        hidden_states = self.embed_tokens(inputs)
        
        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            
        # Apply final normalization
        hidden_states = self.norm(hidden_states)
        
        # Apply language modeling head
        logits = self.lm_head(hidden_states)
        
        return logits

def test_simple_model():
    """Test the simple model implementation with basic forward pass"""
    # Create a simple config
    config = SimpleModelArgs(
        vocab_size=1000, 
        hidden_size=128, 
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512
    )
    
    # Create model
    model = SimpleModel(config)
    
    # Create sample inputs (batch_size=2, seq_length=16)
    inputs = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                      [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
    
    # Forward pass
    logits = model(inputs)
    
    # Check output shape
    print(f"Input shape: {inputs.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: [2, 16, 1000]")
    
    return logits

if __name__ == "__main__":
    print("Testing simple model...")
    test_simple_model()
    print("Test completed successfully!")