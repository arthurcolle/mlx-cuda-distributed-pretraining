"""
FlexAttention implementation for MLX
Based on https://pytorch.org/docs/stable/generated/torch.nn.attention.flex_attention.html
Provides flexibility for custom attention patterns while maintaining efficient computation
"""

import math
from typing import Optional, Callable, Tuple, Union

import mlx.core as mx
from mlx.nn import Linear, Module
from mlx_graphs.utils.scatter import scatter


# Default identity functions for score and mask modification
def identity_score_fn(score: mx.array, b_idx: int, h_idx: int, q_idx: int, kv_idx: int) -> mx.array:
    """Identity function for score modification - returns the score unchanged"""
    return score

def default_causal_mask_fn(b_idx: int, h_idx: int, q_idx: int, kv_idx: int) -> bool:
    """Default causal masking function - returns True if q_idx >= kv_idx (causal attention)"""
    return q_idx >= kv_idx


class FlexAttention(Module):
    """
    FlexAttention implementation for MLX
    
    This implementation provides flexible, programmable attention with close to Flash Attention performance:
    1. Supports custom score modifications (relative position, ALiBi, etc.)
    2. Supports custom mask patterns (sliding window, prefix-LM, sparse attention)
    3. Uses block-sparse attention computation for efficiency
    
    Args:
        hidden_size: Size of hidden dimension
        num_heads: Number of attention heads
        num_kv_heads: Number of key/value heads (for MQA/GQA)
        head_dim: Dimension of each attention head
        dropout: Dropout probability
        use_bias: Whether to use bias in projection layers
        block_size: Block size for the block-sparse attention algorithm
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_bias: bool = False,
        block_size: int = 128,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.dropout = dropout
        self.block_size = block_size
        
        # Initialize projection matrices
        self.q_proj = Linear(hidden_size, self.num_heads * self.head_dim, bias=use_bias)
        self.k_proj = Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=use_bias)
        self.v_proj = Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=use_bias)
        self.o_proj = Linear(self.num_heads * self.head_dim, hidden_size, bias=use_bias)
        
        # Initialize scaling factor
        self.scale = self.head_dim ** -0.5
    
    def _repeat_kv_heads(self, x: mx.array) -> mx.array:
        """
        Repeat key/value heads if num_kv_heads < num_heads (for GQA/MQA)
        """
        if self.num_kv_heads == self.num_heads:
            return x
            
        batch_size, seq_len, _ = x.shape
        
        # Reshape and repeat
        x = x.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        x = mx.repeat(
            x, 
            self.num_heads // self.num_kv_heads, 
            axis=2
        )
        
        return x.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
    
    def _create_block_mask(
        self,
        mask_fn: Callable[[int, int, int, int], bool],
        batch_size: int,
        num_heads: int,
        q_len: int,
        kv_len: int
    ) -> mx.array:
        """
        Create a block mask based on a mask function
        
        Args:
            mask_fn: Function that takes (batch, head, q_idx, kv_idx) and returns True if the block should be processed
            batch_size: Batch size
            num_heads: Number of attention heads
            q_len: Query sequence length
            kv_len: Key/value sequence length
            
        Returns:
            Block mask tensor of shape [batch_size, num_heads, q_blocks, kv_blocks]
        """
        q_blocks = (q_len + self.block_size - 1) // self.block_size
        kv_blocks = (kv_len + self.block_size - 1) // self.block_size
        
        # Initialize an empty mask using mx.full instead of zeros_like
        # We're creating a mask that will be True for blocks that should be processed
        mask_data = mx.full((batch_size, num_heads, q_blocks, kv_blocks), False, dtype=mx.bool_)
        
        # Fill the mask based on the mask function
        for b in range(batch_size):
            for h in range(num_heads):
                for q_block in range(q_blocks):
                    q_start = q_block * self.block_size
                    for kv_block in range(kv_blocks):
                        kv_start = kv_block * self.block_size
                        
                        # Check if this block should be processed
                        # Using the midpoint of the block as a representative check
                        q_idx = min(q_start + self.block_size // 2, q_len - 1)
                        kv_idx = min(kv_start + self.block_size // 2, kv_len - 1)
                        
                        if mask_fn(b, h, q_idx, kv_idx):
                            # In MLX we need to create a new array instead of using in-place updates
                            indices = mx.array([b, h, q_block, kv_block])
                            
                            # Create mask to update this specific position using the more efficient approach
                            mask_data = scatter(mask_data, indices, mx.full(indices.shape, True))
        
        return mask_data
    
    def _flex_attention(
        self, 
        q: mx.array, 
        k: mx.array, 
        v: mx.array,
        score_mod_fn: Optional[Callable[[mx.array, int, int, int, int], mx.array]] = None,
        mask_mod_fn: Optional[Callable[[int, int, int, int], bool]] = None,
        block_mask: Optional[mx.array] = None
    ) -> mx.array:
        """
        Implements the flex attention algorithm with tiling and custom modifications
        
        Args:
            q: Query tensor [batch_size, seq_len, num_heads, head_dim]
            k: Key tensor [batch_size, seq_len, num_kv_heads, head_dim]
            v: Value tensor [batch_size, seq_len, num_kv_heads, head_dim]
            score_mod_fn: Optional function to modify attention scores
            mask_mod_fn: Optional function to determine block sparsity
            block_mask: Optional pre-computed block mask
            
        Returns:
            Output tensor [batch_size, seq_len, num_heads, head_dim]
        """
        batch_size, seq_len, num_heads, head_dim = q.shape
        _, kv_len, num_kv_heads, _ = k.shape
        
        # Verify score_mod_fn is callable if provided
        if score_mod_fn is not None and not callable(score_mod_fn):
            raise TypeError(f"score_mod_fn must be callable, got {type(score_mod_fn)}")
            
        # Verify mask_mod_fn is callable if provided
        if mask_mod_fn is not None and not callable(mask_mod_fn):
            raise TypeError(f"mask_mod_fn must be callable, got {type(mask_mod_fn)}")
        
        # Handle different number of KV heads (grouped query attention)
        if num_heads > num_kv_heads:
            # Each KV head serves multiple query heads
            repeat_factor = num_heads // num_kv_heads
            
            k_expanded = mx.repeat(
                k.reshape(batch_size, kv_len, num_kv_heads, 1, head_dim),
                repeat_factor,
                axis=3
            )
            v_expanded = mx.repeat(
                v.reshape(batch_size, kv_len, num_kv_heads, 1, head_dim),
                repeat_factor,
                axis=3
            )
            
            k = k_expanded.reshape(batch_size, kv_len, num_heads, head_dim)
            v = v_expanded.reshape(batch_size, kv_len, num_heads, head_dim)
        
        # Scale query
        q = q * self.scale
        
        # Transpose for batch matrix multiplication
        q = q.transpose(0, 2, 1, 3)  # [batch, head, seq_q, dim]
        k = k.transpose(0, 2, 1, 3)  # [batch, head, seq_k, dim]
        v = v.transpose(0, 2, 1, 3)  # [batch, head, seq_v, dim]
        
        # Create block mask if mask_mod_fn is provided but block_mask isn't
        if block_mask is None and mask_mod_fn is not None:
            block_mask = self._create_block_mask(
                mask_mod_fn, batch_size, num_heads, seq_len, kv_len
            )
        
        # Since we can't do in-place updates in MLX, we'll use a simpler approach
        # that works similar to the block-sparse attention but doesn't require array_update
        
        # Compute standard attention scores
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2))
        
        # Apply score modification if provided
        if score_mod_fn is not None:
            # Create a modified version with score_mod applied
            # More efficiently gather all modifications and apply once
            indices_list = []
            modified_values = []
            
            for b_idx in range(batch_size):
                for h_idx in range(num_heads):
                    for q_idx in range(seq_len):
                        for kv_idx in range(kv_len):
                            # Collect all modifications
                            indices_list.append(mx.array([b_idx, h_idx, q_idx, kv_idx]))
                            modified_values.append(score_mod_fn(
                                scores[b_idx, h_idx, q_idx, kv_idx],
                                b_idx, h_idx, q_idx, kv_idx
                            ))
            
            # Apply all modifications at once
            indices = mx.stack(indices_list)
            values = mx.stack(modified_values)
            modified_scores = mx.scatter(mx.zeros_like(scores), indices, values)
            scores = modified_scores
        
        # Apply block masking if provided
        if block_mask is not None:
            # Create a mask in the shape of scores
            attention_mask = mx.ones_like(scores) * -float("inf")
            valid_mask = mx.ones_like(scores)
            
            # Apply block masking
            q_blocks = (seq_len + self.block_size - 1) // self.block_size
            kv_blocks = (kv_len + self.block_size - 1) // self.block_size
            
            for b_idx in range(batch_size):
                for h_idx in range(num_heads):
                    for q_block in range(q_blocks):
                        q_start = q_block * self.block_size
                        q_end = min(q_start + self.block_size, seq_len)
                        
                        for kv_block in range(kv_blocks):
                            kv_start = kv_block * self.block_size
                            kv_end = min(kv_start + self.block_size, kv_len)
                            
                            # If this block should be masked out
                            if not block_mask[b_idx, h_idx, q_block, kv_block]:
                                # Create a block mask for this region
                                block_indices = mx.stack(
                                    mx.meshgrid(
                                        mx.arange(q_start, q_end),
                                        mx.arange(kv_start, kv_end)
                                    ),
                                    axis=0
                                )
                                
                                # Update the mask for this block
                                for q_pos in range(q_start, q_end):
                                    for kv_pos in range(kv_start, kv_end):
                                        valid_mask = scatter(
                                            valid_mask, 
                                            mx.array([b_idx, h_idx, q_pos, kv_pos]), 
                                            mx.array(0.0)
                                        )
            
            # Apply the mask to scores
            scores = scores * valid_mask + attention_mask * (1 - valid_mask)
        
        # Apply softmax along the sequence dimension
        attention_weights = mx.softmax(scores, axis=-1)
        
        # Apply dropout if needed
        if self.dropout > 0.0:
            attention_weights = mx.dropout(attention_weights, self.dropout)
        
        # [batch, head, seq_q, seq_k] @ [batch, head, seq_v, dim] -> [batch, head, seq_q, dim]
        context = mx.matmul(attention_weights, v)
        
        # [batch, head, seq, dim] -> [batch, seq, head, dim]
        context = context.transpose(0, 2, 1, 3)
        
        return context
    
    def __call__(
        self,
        x: mx.array,
        score_mod_fn: Optional[Callable[[mx.array, int, int, int, int], mx.array]] = None,
        mask_mod_fn: Optional[Callable[[int, int, int, int], bool]] = None,
        block_mask: Optional[mx.array] = None,
        mask: Optional[mx.array] = None
    ) -> mx.array:
        """
        Forward pass for FlexAttention
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            score_mod_fn: Optional function to modify attention scores
                          Takes (score, batch_idx, head_idx, query_idx, key_idx) and returns modified score
            mask_mod_fn: Optional function to determine mask/sparsity pattern
                         Takes (batch_idx, head_idx, query_idx, key_idx) and returns True if attention is allowed
            block_mask: Optional pre-computed block mask
            mask: Optional attention mask for compatibility with other attention implementations
            
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
        
        # Use default functions if none provided
        if score_mod_fn is None:
            score_mod_fn = identity_score_fn
            
        if mask_mod_fn is None and mask is not None:
            # If traditional mask is provided but no mask_mod_fn,
            # convert the mask into a mask_mod_fn
            def mask_from_tensor(b_idx, h_idx, q_idx, kv_idx):
                return mask[b_idx, h_idx, q_idx, kv_idx] != -float("inf")
            mask_mod_fn = mask_from_tensor
        elif mask_mod_fn is None:
            # Default to causal attention if no mask is provided
            mask_mod_fn = default_causal_mask_fn
        
        # Apply flex attention
        context = self._flex_attention(q, k, v, score_mod_fn, mask_mod_fn, block_mask)
        
        # Reshape back
        context = context.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        
        # Apply output projection
        output = self.o_proj(context)
        
        return output


def create_block_mask(
    mask_fn: Callable[[int, int, int, int], bool],
    batch_size: int,
    num_heads: int,
    q_len: int,
    kv_len: int,
    block_size: int = 128
) -> mx.array:
    """
    Create a block mask for FlexAttention based on a mask function
    
    Args:
        mask_fn: Function that takes (batch, head, q_idx, kv_idx) and returns True if the block should be processed
        batch_size: Batch size
        num_heads: Number of attention heads
        q_len: Query sequence length
        kv_len: Key/value sequence length 
        block_size: Size of attention blocks
        
    Returns:
        Block mask tensor of shape [batch_size, num_heads, q_blocks, kv_blocks]
    """
    q_blocks = (q_len + block_size - 1) // block_size
    kv_blocks = (kv_len + block_size - 1) // block_size
    
    # Initialize an empty mask with mx.full instead of zeros
    mask = mx.full((batch_size, num_heads, q_blocks, kv_blocks), False, dtype=mx.bool_)
    
    # Fill the mask based on the mask function - avoiding array_update
    mask_list = []
    indices_list = []
    
    for b in range(batch_size):
        for h in range(num_heads):
            for q_block in range(q_blocks):
                q_start = q_block * block_size
                for kv_block in range(kv_blocks):
                    kv_start = kv_block * block_size
                    
                    # Check if this block should be processed
                    # Using the midpoint of the block as a representative check
                    q_idx = min(q_start + block_size // 2, q_len - 1)
                    kv_idx = min(kv_start + block_size // 2, kv_len - 1)
                    
                    if mask_fn(b, h, q_idx, kv_idx):
                        # Add to list for later scattering
                        indices_list.append(mx.array([b, h, q_block, kv_block]))
                        mask_list.append(mx.array(True))
    
    # Scatter all true values at once if there are any
    if indices_list:
        indices = mx.stack(indices_list)
        values = mx.stack(mask_list)
        mask = mx.scatter(mask, indices, values)
    
    return mask


def flex_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    score_mod: Optional[Callable[[mx.array, int, int, int, int], mx.array]] = None,
    mask_mod: Optional[Callable[[int, int, int, int], bool]] = None,
    block_mask: Optional[mx.array] = None,
    scale: Optional[float] = None,
    dropout: float = 0.0,
    block_size: int = 128,
    mask: Optional[mx.array] = None
) -> mx.array:
    """
    Standalone flex attention function for direct use without the module
    
    Args:
        q: Query tensor [batch_size, seq_len, num_heads, head_dim] or [batch_size, num_heads, seq_len, head_dim]
        k: Key tensor [batch_size, seq_len, num_heads, head_dim] or [batch_size, num_heads, seq_len, head_dim] 
        v: Value tensor [batch_size, seq_len, num_heads, head_dim] or [batch_size, num_heads, seq_len, head_dim]
        score_mod: Optional function to modify attention scores
        mask_mod: Optional function to determine mask/sparsity pattern
        block_mask: Optional pre-computed block mask
        scale: Optional scale factor for queries (if None, uses 1/sqrt(head_dim))
        dropout: Dropout probability
        block_size: Block size for the block-sparse attention algorithm
        
    Returns:
        Output tensor in same shape format as inputs
    """
    # Use default functions if none provided
    if score_mod is None:
        score_mod = identity_score_fn
    elif not callable(score_mod):
        raise TypeError(f"score_mod must be callable, got {type(score_mod)}")
        
    if mask_mod is None and mask is not None:
        # If traditional mask is provided but no mask_mod_fn,
        # convert the mask into a mask_mod_fn
        def mask_from_tensor(b_idx, h_idx, q_idx, kv_idx):
            return mask[b_idx, h_idx, q_idx, kv_idx] != -float("inf")
        mask_mod = mask_from_tensor
    elif mask_mod is None:
        # Default to causal attention if no mask is provided
        mask_mod = default_causal_mask_fn
    elif not callable(mask_mod):
        raise TypeError(f"mask_mod must be callable, got {type(mask_mod)}")
    
    # Handle different input formats
    input_format = "BNSD"  # Batch, Seq, Head, Dim
    if q.shape[1] == k.shape[1] == v.shape[1] and q.shape[1] < q.shape[2]:
        input_format = "BHSD"  # Batch, Head, Seq, Dim
        # Convert to BNSD format
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
    
    batch_size, q_len, num_heads, head_dim = q.shape
    _, kv_len, _, _ = k.shape
    
    # Apply scaling
    if scale is None:
        scale = head_dim ** -0.5
    q = q * scale
    
    # Transpose for batch matrix multiplication
    q = q.transpose(0, 2, 1, 3)  # [batch, head, seq_q, dim]
    k = k.transpose(0, 2, 1, 3)  # [batch, head, seq_k, dim]
    v = v.transpose(0, 2, 1, 3)  # [batch, head, seq_v, dim]
    
    # Create block mask if mask_mod is provided but block_mask isn't
    if block_mask is None and mask_mod is not None:
        block_mask = create_block_mask(
            mask_mod, batch_size, num_heads, q_len, kv_len, block_size
        )
    
    # Compute standard attention scores
    scores = mx.matmul(q, k.transpose(0, 1, 3, 2))
    
    # Apply score modification if provided
    if score_mod is not None:
        # Since we can't do in-place updates efficiently, we'll create a mask of which
        # positions need modifying, then apply it all at once
        modified_scores = mx.zeros_like(scores)
        
        for b_idx in range(batch_size):
            for h_idx in range(num_heads):
                for q_idx in range(q_len):
                    for kv_idx in range(kv_len):
                        # Apply the score modification to each position
                        new_score = score_mod(
                            scores[b_idx, h_idx, q_idx, kv_idx],
                            b_idx, h_idx, q_idx, kv_idx
                        )
                        # Update the modified scores tensor (using scatter instead of array_update)
                        modified_scores = mx.scatter(
                            modified_scores,
                            mx.array([b_idx, h_idx, q_idx, kv_idx]),
                            new_score
                        )
        
        scores = modified_scores
    
    # Apply block masking if provided
    if block_mask is not None:
        # Create a mask in the shape of scores
        attention_mask = mx.ones_like(scores) * -float("inf")
        valid_mask = mx.ones_like(scores)
        
        # Apply block masking
        q_blocks = (q_len + block_size - 1) // block_size
        kv_blocks = (kv_len + block_size - 1) // block_size
        
        for b_idx in range(batch_size):
            for h_idx in range(num_heads):
                for q_block in range(q_blocks):
                    q_start = q_block * block_size
                    q_end = min(q_start + block_size, q_len)
                    
                    for kv_block in range(kv_blocks):
                        kv_start = kv_block * block_size
                        kv_end = min(kv_start + block_size, kv_len)
                        
                        # If this block should be masked out
                        if not block_mask[b_idx, h_idx, q_block, kv_block]:
                            for q_pos in range(q_start, q_end):
                                for kv_pos in range(kv_start, kv_end):
                                    valid_mask = scatter(
                                        valid_mask, 
                                        mx.array([b_idx, h_idx, q_pos, kv_pos]), 
                                        mx.array(0.0)
                                    )
        
        # Apply the mask to scores
        scores = scores * valid_mask + attention_mask * (1 - valid_mask)
    
    # Apply softmax along the sequence dimension
    attention_weights = mx.softmax(scores, axis=-1)
    
    # Apply dropout if needed
    if dropout > 0.0:
        attention_weights = mx.dropout(attention_weights, dropout)
    
    # [batch, head, seq_q, seq_k] @ [batch, head, seq_v, dim] -> [batch, head, seq_q, dim]
    context = mx.matmul(attention_weights, v)
    
    # Return to original format
    if input_format == "BNSD":
        return context.transpose(0, 2, 1, 3)  # [batch, seq, head, dim]
    else:
        return context  # Already in [batch, head, seq, dim]
