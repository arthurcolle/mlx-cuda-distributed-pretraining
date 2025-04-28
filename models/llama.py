"""
Llama model implementation with FlashAttention and FlexAttention support
Based on the mlx-lm implementation but with performance optimizations
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten, tree_flatten

from models.attention.flash_attention import FlashAttention
from models.attention.simple_attention import SimpleAttention
from models.attention.flex_attention import FlexAttention

@dataclass
class ModelArgs:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    head_dim: Optional[int] = None
    vocab_size: int = 32000
    num_key_value_heads: Optional[int] = None
    rope_theta: float = 10000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Any]] = None
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 4096
    attention_bias: bool = False
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = False
    logit_scale: Optional[float] = None
    mlp_bias: bool = False
    use_flash_attention: bool = True
    use_flex_attention: bool = False
    flash_block_size: int = 128
    num_local_experts: int = 0
    num_experts_per_tok: int = 0


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x = x.astype(mx.float32)
        # Compute RMS norm
        rms = mx.sqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps)
        x = x / rms * self.weight
        return x.astype(dtype)


class RotaryPositionEncoding:
    def __init__(
        self,
        head_dim: int,
        max_positions: int = 4096,
        theta: float = 10000.0,
        traditional: bool = False,
        scaling_factor: Optional[float] = None,
    ):
        self.head_dim = head_dim
        self.max_positions = max_positions
        self.theta = theta
        self.traditional = traditional
        
        # Initialize positions and frequency bases
        if self.traditional:
            # Traditional RoPE implementations
            freqs = 1.0 / (
                theta ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim)
            )
            freqs = mx.repeat(freqs, 2)
        else:
            # Modified implementation (LLaMA, etc)
            freqs = mx.arange(0, head_dim, 2).astype(mx.float32)
            freqs = mx.power(theta, -freqs / head_dim)
        
        # Apply scaling if provided
        if scaling_factor is not None:
            scale = 1.0 / (scaling_factor ** (mx.arange(max_positions).astype(mx.float32) / max_positions))
            t = mx.arange(max_positions).astype(mx.float32)
            freqs = mx.outer(t, freqs)
            freqs = mx.repeat(freqs, 2, axis=1)
            self.cos = mx.cos(freqs)
            self.sin = mx.sin(freqs)
        else:
            t = mx.arange(max_positions).astype(mx.float32)
            freqs = mx.outer(t, freqs)
            self.freqs = freqs
    
    def __call__(self, x: mx.array, position_ids: mx.array) -> mx.array:
        # x: [batch_size, seq_len, num_heads, head_dim]
        batch_size, seq_len, num_heads, _ = x.shape
        
        # Reshape for computation
        x = x.reshape(batch_size, seq_len, num_heads, -1, 2)
        
        # Apply RoPE rotation
        if hasattr(self, "cos"):
            # Precomputed approach
            idx = position_ids[:, None, None, :]
            cos = mx.take(self.cos, idx, axis=0)
            sin = mx.take(self.sin, idx, axis=0)
            # Reshape for broadcasting
            cos = cos.reshape(batch_size, 1, 1, seq_len, self.head_dim)
            sin = sin.reshape(batch_size, 1, 1, seq_len, self.head_dim)
            
            # Calculate rotations
            x_2d = x.reshape(batch_size, seq_len, num_heads, self.head_dim // 2, 2)
            x_2d_rotate = mx.stack(
                [-x_2d[..., 1], x_2d[..., 0]], axis=-1
            )
            
            # Apply rotation
            x_rotated = x_2d * cos + x_2d_rotate * sin
        else:
            # Traditional approach with computed rotations
            freqs = mx.gather(self.freqs, position_ids)
            freqs = freqs[:, :, None, :]  # Shape: [batch, seq, 1, head_dim]
            
            # Calculate rotations
            cos = mx.cos(freqs)
            sin = mx.sin(freqs)
            x0 = x[..., 0::2]
            x1 = x[..., 1::2]
            x_rotated = mx.stack(
                [x0 * cos - x1 * sin, x0 * sin + x1 * cos], axis=-1
            )
        
        # Reshape back
        x_rotated = x_rotated.reshape(batch_size, seq_len, num_heads, self.head_dim)
        return x_rotated


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, use_bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=use_bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=use_bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=use_bias)

    def __call__(self, x: mx.array) -> mx.array:
        # Apply SwiGLU activation
        return self.down_proj(self.gate_proj(x) * mx.sigmoid(self.up_proj(x)) * 2)


class AttentionModule(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        max_positions: int = 4096,
        rope_theta: float = 10000.0,
        rope_traditional: bool = False,
        rope_scaling: Optional[Dict[str, Any]] = None,
        use_bias: bool = False,
        use_flash_attention: bool = True,
        use_flex_attention: bool = False,
        flash_block_size: int = 128,
    ):
        super().__init__()
        
        # Initialize parameters
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.use_flash_attention = use_flash_attention
        self.use_flex_attention = use_flex_attention
        
        # Initialize projection matrices - prioritize flex over flash
        if use_flex_attention:
            # Use FlexAttention for programmable attention patterns
            self.attn = FlexAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                use_bias=use_bias,
                block_size=flash_block_size,
            )
        elif use_flash_attention:
            # Use the dedicated FlashAttention implementation
            self.attn = FlashAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                use_bias=use_bias,
                flash_block_size=flash_block_size,
            )
        else:
            # Use the SimpleAttention implementation instead (safer)
            self.attn = SimpleAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                use_bias=use_bias,
            )
        
        # Initialize positional encoding
        scaling_factor = None
        if rope_scaling is not None:
            scaling_type = rope_scaling.get("type", "")
            if scaling_type == "linear":
                scaling_factor = rope_scaling.get("factor", 1.0)
            else:
                # Add other scaling methods if needed
                pass
                
        self.rope = RotaryPositionEncoding(
            head_dim=self.head_dim,
            max_positions=max_positions,
            theta=rope_theta,
            traditional=rope_traditional,
            scaling_factor=scaling_factor,
        )
        
        # Initialize scaling factor
        self.scale = self.head_dim ** -0.5
    
    def __call__(
        self, 
        x: mx.array, 
        mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        score_mod_fn: Optional[Callable] = None,
        mask_mod_fn: Optional[Callable] = None
    ) -> mx.array:
        batch_size, seq_len, hidden_size = x.shape
        
        # Use position IDs if provided, otherwise default to incremental indices
        if position_ids is None:
            position_ids = mx.arange(seq_len)[None, :]
            
        # Use attention implementation - pass the right parameters based on attention type
        if hasattr(self.attn, '_flex_attention'):
            # For FlexAttention, pass score_mod_fn and mask_mod_fn
            return self.attn(x, score_mod_fn=score_mod_fn, mask_mod_fn=mask_mod_fn, mask=mask)
        else:
            # For other attention types
            return self.attn(x, mask=mask)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: Optional[int] = None,
        intermediate_size: int = None,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        rms_norm_eps: float = 1e-5,
        max_positions: int = 4096,
        rope_theta: float = 10000.0,
        rope_traditional: bool = False,
        rope_scaling: Optional[Dict[str, Any]] = None,
        use_flash_attention: bool = True,
        use_flex_attention: bool = False,
        flash_block_size: int = 128,
        head_dim: Optional[int] = None,
    ):
        super().__init__()
        
        # Determine intermediate size if not provided
        intermediate_size = intermediate_size or 4 * hidden_size
        
        # Initialize layers
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = AttentionModule(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_key_value_heads,
            head_dim=head_dim,
            max_positions=max_positions,
            rope_theta=rope_theta,
            rope_traditional=rope_traditional,
            rope_scaling=rope_scaling,
            use_bias=attention_bias,
            use_flash_attention=use_flash_attention,
            use_flex_attention=use_flex_attention,
            flash_block_size=flash_block_size,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = MLP(hidden_size, intermediate_size, use_bias=mlp_bias)

    def __call__(
        self, 
        x: mx.array, 
        mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        score_mod_fn: Optional[Callable] = None,
        mask_mod_fn: Optional[Callable] = None
    ) -> mx.array:
        # First sublayer: Self-attention
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, mask=mask, position_ids=position_ids, 
                          score_mod_fn=score_mod_fn, mask_mod_fn=mask_mod_fn)
        x = residual + x
        
        # Second sublayer: MLP
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        self.args = args
        self.vocab_size = args.vocab_size
        
        # Initialize token embedding
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        
        # Initialize transformer blocks
        self.layers = [
            TransformerBlock(
                hidden_size=args.hidden_size,
                num_attention_heads=args.num_attention_heads,
                num_key_value_heads=args.num_key_value_heads,
                intermediate_size=args.intermediate_size,
                attention_bias=args.attention_bias,
                mlp_bias=args.mlp_bias,
                rms_norm_eps=args.rms_norm_eps,
                max_positions=args.max_position_embeddings,
                rope_theta=args.rope_theta,
                rope_traditional=args.rope_traditional,
                rope_scaling=args.rope_scaling,
                use_flash_attention=args.use_flash_attention,
                use_flex_attention=args.use_flex_attention,
                flash_block_size=args.flash_block_size,
                head_dim=args.head_dim,
            )
            for _ in range(args.num_hidden_layers)
        ]
        
        # Initialize output normalization
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        
        # Initialize logit scaling
        self.logit_scale = args.logit_scale
        
        # Initialize LM head
        if args.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
    
    def __call__(
        self,
        inputs: mx.array,
        position_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        score_mod_fn: Optional[Callable] = None,
        mask_mod_fn: Optional[Callable] = None
    ) -> mx.array:
        # Process inputs
        batch_size, seq_len = inputs.shape
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = mx.arange(seq_len)[None, :]
        
        # Create causal mask if needed
        if attention_mask is None:
            # Create causal attention mask
            mask = mx.full((seq_len, seq_len), -float("inf"))
            mask = mx.triu(mask, k=1)
            # Add batch dimension
            attention_mask = mask[None, None, :, :]
        
        # Embed tokens
        hidden_states = self.embed_tokens(inputs)
        
        # Process through transformer blocks
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask=attention_mask, position_ids=position_ids,
                                 score_mod_fn=score_mod_fn, mask_mod_fn=mask_mod_fn)
        
        # Apply final normalization
        hidden_states = self.norm(hidden_states)
        
        # Compute logits
        if self.lm_head is None:
            # Tie weights with token embeddings
            logits = mx.matmul(hidden_states, self.embed_tokens.weight.T)
        else:
            # Use dedicated LM head
            logits = self.lm_head(hidden_states)
        
        # Apply logit scaling if provided
        if self.logit_scale is not None:
            logits = logits * self.logit_scale
        
        return logits

    def load_weights(self, path: str, strict: bool = True):
        """Load weights from a checkpoint file"""
        try:
            # Try to load as safetensors first
            weights = mx.load(path)
        except:
            try:
                # Try to load from PyTorch format
                import torch
                if path.endswith(".safetensors"):
                    # For safetensors specific loading
                    from safetensors.torch import load_file
                    weights_torch = load_file(path)
                else:
                    # For PyTorch pth files
                    weights_torch = torch.load(path, map_location="cpu")
                
                # Convert PyTorch tensors to MLX arrays
                weights = {k: mx.array(v.numpy()) for k, v in weights_torch.items()}
            except:
                # Fall back to trying MLX loading formats
                weights = mx.load(path)
        
        # Backward-compatibility shim ----------------------------------------------------
        # Older checkpoints were saved **before** the introduction of the
        # `AttentionModule` wrapper around the low-level attention implementation
        # (Simple/Flash/Flex).  Those checkpoints therefore store the projection
        # matrices directly under `self_attn.<proj_name>` whereas the current
        # code expects `self_attn.attn.<proj_name>`.
        #
        # To remain able to load such checkpoints we detect parameter names that
        # match the old scheme and rewrite them to the new path by inserting an
        # extra "attn" component straight after "self_attn".
        # ---------------------------------------------------------------------------
        remapped_items = []
        for k, v in weights.items():
            if ".self_attn." in k and ".self_attn.attn." not in k:
                parts = k.split(".")
                # Insert "attn" after the first occurrence of "self_attn"
                for idx, part in enumerate(parts):
                    if part == "self_attn":
                        if idx + 1 < len(parts) and parts[idx + 1] != "attn":
                            parts.insert(idx + 1, "attn")
                        break
                new_key = ".".join(parts)
                remapped_items.append((new_key, v))
            else:
                remapped_items.append((k, v))

        # Convert weights to a properly structured dictionary
        param_dict = dict(tree_unflatten(remapped_items))
        
        # Get current model parameters
        model_params = dict(tree_flatten(self.parameters()))
        
        # Check for layer count mismatch
        if not strict:
            # Filter out parameters that don't exist in the model
            filtered_params = {}
            for k, v in param_dict.items():
                if k in model_params:
                    filtered_params[k] = v
                elif k.startswith('model.layers.'):
                    # Extract layer number from parameter name
                    parts = k.split('.')
                    if len(parts) > 2:
                        try:
                            layer_num = int(parts[2])
                            if layer_num >= len(self.layers):
                                print(f"Skipping parameter {k} - model only has {len(self.layers)} layers")
                                continue
                        except ValueError:
                            pass
                    filtered_params[k] = v
            
            param_dict = filtered_params
            print(f"Non-strict loading: filtered out parameters that don't match model architecture")
        
        # Load the weights into the model
        try:
            self.update(param_dict)
            print(f"Successfully loaded weights from {path}")
        except ValueError as e:
            if strict:
                raise
            else:
                print(f"Warning: {str(e)}")
                print("Continuing with partial weight loading due to non-strict mode")
        
        return self
