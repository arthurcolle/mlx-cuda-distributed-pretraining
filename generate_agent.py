#!/usr/bin/env python3
"""
generate_agent.py

Launches a multimodal Llama model with simple tool-use (e.g. calculator) in the generation loop.
"""
import argparse
import json
import time
import re
from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer
from PIL import Image
import numpy as np

def load_model_and_tokenizer(run_name: str):
    """Load a multimodal model and its tokenizer from a training run."""
    run_dir = Path("runs") / run_name
    # Load metadata
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        raise ValueError(f"metadata.json not found in run directory: {run_dir}")
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    # Determine architecture
    arch = metadata.get('config', {}).get('model', {}).get('architecture')
    if arch != 'multimodal_llama':
        raise ValueError(f"generate_agent only supports 'multimodal_llama', got: {arch}")
    # Import model class
    from models.multimodal_llama import Model, ModelArgs
    # Reconstruct ModelArgs from metadata
    mc = metadata['config']['model']
    dims = mc['dimensions']
    att = mc['attention']
    norm = mc['normalization']
    rope = mc['rope']
    misc = mc['misc']
    tok_info = metadata.get('tokenizer', {})
    vocab_size = tok_info.get('vocab_size', None)
    # Build args
    args = ModelArgs(
        model_type=arch,
        hidden_size=dims['hidden_size'],
        num_hidden_layers=dims.get('num_layers', 8),
        intermediate_size=dims['intermediate_size'],
        num_attention_heads=att['num_heads'],
        head_dim=att.get('head_dim', None),
        vocab_size=vocab_size,
        num_key_value_heads=att.get('num_kv_heads', None),
        rms_norm_eps=norm['rms_norm_eps'],
        max_position_embeddings=misc.get('max_position_embeddings', 256),
        attention_bias=misc.get('attention_bias', False),
        mlp_bias=misc.get('mlp_bias', False),
        rope_theta=rope.get('theta', 10000.0),
        rope_traditional=rope.get('traditional', False),
        rope_scaling=rope.get('scaling', None),
        tie_word_embeddings=misc.get('tie_word_embeddings', False),
        logit_scale=misc.get('logit_scale', None),
        use_flash_attention=att.get('use_flash_attention', True),
        use_flex_attention=att.get('use_flex_attention', False),
        flash_block_size=att.get('flash_block_size', 128),
        image_size=dims.get('image_size', 224),
    )
    # Instantiate and load
    model = Model(args)
    # Find latest checkpoint
    ckp_dir = run_dir / 'checkpoints'
    candidates = list(ckp_dir.glob('step_final_model.safetensors'))
    if not candidates:
        candidates = list(ckp_dir.glob('step_*_model.safetensors'))
    if not candidates:
        raise ValueError(f"No model checkpoint found in {ckp_dir}")
    checkpoint = sorted(candidates)[-1]
    print(f"Loading model weights from {checkpoint}")
    model.load_weights(str(checkpoint))
    # Load tokenizer
    tok_path = run_dir / 'tokenizer' / 'tokenizer.json'
    if not tok_path.exists():
        raise ValueError(f"Tokenizer file not found: {tok_path}")
    print(f"Loading tokenizer from {tok_path}")
    tokenizer = Tokenizer.from_file(str(tok_path))
    return model, tokenizer

def call_tool(text: str) -> str:
    """Detect special TOOL blocks and execute a simple calculator."""
    pattern = re.compile(r"<<TOOL:(\w+)>>(.*?)<</TOOL>>", re.DOTALL)
    def _repl(m):
        tool = m.group(1)
        expr = m.group(2).strip()
        if tool == 'calculator':
            try:
                # Safe eval context
                result = eval(expr, {'__builtins__': {}})
            except Exception as e:
                result = f"Error: {e}"
        else:
            result = f"Unsupported tool: {tool}"
        return f"{m.group(0)}\n[ToolResult:{tool}] {result}"  # annotate result
    return pattern.sub(_repl, text)

def generate_agent(model, tokenizer, image_path: str, prompt: str,
                   max_tokens: int = 100, temperature: float = 1.0, seed: int = None) -> str:
    """Generate text token-by-token, invoking tools when requested."""
    # Set seed
    if seed is not None:
        mx.random.seed(seed)
    else:
        mx.random.seed(int(time.time() * 1000))
    # Prepare image
    img = Image.open(image_path).convert('RGB')
    size = model.args.image_size
    img = img.resize((size, size))
    arr = np.array(img).astype('float32') / 255.0
    images = mx.array(arr.transpose(2, 0, 1))[None, ...]
    # Tokenize prompt
    enc = tokenizer.encode(prompt)
    tokens = mx.array(enc.ids)[None, ...]
    model.eval()
    # Generation loop
    for _ in range(max_tokens):
        logits = model(images, tokens)[0, -1]
        if temperature > 0:
            logits = logits / temperature
            probs = mx.softmax(logits, axis=-1)
            next_id = int(mx.random.categorical(probs[None, ...]).item())
        else:
            next_id = int(mx.argmax(logits).item())
        tokens = mx.concat([tokens, mx.array([next_id])[None, ...]], axis=1)
        # Decode and check tools
        text = tokenizer.decode(tokens[0].tolist())
        if '<<TOOL:' in text:
            text = call_tool(text)
            # re-encode full text to include tool results
            tokens = mx.array(tokenizer.encode(text).ids)[None, ...]
        # Stop on EOS
        try:
            eos_id = tokenizer.token_to_id('</s>')
            if eos_id is not None and next_id == eos_id:
                break
        except Exception:
            pass
    return tokenizer.decode(tokens[0].tolist())

def main():
    parser = argparse.ArgumentParser(description='Multimodal Agent Generator')
    parser.add_argument('--run', type=str, required=True, help='Name of the training run')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt')
    parser.add_argument('--max-tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    # Select device
    if mx.gpu_is_available():
        mx.set_default_device(mx.gpu)
        print('Using GPU for generation')
    else:
        mx.set_default_device(mx.cpu)
        print('Using CPU for generation')
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.run)
    # Generate
    print('\nStarting generation with agentic tool support...')
    out = generate_agent(
        model, tokenizer,
        args.image, args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed
    )
    print('\n=== Generated Output ===\n')
    print(out)

if __name__ == '__main__':
    main()