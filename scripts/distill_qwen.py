#!/usr/bin/env python3
"""
Distillation script: train a student model from a Qwen teacher via knowledge distillation.
"""
import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Optionally support MLX as teacher
def try_load_mlx_teacher(teacher_name):
    try:
        from mlx_lm import load as mlx_load
        model, tokenizer = mlx_load(teacher_name)
        return model, tokenizer
    except Exception:
        return None, None

def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    # Soft target loss (KL divergence)
    p_student = F.log_softmax(student_logits / temperature, dim=-1)
    p_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    loss_soft = F.kl_div(p_student, p_teacher, reduction='batchmean') * (temperature ** 2)
    # Hard target loss (cross-entropy)
    loss_hard = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1), ignore_index=-100
    )
    return alpha * loss_soft + (1 - alpha) * loss_hard

def tokenize_fn(examples, tokenizer, max_length):
    # expects examples to have a 'text' field
    texts = examples['text']
    # Handle HuggingFace and MLX tokenizers
    if callable(tokenizer):
        out = tokenizer(
            texts, padding='max_length', truncation=True,
            max_length=max_length, return_tensors=None
        )
    elif hasattr(tokenizer, "batch_encode") or hasattr(tokenizer, "batch_encode_plus"):
        # MLX TokenizerWrapper
        if hasattr(tokenizer, "batch_encode"):
            batch_encode = tokenizer.batch_encode
        else:
            batch_encode = tokenizer.batch_encode_plus
        out = batch_encode(
            texts, padding='max_length', truncation=True,
            max_length=max_length, return_tensors=None
        )
    else:
        raise ValueError("Unknown tokenizer type: cannot tokenize input.")
    # labels are just the input_ids (auto-regressive LM)
    out['labels'] = out['input_ids'].copy()
    return out

def main():
    parser = argparse.ArgumentParser(description="Distill Qwen into a smaller student model.")
    parser.add_argument('--teacher_name', type=str, required=True,
                        help='HuggingFace or MLX name/path for the Qwen teacher model')
    parser.add_argument('--student_name', type=str, required=True,
                        help='HuggingFace name or path for the student model to initialize')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to JSONL dataset with a field \"text\" for inputs')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Where to save the distilled student model')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Weight for soft-target loss (vs. hard-label loss)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for training (e.g., cuda or cpu)')
    parser.add_argument('--teacher_type', type=str, default='auto', choices=['auto', 'hf', 'mlx'],
                        help='Force teacher type: auto (try MLX, fallback to HF), hf, or mlx')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')

    # Load teacher (MLX or HF)
    teacher = None
    teacher_tokenizer = None
    teacher_type = args.teacher_type
    if teacher_type == 'mlx' or teacher_type == 'auto':
        teacher, teacher_tokenizer = try_load_mlx_teacher(args.teacher_name)
        if teacher is not None:
            teacher_type = 'mlx'
    if teacher is None and (teacher_type == 'hf' or teacher_type == 'auto'):
        print(f"Loading teacher model {args.teacher_name} (HuggingFace)...")
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher_name, torch_dtype=torch.float16
        ).to(device)
        teacher.eval()
        teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_name, use_fast=True)
        teacher_type = 'hf'
    elif teacher is not None:
        print(f"Loaded MLX teacher model {args.teacher_name}")
    else:
        raise RuntimeError("Could not load teacher model (tried MLX and HuggingFace).")

    # Use teacher_tokenizer for data
    tokenizer = teacher_tokenizer

    # Load student
    print(f"Loading student model {args.student_name}...")
    student = AutoModelForCausalLM.from_pretrained(args.student_name).to(device)
    student.train()

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    ds = load_dataset('json', data_files={'train': args.dataset}, split='train')
    # Tokenize
    ds = ds.map(lambda ex: tokenize_fn(ex, tokenizer, args.max_length),
                batched=True, remove_columns=ds.column_names)
    ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate)
    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(loader, start=1):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with torch.no_grad():
                if teacher_type == 'hf':
                    teacher_logits = teacher(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    ).logits
                elif teacher_type == 'mlx':
                    import numpy as np
                    import mlx.core as mx
                    # Convert input_ids to numpy, then to mlx array, then to int32
                    np_input_ids = input_ids.cpu().numpy()
                    mx_input_ids = mx.array(np_input_ids).astype(mx.int32)
                    # MLX models expect mx.array of int32, not numpy array
                    teacher_logits = teacher(mx_input_ids)[0]
                    # Convert back to numpy, then to torch tensor
                    teacher_logits = torch.tensor(teacher_logits.numpy(), dtype=torch.float32, device=device)
                else:
                    raise RuntimeError("Unknown teacher type")
            student_outputs = student(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            loss = distillation_loss(
                student_outputs.logits, teacher_logits,
                labels, args.temperature, args.alpha
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % 50 == 0:
                print(f"  Step {step} Loss {loss.item():.4f}")

    # Save distilled model
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving student model to {args.output_dir}")
    student.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == '__main__':
    main()
