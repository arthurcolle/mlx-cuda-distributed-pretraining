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
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for training (e.g., cuda or cpu)')
    parser.add_argument('--teacher_type', type=str, default='auto', choices=['auto', 'hf', 'mlx'],
                        help='Force teacher type: auto (try MLX, fallback to HF), hf, or mlx')
    args = parser.parse_args()
    # Fallback to CPU if CUDA is requested but not available
    try:
        import torch
        if args.device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU.")
            args.device = 'cpu'
    except ImportError:
        pass

    # Set MLX device (ignore torch.device)
    import mlx.core as mx
    if args.device == "gpu":
        mx.set_default_device(mx.gpu)
    else:
        mx.set_default_device(mx.cpu)

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
        ).to(args.device)
        teacher.eval()
        teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_name, use_fast=True)
        teacher_type = 'hf'
    elif teacher is not None:
        print(f"Loaded MLX teacher model {args.teacher_name}")
    else:
        raise RuntimeError("Could not load teacher model (tried MLX and HuggingFace).")

    # Use teacher_tokenizer for data
    tokenizer = teacher_tokenizer

    # Load student model (MLX or HuggingFace)
    print(f"Loading student model {args.student_name}...")
    # Try loading via MLX
    student_mlx, student_tokenizer_mlx = try_load_mlx_teacher(args.student_name)
    if student_mlx is not None:
        student = student_mlx
        student_type = 'mlx'
        print(f"Loaded MLX student model {args.student_name}")
    else:
        print(f"Loading student model {args.student_name} (HuggingFace)...")
        student = AutoModelForCausalLM.from_pretrained(args.student_name).to(args.device)
        student_type = 'hf'
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
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)
            # MLX-only version: use MLX for both teacher and student
            import mlx.core as mx
            import mlx.nn as nn
            # Convert input_ids to mx.array of int32
            mx_input_ids = mx.array(input_ids.cpu().numpy()).astype(mx.int32)
            mx_attention_mask = mx.array(attention_mask.cpu().numpy()).astype(mx.int32)
            mx_labels = mx.array(labels.cpu().numpy()).astype(mx.int32)

            # Forward pass teacher
            if teacher_type == 'mlx':
                teacher_logits = teacher(mx_input_ids)[0]
            else:
                teacher_logits = teacher(input_ids)[0]

            # Forward pass student
            if student_type == 'mlx':
                student_logits = student(mx_input_ids)[0]
            else:
                student_logits = student(input_ids)[0]

            # Compute distillation loss (KL + CE) in numpy
            import numpy as np
            import mlx
            # Convert to numpy for loss calculation
            t_logits = teacher_logits.astype(mx.float32).numpy()
            s_logits = student_logits.astype(mx.float32).numpy()
            lbls = mx_labels.numpy()

            # Soft target loss (KL divergence)
            import torch.nn.functional as F
            import torch
            temperature = args.temperature
            alpha = args.alpha
            p_student = torch.log_softmax(torch.from_numpy(s_logits) / temperature, dim=-1)
            p_teacher = torch.softmax(torch.from_numpy(t_logits) / temperature, dim=-1)
            loss_soft = F.kl_div(p_student, p_teacher, reduction='batchmean') * (temperature ** 2)
            # Hard target loss (cross-entropy)
            loss_hard = F.cross_entropy(
                torch.from_numpy(s_logits).view(-1, s_logits.shape[-1]),
                torch.from_numpy(lbls).view(-1), ignore_index=-100
            )
            loss = alpha * loss_soft + (1 - alpha) * loss_hard

            # Backward/update student (MLX)
            # Implement MLX student backward and optimizer update
            import mlx.core as mx
            import mlx.nn as nn
            import mlx.optimizers as optim

            # Convert s_logits and lbls to mx.array for MLX loss/grad
            mx_s_logits = mx.array(s_logits)
            mx_lbls = mx.array(lbls)

            # Compute loss in MX for backward
            def mlx_loss_fn(student_params):
                # Forward pass with current student params
                # (Assume student is an nn.Module and supports .replace_parameters)
                student_model = student.replace_parameters(student_params)
                out_logits = student_model(mx_input_ids)[0].astype(mx.float32)
                # Cross-entropy loss (hard targets)
                ce_loss = nn.losses.cross_entropy(
                    out_logits.reshape(-1, out_logits.shape[-1]),
                    mx_lbls.reshape(-1),
                    reduction="mean"
                )
                # KL loss (soft targets)
                mx_teacher_logits = mx.array(t_logits)
                T = args.temperature
                log_p = nn.log_softmax(out_logits / T, axis=-1)
                q = nn.softmax(mx_teacher_logits / T, axis=-1)
                log_q = nn.log_softmax(mx_teacher_logits / T, axis=-1)
                kl = mx.mean(mx.sum(q * (log_q - log_p), axis=-1)) * (T * T)
                return args.alpha * kl + (1 - args.alpha) * ce_loss

            # Compute gradients
            loss_value, grads = nn.value_and_grad(student, mlx_loss_fn)(student.parameters())
            # Update student parameters (simple SGD for demonstration)
            learning_rate = args.learning_rate
            new_params = {}
            for k, v in student.parameters().items():
                new_params[k] = v - learning_rate * grads[k]
            student = student.replace_parameters(new_params)

            if step % 50 == 0:
                print(f"  Step {step} Loss {loss_value.item():.4f}")

    # Save distilled model
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving student model to {args.output_dir}")
    student.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == '__main__':
    main()
