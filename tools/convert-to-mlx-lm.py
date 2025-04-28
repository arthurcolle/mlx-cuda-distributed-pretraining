import argparse
from pathlib import Path
import mlx.core as mx
from train import Trainer
from mlx_lm.sample_utils import make_sampler, make_logits_processors
import mlx.nn as nn
from mlx_lm import load, generate
import time
from generate_lite import generate_lite, beam_search
mx.set_default_device(mx.gpu)
import os
import json
def main():
    parser = argparse.ArgumentParser(description='Convert a model to MLX format')
    parser.add_argument('--run', type=str, required=True,
                       help='Name of the training run to use')
    parser.add_argument('--out-path', type=str, default='output',
                       help='Path for MLX-LM Model output directory')
    args = parser.parse_args()

    # Load run configuration and initialize trainer
    config_path = Path('runs') / args.run / 'config.yaml'
    if not config_path.exists():
        raise ValueError(f"Config not found for run: {args.run}")
    
    trainer = Trainer(str(config_path), for_training=False)

    # Load the config

    
    # Load the final checkpoint
    checkpoint_path = Path('runs') / args.run / 'checkpoints' / 'step_final_model.safetensors'
    if not checkpoint_path.exists():
        checkpoint_path = Path('runs') / args.run / 'checkpoints' / 'step_final.safetensors'
        if not checkpoint_path.exists():
            raise ValueError(f"Final checkpoint not found for run: {args.run}")
    checkpoint_path = str(checkpoint_path)
    
    trainer.model.load_weights(checkpoint_path)

    # Create output directory
    out_dir = Path(args.out_path)
    os.makedirs(out_dir, exist_ok=True)
    
    # Set the output path for the model file
    out_path_model = out_dir / 'model.safetensors'
    
    # Copy the model file
    import shutil
    print(f"Copying model from {checkpoint_path} to {out_path_model}")
    shutil.copy2(checkpoint_path, out_path_model)

    # Copy the tokenizer

    tokenizer_path = Path('runs') / args.run / 'tokenizer' / 'tokenizer.json'

    shutil.copy2(tokenizer_path, out_dir / 'tokenizer.json')
   
    config = {
        "architectures": [
            "LlamaForCausalLM"
        ],
        "attention_bias": False,
        "attention_dropout": 0.0,
    }
    config["attention_bias"] = trainer.config.model.misc['attention_bias']
    config["bos_token_id"] = trainer.tokenizer.tokenize(trainer.config.data.tokenizer['special_tokens']['bos'])[0]
    config["eos_token_id"] = [trainer.tokenizer.tokenize(trainer.config.data.tokenizer['special_tokens']['eos'])[0]]
    #print(trainer.config.model)
    config["hidden_act"] = "silu"
    config["hidden_size"] = trainer.config.model.dimensions["hidden_size"]
    config["intermediate_size"] = trainer.config.model.dimensions["intermediate_size"]
    config["max_position_embeddings"] = trainer.config.data.preprocessing["max_context_size"]
    config["mlp_bias"] = trainer.config.model.misc['mlp_bias']
    config["model_type"] = trainer.config.model.architecture
    config["num_attention_heads"] = trainer.config.model.attention["num_heads"]
    config["num_hidden_layers"] = trainer.config.model.dimensions["num_layers"]
    config["rms_norm_eps"] = trainer.config.model.normalization['rms_norm_eps']
    config["rope_scaling"] = trainer.config.model.rope['scaling']
    config["rope_theta"] = trainer.config.model.rope['theta']
    config["tie_word_embeddings"] = trainer.config.model.misc['tie_word_embeddings']
    config["torch_dtype"] = "float32" # Only support float32 for now
    config["use_cache"] = True
    config["vocab_size"] = trainer.tokenizer.VOCAB_SIZE

    # Save the config
    config_path = out_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    tokenizer_config = {
        "bos_token": trainer.config.data.tokenizer['special_tokens']['bos'],
        "eos_token": trainer.config.data.tokenizer['special_tokens']['eos'],
        "model_input_names": [
            "input_ids",
            "attention_mask"
        ],
        "model_max_length": trainer.config.data.preprocessing["max_context_size"],
        "tokenizer_class": "PreTrainedTokenizerFast",
        

    }

    # Save the tokenizer config
    tokenizer_config_path = out_dir / 'tokenizer_config.json'
    with open(tokenizer_config_path, 'w') as f:
        json.dump(tokenizer_config, f, indent=4)

    # Modify the tokenizer to start with BOS using a post-processor

    tokenizer_path = out_dir / 'tokenizer.json'
    bos_token = tokenizer_config["bos_token"]
    bos_id = trainer.tokenizer.tokenize(bos_token)[0]
    with open(tokenizer_path, 'r') as f:
        tokenizer_data = json.load(f)
        tokenizer_data['post_processor'] = {
            "type": "Sequence",
            "processors": [
                {
                    "type": "TemplateProcessing",
                    "single": [
                    {
                        "SpecialToken": {
                        "id": bos_token,
                        "type_id": 0
                        }
                    },
                    {
                        "Sequence": {
                        "id": "A",
                        "type_id": 0
                        }
                    }
                    ],
                    "pair": [
                    {
                        "SpecialToken": {
                        "id":  bos_token,
                        "type_id": 0
                        }
                    },
                    {
                        "Sequence": {
                        "id": "A",
                        "type_id": 0
                        }
                    },
                    {
                        "SpecialToken": {
                        "id":  bos_token,
                        "type_id": 1
                        }
                    },
                    {
                        "Sequence": {
                        "id": "B",
                        "type_id": 1
                        }
                    }
                    ],
                    "special_tokens": {
                        bos_token: {
                            "id":  bos_token,
                            "ids": [
                                bos_id
                            ],
                            "tokens": [
                                bos_token,
                            ]
                        }
                    }
                }
            ]
        }
        # Save
        with open(tokenizer_path, 'w') as f:
            json.dump(tokenizer_data, f, indent=4)
   

if __name__ == "__main__":
    main()
