#!/usr/bin/env python
# Client script to launch MLX distributed training on Modal A100 GPUs

import os
import sys
import argparse
import yaml
import time
import json
import logging
from pathlib import Path
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('modal_client.log')
    ]
)
logger = logging.getLogger('modal_client')

def parse_args():
    parser = argparse.ArgumentParser(description="Launch MLX training on A100 GPUs via Modal")
    parser.add_argument("--config", type=str, required=True, help="Path to model configuration YAML")
    parser.add_argument("--run-id", type=str, default=None, help="Unique ID for this run (generated if not provided)")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="Path to tokenizer directory")
    parser.add_argument("--data-path", type=str, default=None, help="Path to data directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--workers", type=str, default="remote_worker_config.json", help="Path to workers configuration file")
    
    # Parse and log arguments
    args, unknown = parser.parse_known_args()
    if unknown:
        logger.warning(f"Unknown arguments: {unknown}")
    
    logger.info(f"Command line arguments: {sys.argv}")
    logger.info(f"Parsed arguments: {args}")
    
    return args

def main():
    args = parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        logger.error(f"Config file '{args.config}' not found!")
        sys.exit(1)
    
    # Generate a run ID if not provided
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    logger.info(f"Using run ID: {run_id}")
    
    # Prepare data directory
    data_dir = args.data_path or os.getcwd()
    logger.info(f"Using data directory: {data_dir}")
    
    # Check for required files
    required_files = [
        os.path.join(data_dir, "train.jsonl"),
        os.path.join(data_dir, "val.jsonl")
    ]
    
    logger.debug(f"Checking for required files: {required_files}")
    for file_path in required_files:
        if not os.path.exists(file_path):
            logger.error(f"Required file '{file_path}' not found!")
            sys.exit(1)
        else:
            logger.debug(f"Found required file: {file_path}")
    
    # Load config to get model details
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded config from {args.config}")
        logger.debug(f"Config contents: {config}")
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        sys.exit(1)
    
    model_name = config.get("name", "MLX-Model")
    logger.info(f"Preparing to train model: {model_name}")
    
    # Check model size (approximately)
    hidden_size = config.get("model", {}).get("dimensions", {}).get("hidden_size", 0)
    num_layers = config.get("model", {}).get("dimensions", {}).get("num_layers", 0)
    vocab_size = config.get("data", {}).get("tokenizer", {}).get("normal_vocab_size", 0)
    
    if not all([hidden_size, num_layers, vocab_size]):
        logger.warning(f"Some model dimensions missing: hidden_size={hidden_size}, num_layers={num_layers}, vocab_size={vocab_size}")
    
    # Rough parameter count calculation
    param_count = (
        12 * hidden_size * hidden_size * num_layers +  # Transformer blocks
        2 * hidden_size * vocab_size                  # Embedding & output
    ) / 1_000_000  # Convert to millions
    
    logger.info(f"Estimated model size: ~{param_count:.2f}M parameters")
    logger.info(f"Using {num_layers} layers with {hidden_size} hidden dimension")
    
    # Create temp directory for uploads and logs directory
    temp_dir = Path("./temp_modal_upload")
    logs_dir = Path("./logs")
    runs_dir = Path("./runs")
    
    try:
        os.makedirs(temp_dir, exist_ok=True)
        logger.debug(f"Created temp directory: {temp_dir}")
        os.makedirs(logs_dir, exist_ok=True)
        logger.debug(f"Created logs directory: {logs_dir}")
        os.makedirs(runs_dir, exist_ok=True)
        logger.debug(f"Created runs directory: {runs_dir}")
    except Exception as e:
        logger.error(f"Failed to create directories: {e}")
        sys.exit(1)
    
    # Ensure run directories exist
    run_dir_name = f"Muon-400M-{run_id}"
    run_path = os.path.join(runs_dir, run_dir_name)
    try:
        os.makedirs(run_path, exist_ok=True)
        logger.info(f"Created run directory: {run_path}")
    except Exception as e:
        logger.error(f"Failed to create run directory: {e}")
        sys.exit(1)
    
    # Copy config file to temp directory
    config_path = os.path.join(temp_dir, os.path.basename(args.config))
    try:
        shutil.copy(args.config, config_path)
        logger.debug(f"Copied config to temp directory: {config_path}")
    except Exception as e:
        logger.error(f"Failed to copy config to temp directory: {e}")
        sys.exit(1)
    
    # Copy config file to working directory if it's not already there
    local_config_path = os.path.basename(args.config)
    if not os.path.exists(local_config_path) or os.path.normpath(args.config) != os.path.normpath(local_config_path):
        try:
            shutil.copy(args.config, local_config_path)
            logger.info(f"Copied config to local directory: {local_config_path}")
        except Exception as e:
            logger.error(f"Failed to copy config to local directory: {e}")
            sys.exit(1)
    
    # Load workers configuration if provided
    workers_config = None
    if args.workers and os.path.exists(args.workers):
        try:
            with open(args.workers, 'r') as f:
                workers_config = json.load(f)
            logger.info(f"Loaded workers configuration from {args.workers}")
            logger.debug(f"Workers config: {workers_config}")
        except Exception as e:
            logger.error(f"Error loading workers configuration: {e}")
    else:
        logger.warning(f"Workers configuration file not found: {args.workers}")
    
    # Set up Modal
    logger.info("Initializing Modal...")
    try:
        # Import the train_model_a100 function from train_a100.py
        logger.debug("Importing Modal components from train_a100.py")
        from train_a100 import train_model_a100, app
        
        # Deploy and run on Modal
        logger.info(f"Deploying to Modal with run ID: {run_id}")
        try:
            # Set Modal debug mode
            os.environ["MODAL_DEBUG"] = "1"
            
            with app.run():
                logger.debug("Calling train_model_a100.remote")
                result = train_model_a100.remote(
                    config_path=os.path.basename(args.config),
                    data_dir=data_dir,
                    run_id=run_id,
                    checkpoint=args.checkpoint
                )
                logger.info(f"Training initiated successfully: {result}")
        except Exception as e:
            logger.error(f"Error during Modal app.run(): {e}")
            logger.error(f"Full exception details: {str(e)}")
            # Print more details about the exception
            import traceback
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            sys.exit(1)
            
    except ImportError as e:
        logger.error(f"Error importing Modal components: {e}")
        logger.error("Make sure train_a100.py is in the correct location and contains required components")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error launching Modal training: {e}")
        logger.error(f"Full exception details: {str(e)}")
        sys.exit(1)
    finally:
        # Clean up temp directory
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.debug(f"Cleaned up temp directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Error cleaning up temp directory: {e}")
    
    logger.info(f"Training job submitted to Modal with run ID: {run_id}")
    logger.info(f"Logs will be available in: runs/{model_name}-{run_id}/ and logs/")
    logger.info("You can monitor the job status in the Modal dashboard: https://modal.com/apps")

if __name__ == "__main__":
    try:
        logger.info("Starting modal_client.py")
        logger.info(f"System arguments: {sys.argv}")
        main()
        logger.info("modal_client.py completed successfully")
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {e}")
        logger.critical(f"Full exception details: {str(e)}", exc_info=True)
        sys.exit(1)