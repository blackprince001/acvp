import argparse
import sys
from loguru import logger
from src.utils.config import Config
from src.utils.logging import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Traffic Density Estimation")
    parser.add_argument("--config", type=str, help="Path to experiment config YAML file")
    
    # Unknown arguments will be parsed to override config values
    args, unknown = parser.parse_known_args()
    return args, unknown

def main():
    args, unknown_args = parse_args()
    
    # Load base config
    config = Config(
        config_path=args.config, 
        base_config_path="configs/base.yaml"
    )
    
    # Apply CLI overrides
    if unknown_args:
        config.override_from_cli(unknown_args)
        
    setup_logger(
        log_level=config.get("logging.level", "INFO"), 
        log_file="experiments/logs/run.log"
    )
    
    logger.info("Initializing Real-Time Traffic Density Estimation Pipeline")
    logger.info(f"Mode: {config.get('mode')}")
    
    # Placeholder for actual module invocation
    mode = config.get("mode")
    if mode == "detect":
        logger.info("Starting detection pipeline...")
    elif mode == "segment":
        logger.info("Starting segmentation pipeline...")
    elif mode == "predict":
        logger.info("Starting prediction pipeline...")
    elif mode == "benchmark":
        logger.info("Starting benchmark...")
    else:
        logger.error(f"Unknown execution mode: {mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
