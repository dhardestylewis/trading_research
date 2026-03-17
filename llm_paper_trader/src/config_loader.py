import yaml
import os
from pathlib import Path

def load_config():
    # Load config relative to this file's root directory
    base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / "config.yaml"
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    return config

config = load_config()
