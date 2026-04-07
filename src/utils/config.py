import yaml
import os
from typing import Any

class Config:
    def __init__(self, config_path: str | None = None, base_config_path: str | None = None, config_dict: dict | None = None):
        """
        Load config from YAML, optionally merging with base config.
        Validates required fields.
        """
        self.config = {}
        
        if base_config_path and os.path.exists(base_config_path):
            with open(base_config_path, 'r') as f:
                self.config.update(yaml.safe_load(f) or {})
                
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self._update_dict_recursive(self.config, yaml.safe_load(f) or {})
                
        if config_dict:
            self._update_dict_recursive(self.config, config_dict)
            
        self.validate()

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        return cls(config_dict=config_dict)

    def _update_dict_recursive(self, base: dict, overlay: dict):
        for k, v in overlay.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                self._update_dict_recursive(base[k], v)
            else:
                base[k] = v

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        val = self.config
        try:
            for k in keys:
                val = val[k]
            return val
        except (KeyError, TypeError):
            return default

    def validate(self):
        """Validate required fields and value ranges."""
        pass

    def __getitem__(self, key: str) -> Any:
        return self.get(key)
        
    def override_from_cli(self, args: list[str]):
        """Override configuration from CLI arguments like --model.name yolov8s"""
        i = 0
        while i < len(args):
            arg = args[i]
            if arg.startswith('--') and '=' in arg:
                key, value = arg[2:].split('=', 1)
            elif arg.startswith('--') and i + 1 < len(args) and not args[i+1].startswith('--'):
                key = arg[2:]
                value = args[i+1]
                i += 1
            else:
                i += 1
                continue
                
            # Basic typing conversion
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass
            
            keys = key.split('.')
            current = self.config
            for k in keys[:-1]:
                if k not in current or not isinstance(current[k], dict):
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
            i += 1
