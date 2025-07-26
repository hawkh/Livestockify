"""
Configuration manager for loading and validating configurations.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import os


@dataclass
class ConfigManager:
    """Manages configuration loading and validation."""
    
    config_dir: str = "src/utils/config"
    
    def load_config(self, config_name: str, config_type: str = "yaml") -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_name: Name of config file (without extension)
            config_type: Type of config file (yaml, json)
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(self.config_dir) / f"{config_name}.{config_type}"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_type.lower() == 'yaml':
                return yaml.safe_load(f)
            elif config_type.lower() == 'json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config type: {config_type}")
    
    def save_config(
        self, 
        config: Union[Dict[str, Any], object], 
        config_name: str, 
        config_type: str = "yaml"
    ) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration data or dataclass object
            config_name: Name of config file (without extension)
            config_type: Type of config file (yaml, json)
        """
        config_path = Path(self.config_dir) / f"{config_name}.{config_type}"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert dataclass to dict if needed
        if hasattr(config, '__dataclass_fields__'):
            config_dict = asdict(config)
        else:
            config_dict = config
        
        with open(config_path, 'w') as f:
            if config_type.lower() == 'yaml':
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif config_type.lower() == 'json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported config type: {config_type}")
    
    def load_from_env(self, prefix: str = "CHICKEN_WEIGHT_") -> Dict[str, Any]:
        """
        Load configuration from environment variables.
        
        Args:
            prefix: Prefix for environment variables
            
        Returns:
            Configuration dictionary from environment
        """
        config = {}
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                # Try to parse as different types
                if value.lower() in ('true', 'false'):
                    config[config_key] = value.lower() == 'true'
                elif value.isdigit():
                    config[config_key] = int(value)
                elif self._is_float(value):
                    config[config_key] = float(value)
                else:
                    config[config_key] = value
        return config
    
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple configuration dictionaries.
        Later configs override earlier ones.
        
        Args:
            *configs: Configuration dictionaries to merge
            
        Returns:
            Merged configuration dictionary
        """
        merged = {}
        for config in configs:
            merged.update(config)
        return merged
    
    def validate_config(self, config: Dict[str, Any], required_keys: list) -> bool:
        """
        Validate that configuration contains required keys.
        
        Args:
            config: Configuration to validate
            required_keys: List of required keys
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
        return True
    
    @staticmethod
    def _is_float(value: str) -> bool:
        """Check if string can be converted to float."""
        try:
            float(value)
            return True
        except ValueError:
            return False