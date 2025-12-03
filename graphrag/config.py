import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Manage configuration from YAML file."""
    
    def __init__(self):
        config_path = Path(__file__).parent / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"config.yaml not found in {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.data = yaml.safe_load(f)
    
    def models(self, node: str) -> Dict[str, Any]:
        """Model configuration per node."""
        return self.data['models'][node]
    
    def prompt(self, node: str) -> str:
        """Prompt per node."""
        return self.data['prompts'][node]