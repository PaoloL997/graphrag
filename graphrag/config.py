"""Configuration management for GraphRAG."""

from pathlib import Path
from typing import Dict, Any
import yaml


class Config:
    """Manage configuration from YAML file."""

    def __init__(self):
        """Initialize configuration from YAML file.

        Raises:
            FileNotFoundError: If config.yaml is not found.
        """
        config_path = Path(__file__).parent / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"config.yaml not found in {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f)

    def models(self, node: str) -> Dict[str, Any]:
        """Model configuration per node.

        Args:
            node: The node name to get model configuration for.

        Returns:
            Dict[str, Any]: Model configuration dictionary.
        """
        return self.data["models"][node]

    def prompt(self, node: str) -> str:
        """Prompt per node.

        Args:
            node: The node name to get prompt for.

        Returns:
            str: The prompt template string.
        """
        return self.data["prompts"][node]
