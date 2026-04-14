from dataclasses import dataclass


@dataclass
class PromptsConfig:
    """Configuration for customizable prompts. Defaults to built-in prompts."""

    generate_response: str
    evaluate_context: str
    refine_query: str
