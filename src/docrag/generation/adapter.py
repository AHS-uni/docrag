from abc import ABC, abstractmethod
from typing import Any

import torch

from docrag.schema.config import GeneratorConfig
from docrag.schema.inputs import GeneratorInput


class Adapter(ABC):
    """
    Abstract base for all Vision–Language model adapters.
    """

    def __init__(self, config: GeneratorConfig):
        """
        Args:
            settings: fully-validated Pydantic config for model, LoRA, image, tokenizer, generation.
        """
        self.config = config
        self.model: torch.nn.Module
        self.processor: Any
        self.load()

    @abstractmethod
    def load(self) -> None:
        """
        Load model & processor/tokenizer according to `self.settings.model`.
        """
        ...

    def to(self, device: str) -> None:
        """
        Move model to a new device (e.g. 'cuda:1' or 'cpu').
        """
        self.model.to(torch.device(device))
        self.config.model.device = device

    @abstractmethod
    def generate(self, input: GeneratorInput) -> tuple[str, float, int]:
        """
        Run inference on a single input using the underlying model.

        Args:
            input (GeneratorInput): Structured input for generation,
            including text, and images.

        Returns:
            tuple[str, float, int]: A tuple of
                - text (str): Decoded model output
                - elapsed_seconds (float): Time taken for generation in seconds
                - count_tokens (int): Number of generated tokens
        """
        ...

    def batch_generate(
        self, batch_input: list[GeneratorInput]
    ) -> list[tuple[str, float, int]]:
        """
        Run inference over a list (batch) of inputs.

        Args:
            inputs (list): List of GeneratorInput objects.
        Returns:
            list[tuple[str, float, int]]:
                A list of tuples (text, elapsed_seconds, count_tokens).
        """
        return [self.generate(inp) for inp in batch_input]

    def _apply_prompt_template(self, text: str) -> str:
        """
        If settings.prompt_template is set, apply it
        (e.g. Jinja2 or Python .format) to produce the final prompt.
        """
        template = self.config.prompt_template
        if template:
            try:
                return template.format(text=text)
            except Exception:
                return template.replace("{text}", text)
        return text
