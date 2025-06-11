from abc import ABC, abstractmethod
from typing import Any

import torch
from PIL import Image

from docrag.schema.config import GeneratorConfig


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
        self._load()

    @abstractmethod
    def _load(self) -> None:
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
    def generate(
        self,
        images: list[Image.Image],
        text: str,
    ) -> str:
        """
        Run a single forward+decode pass.

        Args:
            images (list[Image.Image]): List of PIL images to use as visual context.
            text (str): Prompt/question string.

        Returns:
            str: Decoded model output.
        """
        ...

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
