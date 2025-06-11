"""
Defines the Generator facade, wrapping any registered VLM adapter
for inference. Receives GeneratorSettings and exposes generate/batch_generate/to, etc.
"""

from pathlib import Path
from typing import Any

from PIL import Image

from docrag.schema.config import GeneratorConfig
from .adapter import Adapter
from .registry import get_adapter


class Generator:
    """
    High-level facade for multimodal text generation using Vision-Language models.

    Attributes:
        config (GeneratorConfig): Validated configuration for model, tokenizer, etc.
    """

    def __init__(self, config: GeneratorConfig):
        """
        Initialize Generator.

        Args:
            settings (GeneratorSettings): Configuration for generator.
        """
        self.config = config
        self.adapter: Adapter | None = None

    def init_adapter(self) -> None:
        """
        Lazily load and instantiate the appropriate adapter based on model name.
        """
        if self.adapter is None:
            AdapterCls = get_adapter(self.config.model.name)
            self.adapter = AdapterCls(self.config)

    @property
    def model(self) -> Any:
        """
        Access the underlying model instance.
        """
        self.init_adapter()
        return self.adapter.model

    @property
    def processor(self) -> Any:
        """
        Access the underlying processor/tokenizer instance.
        """
        self.init_adapter()
        return self.adapter.processor

    def generate(
        self,
        images: list[Image.Image],
        text: str = "",
    ) -> str:
        """
        Generate a response on a single example from the model.

        Args:
            images (list[Image.Image]): Input images for inference.
            text (str): Textual question or prompt to accompany images.

        Returns:
            str: Decoded model output.
        """
        self.init_adapter()
        return self.adapter.generate(images=images, text=text)

    # def batch_generate(self, batch: list[dict[str, Any]]) -> list[str]:
    #     """
    #     Batch-generate responses for multiple examples.

    #     Args:
    #         batch (list[dict[str, Any]]): List of dicts. Each dict should have:
    #             - "images": list of PIL images
    #             - "text":   prompt string

    #     Returns:
    #         list[str]: List of decoded outputs for each example.
    #     """
    #     return [
    #         self.generate(
    #             images=example.get("images", []),
    #             text=example.get("text", "")
    #         )
    #         for example in batch
    #     ]

    def to(self, device: str) -> None:
        """
        Move the underlying model and processor to the specified device.

        Args:
            device (str): Target device (e.g., 'cpu', 'cuda:0').
        """
        self.init_adapter()
        self.adapter.to(device)
        self.config.model.device = device

    def export_config(self, path: Path | str) -> None:
        """
        Export the current settings to a JSON file.

        Args:
            path (str | Path): File path to write settings JSON.
        """
        Path(path).write_text(self.config.model_dump_json(indent=2))
