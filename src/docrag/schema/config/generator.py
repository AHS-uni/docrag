from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from .generation import GenerationConfig
from .image_processor import ImageProcessorConfig
from .model import ModelConfig
from .tokenizer import TokenizerConfig

__all__ = [
    "GeneratorConfig",
]


class GeneratorConfig(BaseModel):
    """
    Top-level configuration for the generation system.

    Attributes:
        model (ModelConfig): Model loading configuration.
        image_processor (ImageProcessorConfig): Preprocessing config for vision inputs.
        tokenizer (TokenizerConfig): Tokenizer configuration.
        generation (GenerationConfig): Generation settings.
        system_prompt (str): System prompt for ChatML-style input.
        prompt_template (str | None): Optional prompt formatting template using `{text}`.
        batch_size (int | None): Number of examples to process in each batch during generation.
        If None, all inputs are processed in a single batch.
    """

    model: ModelConfig
    image_processor: ImageProcessorConfig = Field(default_factory=ImageProcessorConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    system_prompt: str = Field(
        "You are a helpful assistant.", description="System prompt for ChatML input"
    )
    prompt_template: str | None = Field(
        None, description="Prompt template using `{text}` placeholder"
    )
    batch_size: int | None = Field(
        None,
        description="Batch size for generation (process all inputs if None)",
    )

    @field_validator("prompt_template")
    @classmethod
    def _check_template(cls, v: str | None) -> str | None:
        if v is not None and "{text}" not in v:
            raise ValueError("prompt_template must contain a '{text}' placeholder")
        return v

    @classmethod
    def from_yaml(cls, yaml_text: str | bytes | dict[str, Any]) -> "GeneratorConfig":
        """
        Build a GeneratorConfig from a YAML *string* / *bytes* / *dict*.
        """
        data: dict[str, Any]
        if isinstance(yaml_text, dict):
            data = yaml_text
        else:
            data = yaml.safe_load(yaml_text)
        return cls.model_validate(data)

    @classmethod
    def from_yaml_path(cls, path: str | Path) -> "GeneratorConfig":
        """Shortcut for `GeneratorConfig.from_yaml(Path.read_text())`."""
        return cls.from_yaml(Path(path).read_text())

    def to_yaml(
        self,
        path: str | Path | None = None,
        *,
        sort_keys: bool = False,
        **yaml_kwargs: Any,
    ) -> str:
        """
        Serialize the current config to YAML.

        Args:
            path (str | Path | None): If provided, the YAML text is also written to this file.
            sort_keys (bool): Pass-through to `yaml.safe_dump`.
            yaml_kwargs (Any): Additional kwargs forwarded to `yaml.safe_dump`.

        Returns:
            str: YAML representation.
        """
        text = yaml.safe_dump(
            self.model_dump(exclude_none=True),
            sort_keys=sort_keys,
            **yaml_kwargs,
        )
        if path is not None:
            Path(path).write_text(text)
        return text
