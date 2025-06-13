from pydantic import BaseModel, Field

from .image_processor import ImageProcessorConfig
from .tokenizer import TokenizerConfig
from .model import ModelConfig
from .generation import GenerationConfig

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
