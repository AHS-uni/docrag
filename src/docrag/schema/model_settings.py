"""
Pydantic models for configuring a vision-language model.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator
from transformers.generation.configuration_utils import GenerationConfig

__all__ = [
    "GeneratorSettings",
    "ModelConfig",
    "ImageProcessorConfig",
    "TokenizerConfig",
]


class ImageProcessorConfig(BaseModel):
    """
    Configuration for image preprocessing before model input.

    Attributes:
        resize (bool): Whether to resize input images.
        size (dict[str, int]): Target image size with "height" and "width".
        crop_strategy (Literal['center', 'random', 'none']): Crop method applied after resizing.
        rescale (bool): Whether to rescale pixel values.
        rescale_factor (float): Scaling factor (e.g., 1/255).
        normalize (bool): Whether to normalize using mean and std.
        mean (list[float]): RGB channel-wise means for normalization.
        std (list[float]): RGB channel-wise stds for normalization.
    """

    resize: bool = Field(True, description="Enable resizing")
    size: dict[str, int] = Field(
        default_factory=lambda: {"height": 768, "width": 768},
        description="Target size as {'height': int, 'width': int}",
    )
    crop_strategy: Literal["center", "random", "none"] = Field(
        "center", description="Cropping strategy after resize"
    )
    rescale: bool = Field(False, description="Enable rescaling")
    rescale_factor: float = Field(
        1 / 255, gt=0, description="Factor for pixel rescaling"
    )
    normalize: bool = Field(True, description="Enable normalization")
    mean: list[float] = Field(
        default_factory=lambda: [0.485, 0.456, 0.406], description="RGB mean values"
    )
    std: list[float] = Field(
        default_factory=lambda: [0.229, 0.224, 0.225], description="RGB std values"
    )

    @model_validator(mode="after")
    def _validate_and_normalize(self):
        if set(self.size.keys()) != {"height", "width"}:
            raise ValueError("size must be a dict with keys {'height', 'width'}")
        if self.size["height"] <= 0 or self.size["width"] <= 0:
            raise ValueError("height and width must be positive integers")
        return self


class TokenizerConfig(BaseModel):
    """
    Configuration for text tokenization and encoding.

    Attributes:
        padding_side (Literal['left','right']): Side to add pad tokens.
        model_max_length (int): Length threshold for pad/truncate.
        pad_token (Optional[str]): Override pad token.
        return_tensors (Literal['pt','np','tf']): Tensor type to return.
    """

    padding_side: Literal["left", "right"] = Field("right", description="Pad side")
    model_max_length: int = Field(4096, ge=1, description="Max token length")
    pad_token: str | None = Field(None, description="Custom pad token")
    return_tensors: Literal["pt", "np", "tf"] = Field("pt", description="Tensor type")


class ModelConfig(BaseModel):
    """
    Configuration for loading a Hugging Face model.

    Attributes:
        name (str): Model alias or identifier.
        path (str): HF hub ID or local path to model.
        dtype (Literal['float16', 'bfloat16', 'float32']): Precision of model weights.
        device (str): Torch device string (e.g., 'cuda', 'cpu').
        trust_remote_code (bool): Whether to allow loading custom model code.
        device_map (str | dict[str, Any] | None): Strategy for model sharding across devices.
        cache_dir (str | None): Cache directory for downloaded files.
        low_cpu_mem_usage (bool): Use optimized loading to reduce CPU memory usage.
    """

    name: str = Field(..., description="Model name in registry or system")
    path: str = Field(..., description="Hugging Face hub ID or local model path")
    dtype: Literal["float16", "bfloat16", "float32"] = Field(
        "float16", description="Precision for model weights"
    )
    device: str = Field("cuda", description="Target device for inference")
    trust_remote_code: bool = Field(False, description="Allow loading custom code")
    device_map: str | dict[str, Any] | None = Field(
        "auto", description="Device mapping strategy"
    )
    cache_dir: str | None = Field(None, description="Cache directory for model files")
    low_cpu_mem_usage: bool = Field(
        True, description="Use memory-efficient model loading"
    )

    @model_validator(mode="after")
    def _validate_device(self):
        allowed = {"cpu", "cuda", "mps"}
        if self.device not in allowed and not self.device.startswith("cuda:"):
            raise ValueError("device must be 'cpu', 'cuda', 'mps', or 'cuda:<idx>'")
        return self


class GeneratorSettings(BaseModel):
    """
    Top-level configuration for the generation system.

    Attributes:
        model (ModelConfig): Model loading configuration.
        image_processor (ImageProcessorConfig): Preprocessing config for vision inputs.
        tokenizer (TokenizerConfig): Tokenizer configuration.
        generate (GenerationConfig): Hugging Face-compatible generation settings.
        system_prompt (str): System prompt for ChatML-style input.
        prompt_template (str | None): Optional prompt formatting template using `{text}`.
    """

    model: ModelConfig
    image_processor: ImageProcessorConfig = Field(default_factory=ImageProcessorConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    generate: GenerationConfig = Field(default_factory=GenerationConfig)
    system_prompt: str = Field(
        "You are a helpful assistant.", description="System prompt for ChatML input"
    )
    prompt_template: str | None = Field(
        None, description="Prompt template using `{text}` placeholder"
    )

    class Config:
        arbitrary_types_allowed = True
