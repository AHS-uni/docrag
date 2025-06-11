from typing import Literal, Any

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "ModelConfig",
]

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
        attn_implementation (str | None): Specific attention implementation (e.g., 'flash_attention_2').
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
    attn_implementation: str | None = Field(
        None, description="Attention implementation to use, e.g., 'flash_attention_2'"
    )

    @model_validator(mode="after")
    def _validate_device(self):
        allowed = {"cpu", "cuda", "mps"}
        if self.device not in allowed and not self.device.startswith("cuda:"):
            raise ValueError("device must be 'cpu', 'cuda', 'mps', or 'cuda:<idx>'")
        return self
