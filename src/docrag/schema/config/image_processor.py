from typing import Literal, Any

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "ImageProcessorConfig",
]


class ImageProcessorConfig(BaseModel):
    """
    Configuration for image preprocessing.

    Attributes:
        resize (bool): Whether to resize images.
        size (Dict[str,int]): {'height': H, 'width': W} target size.
        crop_strategy (Literal['center','random','none']): Crop after resize.
        rescale (bool): Whether to rescale pixel values.
        rescale_factor (float): Scaling factor for pixels.
        normalize (bool): Whether to normalize images.
        mean (list[float]): RGB mean values.
        std (list[float]): RGB std values.
        extra (dict[str, Any]): Additional model specific configurations.
    """

    resize: bool = Field(True, description="Enable resizing of images")
    size: dict[str, int] = Field(
        default_factory=lambda: {"height": 768, "width": 768},
        description="Target size as {'height': int, 'width': int}",
    )
    crop_strategy: Literal["center", "random", "none"] = Field(
        "center", description="Cropping strategy after resizing"
    )
    rescale: bool = Field(False, description="Enable pixel rescaling")
    rescale_factor: float = Field(
        1 / 255, gt=0, description="Factor to rescale pixel values"
    )
    normalize: bool = Field(True, description="Enable normalization")
    mean: list[float] = Field(
        default_factory=lambda: [0.485, 0.456, 0.406], description="RGB mean values"
    )
    std: list[float] = Field(
        default_factory=lambda: [0.229, 0.224, 0.225], description="RGB std values"
    )
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Additional model specific configurations."
    )

    @model_validator(mode="after")
    def _validate(self):
        if set(self.size.keys()) != {"height", "width"}:
            raise ValueError("size must have 'height' and 'width'")
        return self

    def to_kwargs(
        self, *, exclude_none: bool = True, exclude_defaults: bool = False
    ) -> dict[str, object]:
        """
        Prepare kwargs for a Hugging Face image processor.

        Args:
            exclude_none (bool): Drop None fields.
            exclude_defaults (bool): Drop default fields.
        Returns:
            dict[str, object]: Image processor kwargs.
        """
        data = self.model_dump(exclude_none=exclude_none, exclude_defaults=exclude_defaults)
        strategy = data.pop("crop_strategy", None)
        if strategy == "center":
            data["do_center_crop"] = True
            data["crop_size"] = data["size"]
            data["do_random_crop"] = False
        elif strategy == "random":
            data["do_center_crop"] = False
            data["do_random_crop"] = True
            data["crop_size"] = data["size"]
        else:
            data["do_center_crop"] = False
            data["do_random_crop"] = False
        data.update(self.extra)
        return data
