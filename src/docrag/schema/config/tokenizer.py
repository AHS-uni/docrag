from typing import Literal, Any

from pydantic import BaseModel, Field

__all__ = [
    "TokenizerConfig",
]


class TokenizerConfig(BaseModel):
    """
    Configuration for text tokenization.

    Attributes:
        padding_side (Literal['left','right']): Side to pad tokens.
        truncation_side (Literal['left','right']): Side to truncate tokens.
        model_max_length (int): Maximum token length.
        pad_token (str | None): Custom pad token string.
        extra (dict[str, Any]): Additional model specific configurations.
    """

    padding_side: Literal["left", "right"] = Field(
        "right", description="Side to add padding tokens"
    )
    truncation_side: Literal["left", "right"] = Field(
        "right", description="Side to truncate tokens"
    )
    model_max_length: int = Field(
        4096, ge=1, description="Maximum allowed token length"
    )
    pad_token: str | None = Field(None, description="Override pad token id or string")
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Additional model specific configurations."
    )

    def to_kwargs(
        self, *, exclude_none: bool = True, exclude_defaults: bool = False
    ) -> dict[str, object]:
        """
        Prepare kwargs for a Hugging Face tokenizer.

        Args:
            exclude_none (bool): Drop None fields.
            exclude_defaults (bool): Drop default fields.
        Returns:
            dict[str, object]: Tokenizer keyword arguments.
        """
        data = self.model_dump(exclude_none=exclude_none, exclude_defaults=exclude_defaults)
        data.update(self.extra)
        return data
