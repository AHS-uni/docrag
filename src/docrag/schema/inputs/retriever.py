from PIL import Image
from pydantic import BaseModel, model_validator, ConfigDict


class RetrieverInput(BaseModel):
    """
    Input schema for Retriever models.

    Attributes:
        id (str): External identifier for the input.
        text (str): Question/prompt.
        images (list[Image.Image]): One or more PIL images.

    Raises:
        ValueError: If `text` or `images` is empty.
    """

    id: str
    text: str
    images: list[Image.Image]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _ensure_non_empty(self):
        if not self.text.strip():
            raise ValueError("`text` cannot be empty")
        if not self.images:
            raise ValueError("`images` cannot be empty")
        return self
