from PIL import Image
from pydantic import BaseModel, Field, model_validator, ConfigDict

class GeneratorInput(BaseModel):
    """
    Input schema for Generator models.

    Attributes:
        id (str): External identifier.
        text (str): Question/prompt.
        images (list[Image.Image]): Zero or more PIL images.

    Raises:
        ValueError: If `text` is empty.
    """

    id: str
    text: str
    images: list[Image.Image] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _ensure_non_empty(self):
        if not self.text.strip():
            raise ValueError("`text` cannot be empty")
        return self
