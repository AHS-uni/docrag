from datetime import datetime, timezone

from pydantic import BaseModel, Field

from ..config import GeneratorConfig

class GeneratorOutput(BaseModel):
    """
    Output of a single model generation..

    Attributes:
        id (str): External identifier.
        text (str): Decoded model output.
        count_tokens (int): Number of generated tokens.
        elapsed_seconds (float): Wall-clock seconds for generation.
    """

    id: str
    text: (str)
    count_tokens: int
    elapsed_seconds: float

class GeneratorInference(BaseModel):
    """
    Metadata and outputs for a generator inference run.

    Attributes:
        id (str): Identifier for this inference run.
        dataset_id (str): Identifier for the dataset used.
        split (str): Dataset split (e.g., 'train', 'val', 'test').
        generator_config (GeneratorConfig): Configuration used for generator.
        timestamp (datetime): UTC timestamp of when the run was executed.
        notes (str | None): Notes or comments about this run.
        outputs (list[GeneratorOutput]): List of generation results.
    """

    id: str
    dataset_id: str
    split: str
    generator_config: GeneratorConfig
    notes: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    outputs: list[GeneratorOutput]

class EvaluateOutput(BaseModel):
    """
    Per‐example evaluation result.
    """

    id: str
    prediction: str
    answer_variants: list[str]
    anls: float


class GeneratorEvaluate(BaseModel):
    """
    Aggregate evaluation record for a single run.
    """

    id: str
    generator_config: GeneratorConfig
    avg_anls: float
    total_elapsed_seconds: float
    per_example: list[EvaluateOutput]
