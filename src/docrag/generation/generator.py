import uuid
from pathlib import Path
from typing import Any

from docrag.schema.config import GeneratorConfig
from docrag.schema.inputs import GeneratorInput
from docrag.schema.outputs import GeneratorOutput, GeneratorInference

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

    def load(self) -> None:
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
        self.load()
        return self.adapter.model

    @property
    def processor(self) -> Any:
        """
        Access the underlying processor/tokenizer instance.
        """
        self.load()
        return self.adapter.processor

    def generate(self, input: GeneratorInput) -> GeneratorOutput:
        """
        Generate a response from the model on a single example.

        Args:
            input (GeneratorInput): Structured input for generation.

        Returns:
            GeneratorOutput: Structured result containing the decoded model output,
            number of generated tokens and elapsed time.
        """
        self.load()
        text, elapsed_seconds, count_tokens = self.adapter.generate(input)
        return GeneratorOutput(
            id=input.id,
            text=text,
            elapsed_seconds=elapsed_seconds,
            count_tokens=count_tokens,
        )

    def generate_batch(
        self,
        inputs: list[GeneratorInput],
        dataset_id: str,
        split: str,
        notes: str | None = None,
    ) -> GeneratorInference:
        """
        Run a batch of inferences and return a full `GeneratorInference` record.

        Args:
            inputs: List of GeneratorInput to process.
            dataset_id: Identifier of the dataset (e.g., 'slidevqa').
            split: Which split these inputs came from ('train', 'val', 'test').
            notes: Optional freeform notes about this run.

        Returns:
            GeneratorInference: A structured record containing metadata and a list of `GeneratorOutput`.
        """
        self.load()

        batch_size = self.config.batch_size or len(inputs)
        results: list[tuple[str, float, int]] = []

        for i in range(0, len(inputs), batch_size):
            batch = inputs[i : i + batch_size]
            raw = self.adapter.batch_generate(batch)
            results.extend(raw)

        outputs: list[GeneratorOutput] = []
        for input, (text, elapsed_seconds, count_tokens) in zip(inputs, results):
            outputs.append(
                GeneratorOutput(
                    id=input.id,
                    text=text,
                    elapsed_seconds=elapsed_seconds,
                    count_tokens=count_tokens,
                )
            )

        return GeneratorInference(
            id=str(uuid.uuid4()),
            dataset_id=dataset_id,
            split=split,
            generator_config=self.config,
            outputs=outputs,
            notes=notes,
        )

    def to(self, device: str) -> None:
        """
        Move the underlying model and processor to the specified device.

        Args:
            device (str): Target device (e.g., 'cpu', 'cuda:0').
        """
        self.load()
        self.adapter.to(device)
        self.config.model.device = device

    def export_config(self, path: Path | str) -> None:
        """
        Export the current settings to a JSON file.

        Args:
            path (str | Path): File path to write settings JSON.
        """
        Path(path).write_text(self.config.model_dump_json(indent=2))
