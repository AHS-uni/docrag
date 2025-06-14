import json
import time
from pathlib import Path
from typing import Iterator
import uuid

from PIL import Image
from tqdm.auto import tqdm
from datasets import Dataset
from more_itertools import chunked

from docrag.generation.generator import Generator
from docrag.schema.config import GeneratorConfig
from docrag.schema.inputs import GeneratorInput
from docrag.schema.outputs import GeneratorInference


def run_inference(
    generator: Generator | str | Path | GeneratorConfig,
    dataset: Dataset,
    *,
    id_field: str = "id",
    text_field: str = "question",
    images_field: str = "images",
    out_dir: str | Path,
    notes: str | None = None,
    write_jsonl: bool = True,
    show_progress: bool = True,
) -> GeneratorInference:
    """
    Execute a full inference pass with a `Generator` and persist artefacts.

    Args:
        generator (Generator | str | Path | GeneratorConfig):
            An already-initialised :class:`~docrag.generation.generator.Generator`,
            *or* a path/str pointing to a YAML config file,
            *or* a ready :class:`~docrag.schema.config.GeneratorConfig`.
        inputs (Sequence[GeneratorInput]):
            Ordered list of inputs to process.
        dataset_id (str):
            Logical identifier of the dataset (e.g. ``"slidevqa"``).
        split (str):
            Dataset split label (``"train"``, ``"val"``, ``"test"``, etc.).
        out_dir (str | Path):
            Directory in which to write all run artefacts. Created if missing.
        notes (str | None, optional):
            Free-form comment recorded inside the resulting
            :class:`~docrag.schema.outputs.GeneratorInference`.
        write_jsonl (bool, default=True):
            If *True*, write ``results.jsonl`` (one output per line) in *out_dir*.
        show_progress (bool, default=True):
            Display a tqdm progress bar while generating.

    Returns:
        GeneratorInference: Complete structured record containing
           - unique run ID,
           - generator configuration snapshot,
           - per-example outputs with timing,
           - optional user notes.

    Side Effects:
        Creates the following files under *out_dir*:
            - ``inference.json``: Full JSON dump of the returned object
            - ``generator.yaml``: YAML snapshot of the config actually used
            - ``results.jsonl``: newline-delimited outputs (if *write_jsonl*)
            - ``meta.json``: aggregate stats (n_examples, runtime)
    """

    if isinstance(generator, Generator):
        generator = generator
    else:  # YAML path or config → build new Generator instance
        config = (
            GeneratorConfig.from_yaml_path(generator)
            if isinstance(generator, (str, Path))
            else generator  # already a config
        )
        generator = Generator(config)

    info = dataset.info
    dataset_id = str(getattr(info, "dataset_name", "") or getattr(
        info, "builder_name", ""
    ))
    split = str(getattr(dataset, "split", ""))

    def _iter_inputs():
        for row in dataset:
            raw = row[images_field]
            images = [raw] if isinstance(raw, Image.Image) else list(raw)
            yield GeneratorInput(
                id=str(row[id_field]),
                text=row[text_field],
                images=images,
            )

    iterator = _iter_inputs()
    total = len(dataset)
    bar = tqdm(total=total, desc="generating", unit="ex") if show_progress else None

    outputs = []
    start = time.perf_counter()
    for batch in chunked(iterator, generator.config.batch_size or 1):
        inference_batch = generator.generate_batch(
            inputs=list(batch),
            dataset_id=dataset_id,
            split=split,
            notes=notes,
        )
        outputs.extend(inference_batch.outputs)
        if bar:
            bar.update(len(batch))
    total_elapsed = time.perf_counter() - start
    if bar:
        bar.close()

    record = GeneratorInference(
        id=str(uuid.uuid4()),
        dataset_id=dataset_id,
        split=split,
        generator_config=generator.config,
        outputs=outputs,
        notes=notes,
    )

    # --- Write artefacts ---
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "inference.json").write_text(
        record.model_dump_json(indent=2, exclude_none=True)
    )
    generator.config.to_yaml(out / "generator.yaml")
    if write_jsonl:
        with (out / "results.jsonl").open("w", encoding="utf-8") as f:
            for o in record.outputs:
                f.write(o.model_dump_json(exclude_none=True) + "\n")
    (out / "meta.json").write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "split": split,
                "count_examples": len(record.outputs),
                "total_elapsed_seconds": total_elapsed,
                "mean_elapsed_seconds": total_elapsed / max(len(record.outputs), 1),
            },
            indent=2,
        )
    )

    return record
