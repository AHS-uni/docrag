import json
import time
from pathlib import Path

from tqdm.auto import tqdm

from docrag.generation.generator import Generator
from docrag.schema.config import GeneratorConfig
from docrag.schema.inputs import GeneratorInput
from docrag.schema.outputs import GeneratorInference


def run_inference(
    generator: Generator | str | Path | GeneratorConfig,
    inputs: list[GeneratorInput],
    *,
    dataset_id: str,
    split: str,
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

    start = time.perf_counter()
    if show_progress:
        bar = tqdm(total=len(inputs), desc="generating", unit="ex")

    inference = generator.generate_batch(
        inputs=list(inputs),  # typing.Sequence → list
        dataset_id=dataset_id,
        split=split,
        notes=notes,
    )
    total_elapsed = time.perf_counter() - start

    if show_progress:
        bar.update(len(inputs))
        bar.close()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path_json = out_dir / "inference.json"
    path_json.write_text(inference.model_dump_json(indent=2, exclude_none=True))

    generator.config.to_yaml(out_dir / "generator.yaml")

    if write_jsonl:
        path_jsonl = out_dir / "results.jsonl"
        with path_jsonl.open("w", encoding="utf-8") as f:
            for out in inference.outputs:
                f.write(out.model_dump_json(exclude_none=True) + "\n")

    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "split": split,
                "count_examples": len(inputs),
                "total_elapsed_seconds": total_elapsed,
                "mean_elapsed_seconds": total_elapsed / max(len(inputs), 1),
            },
            indent=2,
        )
    )

    return inference
