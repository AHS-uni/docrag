import json
import uuid
import time
from pathlib import Path

from datasets import Dataset
from anls import anls_score
from tqdm.auto import tqdm

from docrag.schema.outputs import GeneratorInference
from docrag.schema.outputs.generator import GeneratorEvaluate, EvaluateOutput


def run_evaluation(
    inference: GeneratorInference,
    dataset: Dataset,
    *,
    id_field: str = "question_id",
    answer_field: str = "answer_variants",  # list[str]
    out_dir: str | Path,
    write_jsonl: bool = True,
    anls_threshold: float = 0.5,
) -> GeneratorEvaluate:
    """
    Stream through the dataset and inference.outputs in tandem,
    compute ANLS via the `anls` package, and write out evaluation artifacts.
    """
    eval_outputs: list[EvaluateOutput] = []
    scores: list[float] = []
    start = time.perf_counter()

    # zip the dataset rows with your outputs list
    for idx, (row, out) in enumerate(
        tqdm(
            zip(dataset, inference.outputs),
            total=len(inference.outputs),
            desc="Evaluating",
            unit="ex",
        )
    ):
        # sanity check: make sure row[id_field] == out.id
        if str(row[id_field]) != out.id:
            raise ValueError(
                f"ID mismatch at index {idx}: dataset has {row[id_field]!r}, "
                f"but inference.outputs[{idx}].id is {out.id!r}"
            )

        pred = out.text.strip()
        variants = row[answer_field]  # list of acceptable answers

        score = anls_score(
            prediction=pred, gold_labels=variants, threshold=anls_threshold
        )
        scores.append(score)

        eval_outputs.append(
            EvaluateOutput(
                id=out.id,
                prediction=pred,
                answer_variants=variants,
                anls=score,
            )
        )

    total_time = time.perf_counter() - start
    avg_anls = sum(scores) / len(scores) if scores else 0.0

    record = GeneratorEvaluate(
        id=str(uuid.uuid4()),
        generator_config=inference.generator_config,
        avg_anls=avg_anls,
        total_elapsed_seconds=total_time,
        per_example=eval_outputs,
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # full JSON
    (out / "evaluation.json").write_text(
        record.model_dump_json(indent=2, exclude_none=True)
    )

    # newline-delimited per-example
    if write_jsonl:
        with (out / "results.jsonl").open("w", encoding="utf-8") as f:
            for ex in record.per_example:
                f.write(ex.model_dump_json(exclude_none=True) + "\n")

    # meta summary
    (out / "meta.json").write_text(
        json.dumps(
            {
                "average_anls": avg_anls,
                "count": len(scores),
                "total_elapsed_seconds": total_time,
                "mean_per_example": total_time / max(len(scores), 1),
            },
            indent=2,
        )
    )

    return record
