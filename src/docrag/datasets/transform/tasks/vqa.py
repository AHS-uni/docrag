from .registry import register

from ..core import filter_dataset, project_fields, add_images_column

__all__ = [
    "_transform_vqa",
]

@register("vqa")
def _transform_vqa(
    *,
    qa_dataset,
    corpus_dataset,
    corpus_index,
    keep_unanswerable: bool = False,
    mode: str = "evidence_pages",
    batched: bool = False,
):
    # 1) (optionally) drop unanswerable
    if not keep_unanswerable:
        qa_dataset = filter_dataset(
            qa_dataset,
            field_filters={"answer.type": "answerable"},
            batched=batched,
        )

    # 2) attach page images
    if corpus_index is None:
        raise ValueError("VQA transform requires a CorpusIndex")
    qa_dataset = add_images_column(
        qa_dataset,
        corpus_index,
        mode=mode,
        evidence_pages_path="evidence.pages",
        batched=batched,
    )

    # 3) pick only the fields our model needs
    qa_dataset = project_fields(
        qa_dataset,
        select_fields={
            "question_id": "id",
            "question_text": "question.text",
            "images": "images",
            "answer_variants": "answer.variants",
            "answer_format": "answer.format",
        },
        batched=batched,
    )
    return qa_dataset
