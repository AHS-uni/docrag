from .registry import register

from ..core import filter_dataset, project_fields, add_images_column

__all__ = [
    "_transform_retrieval",
]


@register("retrieval")
def _transform_retrieval(
    dataset,
    *,
    explode_pages: bool = True,
    batched: bool = False,
    **_,
):
    # 1) optionally filter to only answerable
    dataset = filter_dataset(
        dataset,
        field_filters={"answer.type": "answerable"},
        batched=batched,
    )

    # 2) pull out just question + evidence pages + doc id
    dataset = project_fields(
        dataset,
        select_fields={
            "query": "question.text",
            "doc_id": "document.id",
            "pages": "evidence.pages",
        },
        batched=batched,
    )

    # 3) explode one example per (query, doc_id, page)
    if explode_pages:

        def _explode(ex: dict):
            return [
                {"query": ex["query"], "doc_id": ex["doc_id"], "page": p}
                for p in ex["pages"]
            ]

        dataset = dataset.map(_explode, batched=batched)

    return dataset
