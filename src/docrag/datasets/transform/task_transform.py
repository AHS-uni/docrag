from .tasks import get_task_transform

def task_transform(
    *,
    qa_dataset,
    corpus_dataset,
    corpus_index,
    task: str,
    batched: bool = False,
    **task_kwargs,
):
    """
    Main entry point: run the pipeline registered under `task`.
    `task_kwargs` are passed through to that task function.
    """
    task_transform = get_task_transform(task)
    return task_transform(
        qa_dataset=qa_dataset,
        corpus_dataset=corpus_dataset,
        corpus_index=corpus_index,
        batched=batched,
        **task_kwargs,
    )
