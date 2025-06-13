from datasets import Dataset


def push_to_hub(
    dataset: Dataset,
    repo_id: str,
    token: str = None,
    split_name: str = "train",
    **kwargs,
) -> None:
    dataset.push_to_hub(repo_id, split=split_name, **kwargs)
