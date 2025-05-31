from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, CommitInfo


def push_dataset_to_hub(
    dataset: Dataset | DatasetDict,
    repo_id: str,
    token: str | None = None,
    **kwargs,
) -> CommitInfo:
    """
    Pushes a Hugging Face Dataset or DatasetDict to the Hugging Face Hub.

    Args:
        dataset: Dataset or DatasetDict to push.
        repo_id: The repository ID (e.g., "username/dataset_name").
        token: Optional HF access token.
    """
    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )

    return dataset.push_to_hub(
        repo_id=repo_id,
        token=token,
        **kwargs,
    )
