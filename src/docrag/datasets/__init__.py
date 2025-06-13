"""
The `datasets` package
"""

from .hf import (
    load_corpus_dataset,
    load_qa_dataset,
)
from .index import CorpusIndex
from .transform import task_transform


__all__ = [
    "load_corpus_dataset",
    "load_qa_dataset",
    "CorpusIndex",
    "task_transform"
]
