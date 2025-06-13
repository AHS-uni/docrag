from .vqa import _transform_vqa
from .retrieval import _transform_retrieval

from .registry import get_task_transform

__all__ = [
    "get_task_transform",
    "_transform_vqa",
    "_transform_retrieval",
]
