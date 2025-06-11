"""
Package 'unification'
"""

from .registry import get_unifier
from .datasets import (
    MPDocVQAUnifier,
    DUDEUnifier,
    MMLongBenchDocUnifier,
    ArxivQAUnifier,
    TATDQAUnifier,
    SlideVQAUnifier,
)

__all__ = [
    "get_unifier",
    "MPDocVQAUnifier",
    "DUDEUnifier",
    "MMLongBenchDocUnifier",
    "ArxivQAUnifier",
    "TATDQAUnifier",
    "SlideVQAUnifier",
]
