"""
Package 'unify'
"""

from .mpdocvqa import MPDocVQAUnifier
from .dude import DUDEUnifier
from .mmlongbenchdoc import MMLongBenchDocUnifier
from .arxivqa import ArxivQAUnifier

__all__ = [
    "MPDocVQAUnifier",
    "DUDEUnifier",
    "MMLongBenchDocUnifier",
    "ArxivQAUnifier",
]
