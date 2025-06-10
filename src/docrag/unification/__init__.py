"""
Package 'unification'
"""

from .registry import get_unifier
from .mpdocvqa import MPDocVQAUnifier
from .dude import DUDEUnifier
from .mmlongbenchdoc import MMLongBenchDocUnifier
from .arxivqa import ArxivQAUnifier
from .tatdqa import TATDQAUnifier
from .slidevqa import SlideVQAUnifier

__all__ = [
    "get_unifier",
    "MPDocVQAUnifier",
    "DUDEUnifier",
    "MMLongBenchDocUnifier",
    "ArxivQAUnifier",
    "TATDQAUnifier",
    "SlideVQAUnifier",
]
