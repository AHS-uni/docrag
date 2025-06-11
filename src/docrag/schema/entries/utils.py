from .unified import Tag
from .enums import TagName

__all__ = [
    "tag_missing",
    "tag_low_quality",
    "tag_inferred",
    "tag_predicted",
]

### Tagging Functions ###

def tag_missing(target: str, comment: str = "") -> Tag:
    """
    Create a MISSING tag for fields defaulted or unavailable in the raw data.

    Args:
        target: The field name this tag applies to.
        comment: Optional human-readable note explaining why it’s missing.
    """
    return Tag(name=TagName.MISSING, target=target, comment=comment)


def tag_low_quality(target: str, comment: str = "") -> Tag:
    """
    Create a LOW_QUALITY tag for fields with unreliable or heuristic values.

    Args:
        target: The field name this tag applies to.
        comment: Optional human-readable note explaining the low quality.
    """
    return Tag(name=TagName.LOW_QUALITY, target=target, comment=comment)


def tag_inferred(target: str, comment: str = "") -> Tag:
    """
    Create an INFERRED tag for values filled in via rule‐based logic.

    Args:
        target: The field name this tag applies to.
        comment: Optional human-readable note explaining the inference.
    """
    return Tag(name=TagName.INFERRED, target=target, comment=comment)


def tag_predicted(target: str, comment: str = "") -> Tag:
    """
    Create a PREDICTED tag for values imputed by a trained model.

    Args:
        target: The field name this tag applies to.
        comment: Optional human-readable note explaining the prediction.
    """
    return Tag(name=TagName.PREDICTED, target=target, comment=comment)
