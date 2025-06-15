"""DocRAG demo – per-document FAISS index manager.

We keep one FAISS IndexFlatIP (inner-product) per document. Embeddings are L2-normalized
so inner‐product ≃ cosine similarity.
"""

from typing import Dict

import faiss
import numpy as np

# In-process store: doc_id → FAISS IndexFlatIP
_INDEXES: Dict[str, faiss.IndexFlatIP] = {}


def create_index(doc_id: str, embeddings: np.ndarray) -> None:
    """Build and cache a FAISS index for a single document.

    Args:
        doc_id: Unique identifier for the document.
        embeddings: 2D numpy array of shape (num_pages, dim), dtype float32.

    Returns:
        None
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    _INDEXES[doc_id] = index


def get_index(doc_id: str) -> faiss.IndexFlatIP:
    """Retrieve the cached FAISS index for a document.

    Args:
        doc_id: Unique document identifier.

    Returns:
        FAISS IndexFlatIP instance.

    Raises:
        KeyError: If no index exists for `doc_id`.
    """
    try:
        return _INDEXES[doc_id]
    except KeyError:
        raise KeyError(f"No FAISS index found for doc_id={doc_id!r}")


def delete_index(doc_id: str) -> None:
    """Remove a document’s index from cache.

    Args:
        doc_id: Unique document identifier.

    Returns:
        None
    """
    _INDEXES.pop(doc_id, None)
