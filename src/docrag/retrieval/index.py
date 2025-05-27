import json
from pathlib import Path

import faiss
import numpy as np


class FAISSIndexManager:
    """
    Manages one FAISS index per document.
    Supports both 'dpr' (1-vector/page) and 'li' (multi-vector/page).
    """

    def __init__(
        self,
        index_dir: Path,
        dim: int,
        index_key: str = "Flat",
        metric: int = faiss.METRIC_INNER_PRODUCT,
        use_gpu: bool = False,
        mode: str = "dpr",  # or "li"
    ):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.dim = dim
        self.index_key = index_key
        self.metric = metric
        self.use_gpu = use_gpu
        self.mode = mode  # "dpr" or "li"

        # in-memory maps
        self._indexes: dict[str, faiss.Index] = {}  # doc_id → index
        self._meta: dict[str, list[tuple[int, Path]]] = (
            {}
        )  # doc_id → list of (page_num, image_path) or (page_num, vec_id)

        # persistent map
        self.map_file = self.index_dir / "index_map.json"
        if self.map_file.exists():
            self._index_map = json.loads(self.map_file.read_text())
        else:
            self._index_map: dict[str, str] = {}

    def _make_index(self) -> faiss.Index:
        idx = faiss.index_factory(self.dim, self.index_key, self.metric)
        # if self.use_gpu:
        #     res = faiss.StandardGpuResources()
        #     idx = faiss.index_cpu_to_gpu(res, 0, idx)
        return idx

    def build_index_for_doc(
        self,
        doc_id: str,
        embeddings: np.ndarray,
        metadata: list[tuple[int, Path]],
        train: bool = False,
    ) -> None:
        """
        embeddings: shape (N, dim) where
            N = #pages (mode="dpr") or sum(#vectors per page) (mode="li")
        metadata:  list of tuples in same order, either
            [(page_num, image_path), ...]  for DPR, or
            [(page_num, vector_idx), ...]   for LI
        """
        idx = self._make_index()

        # IVF indexes must be trained first, now without explicit 'n'
        if train and hasattr(idx, "is_trained") and not idx.is_trained:
            idx.train(embeddings)

        if self.metric == faiss.METRIC_INNER_PRODUCT:
            faiss.normalize_L2(embeddings)

        # add vectors without explicit count
        idx.add(embeddings)

        # store in-memory
        self._indexes[doc_id] = idx
        self._meta[doc_id] = metadata

        # persist index
        filename = f"{doc_id}.index"
        faiss.write_index(idx, str(self.index_dir / filename))
        self._index_map[doc_id] = filename
        self._save_map()

    def load_index_for_doc(self, doc_id: str) -> None:
        """
        Loads a saved index from disk into memory for a given doc.
        """
        if doc_id in self._indexes:
            return  # already loaded

        filename = self._index_map.get(doc_id)
        if not filename:
            raise KeyError(f"No index on disk for doc_id={doc_id}")

        idx = faiss.read_index(str(self.index_dir / filename))
        # if self.use_gpu:
        #     res = faiss.StandardGpuResources()
        #     idx = faiss.index_cpu_to_gpu(res, 0, idx)

        self._indexes[doc_id] = idx
        # Note: you must also reload metadata yourself (e.g. from a sidecar JSON)

    def search(
        self, doc_id: str, query_emb: np.ndarray, top_k: int, nprobe: int | None = None
    ) -> list[tuple[int, float]]:
        """
        Returns list of (page_num, score) for DPR,
        or aggregated (page_num, aggregated_score) for LI.
        """
        # ensure index is loaded
        if doc_id not in self._indexes:
            self.load_index_for_doc(doc_id)

        idx = self._indexes[doc_id]
        meta = self._meta[doc_id]

        if self.metric == faiss.METRIC_INNER_PRODUCT:
            faiss.normalize_L2(query_emb)

        if nprobe is not None and hasattr(idx, "nprobe"):
            idx.nprobe = nprobe

        # perform search without explicit 'n' or pre-allocated outputs
        distances, labels = idx.search(query_emb, top_k)

        if self.mode == "dpr":
            # one-to-one mapping
            return [
                (int(meta[labels[0, r]][0]), float(distances[0, r]))
                for r in range(top_k)
            ]
        else:  # mode == "li"
            # multi-vector query → aggregate per page_num
            hits: dict[int, list[float]] = {}
            nq = query_emb.shape[0]
            for qi in range(nq):
                for r in range(top_k):
                    vec_idx = int(labels[qi, r])
                    page_num = meta[vec_idx][0]
                    score = float(distances[qi, r])
                    hits.setdefault(page_num, []).append(score)
            return [(pg, max(scores)) for pg, scores in hits.items()]

    def _save_map(self):
        self.map_file.write_text(json.dumps(self._index_map, indent=2))
