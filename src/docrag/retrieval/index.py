import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import Any
import json


class FAISSIndexManager:
    def __init__(
        self,
        index_dir: Path,
        dim: int,
        index_key: str = "Flat",
        metric: int = faiss.METRIC_INNER_PRODUCT,
        use_gpu: bool = False,
        mode: str = "dpr",  # or "li"
        nprobe: int = 10,
        ef_search: int = 64
    ) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.dim = dim
        self.index_key = index_key
        self.metric = metric
        self.use_gpu = use_gpu
        self.mode = mode

        self.doc_indexes: dict[str, faiss.Index] = {}
        self.doc_metadata: dict[str, list[tuple[int, Path]]] = {}
        self.nprobe = nprobe
        self.ef_search = ef_search
        self.gpu_res = faiss.StandardGpuResources() if use_gpu else None

        # centralized persistent map
        self.map_file = self.index_dir / "index_map.json"
        if self.map_file.exists():
            self._index_map = json.loads(self.map_file.read_text())
        else:
            self._index_map: dict[str, str] = {}


    def _save_map(self):
      self.map_file.write_text(json.dumps(self._index_map, indent=2))


    def _apply_index_tuning(self, idx: faiss.Index) -> None:
     if isinstance(idx, faiss.IndexIVF):
         idx.nprobe = self.nprobe
     if hasattr(idx, 'hnsw'):
         idx.hnsw.efSearch = self.ef_search


    def _make_index(self, index_key: str | None = None) -> faiss.Index:
        key = index_key or self.index_key
        idx = faiss.index_factory(self.dim, key, self.metric)

        # Tune nprobe or efSearch
        self._apply_index_tuning(idx)


        if self.use_gpu:
           idx = faiss.index_cpu_to_gpu(self.gpu_res, 0, idx)
        return idx



    def _save_index_and_metadata(self, doc_id: str, idx: faiss.Index, metadata: list[tuple[int, Path]]) -> None:
     index_path = self.index_dir / f"{doc_id}.index"
     faiss.write_index(faiss.index_gpu_to_cpu(idx) if self.use_gpu else idx, str(index_path))

     with open(self.index_dir / f"{doc_id}_meta.pkl", "wb") as f:
         pickle.dump(metadata, f)


     self._index_map[doc_id] = index_path.name
     self._save_map()



    def build_index_for_doc(
        self,
        doc_id: str,
        embeddings: np.ndarray,
        metadata: list[tuple[int, Path]],
        train: bool = False,
        index_key: str | None = None
    ) -> None:
        if self.metric == faiss.METRIC_INNER_PRODUCT:
            faiss.normalize_L2(embeddings)
        idx = self._make_index(index_key=index_key)

        if train and hasattr(idx, "is_trained") and not idx.is_trained:
            key_used = index_key or self.index_key
            print(f"Training index '{key_used}' for doc '{doc_id}'...")
            idx.train(embeddings)

        idx.add(embeddings)

        self.doc_indexes[doc_id] = idx
        self.doc_metadata[doc_id] = metadata

        self._save_index_and_metadata(doc_id, idx, metadata)


    def load_index(self, doc_id: str) -> None:
        if doc_id in self.doc_indexes:
            return

        filename = self._index_map.get(doc_id)
        if not filename:
            raise FileNotFoundError(f"No index mapped for doc_id: {doc_id}")

        index_path = self.index_dir / filename
        meta_path = self.index_dir / f"{doc_id}_meta.pkl"

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Missing index or metadata for doc_id: {doc_id}")

        idx = faiss.read_index(str(index_path))
        if self.use_gpu:
            idx = faiss.index_cpu_to_gpu(self.gpu_res, 0, idx)

        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        self.doc_indexes[doc_id] = idx
        self.doc_metadata[doc_id] = metadata



    def search(
        self,
        doc_id: str,
        queries: np.ndarray,
        top_k: int = 5,
        return_metadata: bool = False,
        nprobe: int | None = None,
        ef_search: int | None = None,
        auto_nprobe: bool = True,
        auto_efsearch: bool = True,
    ) -> Any:
        if doc_id not in self.doc_indexes:
            self.load_index(doc_id)

        idx = self.doc_indexes[doc_id]
        metadata = self.doc_metadata[doc_id]

        if self.metric == faiss.METRIC_INNER_PRODUCT:
            faiss.normalize_L2(queries)

         # Tune nprobe for IVF indices
        if hasattr(idx, "nprobe"):
            if nprobe is not None:
                idx.nprobe = nprobe
            elif auto_nprobe and hasattr(idx, "nlist"):
                idx.nprobe = max(1, int(np.sqrt(idx.nlist)))

        # Tune efSearch for HNSW indices
        if hasattr(idx, "hnsw") and hasattr(idx.hnsw, "efSearch"):
            if ef_search is not None:
                idx.hnsw.efSearch = ef_search
            elif auto_efsearch and hasattr(idx.hnsw, "efConstruction"):
                idx.hnsw.efSearch = idx.hnsw.efConstruction * 2

        distances, indices = idx.search(queries, top_k)

        if self.mode == "li":
            hits: dict[int, list[float]] = {}
            nq = queries.shape[0]
            for qi in range(nq):
                for r in range(top_k):
                    vec_idx = int(indices[qi, r])
                    if vec_idx >= len(metadata): continue
                    page_num = metadata[vec_idx][0]
                    score = float(distances[qi, r])
                    hits.setdefault(page_num, []).append(score)
            #return [(pg, max(scores)) for pg, scores in hits.items()]
            return sorted([(pg, max(scores)) for pg, scores in hits.items()], key=lambda x: -x[1])

        elif return_metadata:
            # DPR with metadata
            results = []
            for dist_list, idx_list in zip(distances, indices):
                result = []
                for i, d in zip(idx_list, dist_list):
                    if i < len(metadata):
                        result.append((metadata[i][0], metadata[i][1], float(d)))
                results.append(result)
            return results

        return distances, indices


