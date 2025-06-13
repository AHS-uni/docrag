"""
Implements the Retriever class for document retrieval in the RAG pipeline.
"""

from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModel

from docrag.retrieval.base import BaseRetriever
from docrag.retrieval.index import FAISSIndexManager


class Retriever(BaseRetriever):
    """
    Retriever implementation for document VQA using RAG.
    Supports both text-based and image-based retrieval.
    """

    def __init__(
        self,
        *,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        image_embedding_model_name: str = "openai/clip-vit-base-patch32",
        index_dir: Path = Path("./indices"),
        top_k: int = 5,
        sim_threshold: float = 0.7,
        mode: str = "dpr",  # "dpr" or "li"
        use_gpu: bool = torch.cuda.is_available(),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """
        Initialize the retriever with text and image embedding models.

        Args:
            embedding_model_name: Model name for text embeddings
            image_embedding_model_name: Model name for image embeddings
            index_dir: Directory to store FAISS indices
            top_k: Number of documents to retrieve
            sim_threshold: Minimum similarity threshold
            mode: "dpr" (one vector per page) or "li" (multiple vectors per page)
            use_gpu: Whether to use GPU for FAISS
            device: Device for embedding models
        """
        super().__init__(top_k=top_k, sim_threshold=sim_threshold)
        
        self.embedding_model_name = embedding_model_name
        self.image_embedding_model_name = image_embedding_model_name
        self.index_dir = Path(index_dir)
        self.mode = mode
        self.use_gpu = use_gpu
        self.device = device
        
        # Initialize text embedding model
        self.text_model = SentenceTransformer(embedding_model_name, device=device)
        self.text_embedding_dim = self.text_model.get_sentence_embedding_dimension()
        
        # Initialize image embedding model
        self.image_processor = AutoProcessor.from_pretrained(image_embedding_model_name)
        self.image_model = AutoModel.from_pretrained(image_embedding_model_name).to(device)
        self.image_embedding_dim = self.image_model.config.projection_dim
        
        # Initialize FAISS index managers for text and image
        self.text_index_manager = FAISSIndexManager(
            index_dir=self.index_dir / "text",
            dim=self.text_embedding_dim,
            use_gpu=use_gpu,
            mode=mode
        )
        
        self.image_index_manager = FAISSIndexManager(
            index_dir=self.index_dir / "image",
            dim=self.image_embedding_dim,
            use_gpu=use_gpu,
            mode=mode
        )
        
        # Document metadata storage
        self.doc_metadata: Dict[str, Dict[int, Dict[str, Any]]] = {}

    def build_index(
        self, 
        *, 
        corpus_dataset: Dataset, 
        fields: Dict[str, str] = {
            "doc_id": "doc_id",
            "page_number": "page_number",
            "image": "image_path",
            "text": "text"
        }
    ) -> None:
        """
        Build text and image indices from a corpus dataset.
        
        Args:
            corpus_dataset: HF Dataset with document corpus
            fields: Mapping of expected fields to actual field names in dataset
        """
        doc_id_field = fields.get("doc_id", "doc_id")
        page_field = fields.get("page_number", "page_number")
        image_field = fields.get("image", "image_path")
        text_field = fields.get("text", "text")
        
        # Group by document ID
        doc_groups = {}
        for item in corpus_dataset:
            doc_id = item[doc_id_field]
            if doc_id not in doc_groups:
                doc_groups[doc_id] = []
            doc_groups[doc_id].append(item)
        
        # Process each document
        for doc_id, items in doc_groups.items():
            # Initialize document metadata
            self.doc_metadata[doc_id] = {}
            
            # Prepare text embeddings
            texts = []
            text_metadata = []
            
            # Prepare image embeddings
            images = []
            image_metadata = []
            
            for item in items:
                page_num = item[page_field]
                
                # Store metadata for this page
                self.doc_metadata[doc_id][page_num] = {
                    "page_number": page_num,
                    "image_path": item[image_field] if image_field in item else None,
                    "text": item[text_field] if text_field in item else None
                }
                
                # Process text if available
                if text_field in item and item[text_field]:
                    texts.append(item[text_field])
                    text_metadata.append((page_num, None))
                
                # Process image if available
                if image_field in item and item[image_field]:
                    # Load image if it's a path string
                    if isinstance(item[image_field], str):
                        try:
                            img = Image.open(item[image_field])
                            images.append(img)
                        except Exception as e:
                            print(f"Error loading image {item[image_field]}: {e}")
                            continue
                    else:
                        # Assume it's already a PIL Image
                        images.append(item[image_field])
                    
                    image_metadata.append((page_num, None))
            
            # Generate text embeddings
            if texts:
                text_embeddings = self.text_model.encode(
                    texts, 
                    show_progress_bar=True, 
                    convert_to_numpy=True
                )
                
                # Build text index
                self.text_index_manager.build_index_for_doc(
                    doc_id=doc_id,
                    embeddings=text_embeddings,
                    metadata=text_metadata
                )
            
            # Generate image embeddings
            if images:
                image_embeddings = self._encode_images(images)
                
                # Build image index
                self.image_index_manager.build_index_for_doc(
                    doc_id=doc_id,
                    embeddings=image_embeddings,
                    metadata=image_metadata
                )
    
    def _encode_images(self, images: List[Image.Image]) -> np.ndarray:
        """
        Encode images using the image embedding model.
        
        Args:
            images: List of PIL images
            
        Returns:
            np.ndarray: Image embeddings
        """
        embeddings = []
        batch_size = 8  # Process images in batches to avoid OOM
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            
            with torch.no_grad():
                inputs = self.image_processor(images=batch, return_tensors="pt").to(self.device)
                outputs = self.image_model(**inputs)
                
                # Get image embeddings from the model
                if hasattr(outputs, "image_embeds"):
                    batch_embeddings = outputs.image_embeds.cpu().numpy()
                else:
                    # For CLIP-like models
                    batch_embeddings = outputs.image_embeds.cpu().numpy()
                
                embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def retrieve(
        self, 
        query_text: str = None,
        query_image: Image.Image = None,
        doc_id: str = None,
        modality: str = "text",
        top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents based on text or image query.
        
        Args:
            query_text: Text query
            query_image: Image query
            doc_id: Optional document ID to restrict search
            modality: "text", "image", or "both"
            top_k: Number of results to return (overrides instance setting)
            
        Returns:
            List of retrieved documents with metadata and scores
        """
        if top_k is None:
            top_k = self.top_k
            
        results = []
        
        # Text-based retrieval
        if (modality == "text" or modality == "both") and query_text:
            # Encode query text
            query_embedding = self.text_model.encode([query_text], convert_to_numpy=True)
            
            # If doc_id is provided, search only that document
            if doc_id:
                text_results = self._search_doc(
                    doc_id=doc_id,
                    query_embedding=query_embedding,
                    top_k=top_k,
                    index_manager=self.text_index_manager
                )
                
                for page_num, score in text_results:
                    if score >= self.sim_threshold:
                        results.append({
                            "doc_id": doc_id,
                            "page_number": page_num,
                            "score": float(score),
                            "metadata": self.doc_metadata.get(doc_id, {}).get(page_num, {})
                        })
            else:
                # Search all documents
                for doc_id in self.text_index_manager._index_map.keys():
                    text_results = self._search_doc(
                        doc_id=doc_id,
                        query_embedding=query_embedding,
                        top_k=top_k,
                        index_manager=self.text_index_manager
                    )
                    
                    for page_num, score in text_results:
                        if score >= self.sim_threshold:
                            results.append({
                                "doc_id": doc_id,
                                "page_number": page_num,
                                "score": float(score),
                                "metadata": self.doc_metadata.get(doc_id, {}).get(page_num, {})
                            })
        
        # Image-based retrieval
        if (modality == "image" or modality == "both") and query_image:
            # Encode query image
            with torch.no_grad():
                inputs = self.image_processor(images=[query_image], return_tensors="pt").to(self.device)
                outputs = self.image_model(**inputs)
                query_embedding = outputs.image_embeds.cpu().numpy()
            
            # If doc_id is provided, search only that document
            if doc_id:
                image_results = self._search_doc(
                    doc_id=doc_id,
                    query_embedding=query_embedding,
                    top_k=top_k,
                    index_manager=self.image_index_manager
                )
                
                for page_num, score in image_results:
                    if score >= self.sim_threshold:
                        results.append({
                            "doc_id": doc_id,
                            "page_number": page_num,
                            "score": float(score),
                            "metadata": self.doc_metadata.get(doc_id, {}).get(page_num, {})
                        })
            else:
                # Search all documents
                for doc_id in self.image_index_manager._index_map.keys():
                    image_results = self._search_doc(
                        doc_id=doc_id,
                        query_embedding=query_embedding,
                        top_k=top_k,
                        index_manager=self.image_index_manager
                    )
                    
                    for page_num, score in image_results:
                        if score >= self.sim_threshold:
                            results.append({
                                "doc_id": doc_id,
                                "page_number": page_num,
                                "score": float(score),
                                "metadata": self.doc_metadata.get(doc_id, {}).get(page_num, {})
                            })
        
        # Sort by score and limit to top_k
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
        return results
    
    def _search_doc(
        self,
        doc_id: str,
        query_embedding: np.ndarray,
        top_k: int,
        index_manager: FAISSIndexManager
    ) -> List[Tuple[int, float]]:
        """
        Search a specific document using the provided index manager.
        
        Args:
            doc_id: Document ID
            query_embedding: Query embedding
            top_k: Number of results to return
            index_manager: FAISS index manager to use
            
        Returns:
            List of (page_num, score) tuples
        """
        try:
            return index_manager.search(
                doc_id=doc_id,
                query_emb=query_embedding,
                top_k=top_k
            )
        except KeyError:
            # Index not found for this document
            return []
        except Exception as e:
            print(f"Error searching doc {doc_id}: {e}")
            return []
    
    def get_document_page(self, doc_id: str, page_number: int) -> Dict[str, Any]:
        """
        Get metadata for a specific document page.
        
        Args:
            doc_id: Document ID
            page_number: Page number
            
        Returns:
            Dict with page metadata
        """
        return self.doc_metadata.get(doc_id, {}).get(page_number, {})
