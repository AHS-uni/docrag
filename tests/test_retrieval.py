"""
Tests for the retrieval module.
"""

import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import torch
from datasets import Dataset

from docrag.retrieval import Retriever, FAISSIndexManager


class TestRetrieval(unittest.TestCase):
    """Test cases for the retrieval module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for indices
        self.temp_dir = tempfile.TemporaryDirectory()
        self.index_dir = Path(self.temp_dir.name)
        
        # Sample dataset for testing
        self.sample_data = {
            "doc_id": ["doc1", "doc1", "doc2"],
            "page_number": [1, 2, 1],
            "text": [
                "This is page 1 of document 1 about machine learning.",
                "This is page 2 of document 1 about neural networks.",
                "This is page 1 of document 2 about computer vision."
            ],
            "image_path": [None, None, None]
        }
        self.dataset = Dataset.from_dict(self.sample_data)

    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()

    @patch("docrag.retrieval.retriever.SentenceTransformer")
    @patch("docrag.retrieval.retriever.AutoProcessor")
    @patch("docrag.retrieval.retriever.AutoModel")
    def test_retriever_initialization(self, mock_auto_model, mock_auto_processor, mock_sentence_transformer):
        """Test that the retriever initializes correctly."""
        # Mock the embedding models
        mock_text_model = MagicMock()
        mock_text_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_text_model
        
        mock_image_model = MagicMock()
        mock_image_model.config.projection_dim = 512
        mock_auto_model.from_pretrained.return_value = mock_image_model
        
        # Initialize the retriever
        retriever = Retriever(
            embedding_model_name="mock-text-model",
            image_embedding_model_name="mock-image-model",
            index_dir=self.index_dir,
            top_k=3,
            device="cpu"
        )
        
        # Check that models were initialized correctly
        mock_sentence_transformer.assert_called_once_with("mock-text-model", device="cpu")
        mock_auto_model.from_pretrained.assert_called_once_with("mock-image-model")
        
        # Check that index managers were created
        self.assertEqual(retriever.text_index_manager.dim, 384)
        self.assertEqual(retriever.image_index_manager.dim, 512)
        self.assertEqual(retriever.top_k, 3)

    @patch("docrag.retrieval.retriever.SentenceTransformer")
    @patch("docrag.retrieval.retriever.AutoProcessor")
    @patch("docrag.retrieval.retriever.AutoModel")
    def test_build_index(self, mock_auto_model, mock_auto_processor, mock_sentence_transformer):
        """Test building indices from a dataset."""
        # Mock the embedding models
        mock_text_model = MagicMock()
        mock_text_model.get_sentence_embedding_dimension.return_value = 384
        mock_text_model.encode.return_value = np.random.random((3, 384))
        mock_sentence_transformer.return_value = mock_text_model
        
        mock_image_model = MagicMock()
        mock_image_model.config.projection_dim = 512
        mock_auto_model.from_pretrained.return_value = mock_image_model
        
        # Initialize the retriever
        retriever = Retriever(
            embedding_model_name="mock-text-model",
            image_embedding_model_name="mock-image-model",
            index_dir=self.index_dir,
            top_k=3,
            device="cpu"
        )
        
        # Mock the index building methods
        retriever.text_index_manager.build_index_for_doc = MagicMock()
        
        # Build the index
        retriever.build_index(corpus_dataset=self.dataset)
        
        # Check that the text model was called to encode the texts
        mock_text_model.encode.assert_called_once()
        
        # Check that the index building method was called
        retriever.text_index_manager.build_index_for_doc.assert_called()
        
        # Check that metadata was stored
        self.assertIn("doc1", retriever.doc_metadata)
        self.assertIn("doc2", retriever.doc_metadata)
        self.assertIn(1, retriever.doc_metadata["doc1"])
        self.assertIn(2, retriever.doc_metadata["doc1"])
        self.assertIn(1, retriever.doc_metadata["doc2"])

    @patch("docrag.retrieval.retriever.SentenceTransformer")
    @patch("docrag.retrieval.retriever.AutoProcessor")
    @patch("docrag.retrieval.retriever.AutoModel")
    def test_retrieve(self, mock_auto_model, mock_auto_processor, mock_sentence_transformer):
        """Test retrieving documents with a text query."""
        # Mock the embedding models
        mock_text_model = MagicMock()
        mock_text_model.get_sentence_embedding_dimension.return_value = 384
        mock_text_model.encode.return_value = np.random.random((1, 384))
        mock_sentence_transformer.return_value = mock_text_model
        
        mock_image_model = MagicMock()
        mock_image_model.config.projection_dim = 512
        mock_auto_model.from_pretrained.return_value = mock_image_model
        
        # Initialize the retriever
        retriever = Retriever(
            embedding_model_name="mock-text-model",
            image_embedding_model_name="mock-image-model",
            index_dir=self.index_dir,
            top_k=3,
            device="cpu"
        )
        
        # Mock the search method
        retriever._search_doc = MagicMock(return_value=[(1, 0.9), (2, 0.8)])
        
        # Mock the index map
        retriever.text_index_manager._index_map = {"doc1": "doc1.index"}
        
        # Store some metadata
        retriever.doc_metadata = {
            "doc1": {
                1: {"text": "Page 1 content", "page_number": 1},
                2: {"text": "Page 2 content", "page_number": 2}
            }
        }
        
        # Perform retrieval
        results = retriever.retrieve(query_text="neural networks")
        
        # Check that the text model was called to encode the query
        mock_text_model.encode.assert_called_once_with(["neural networks"], convert_to_numpy=True)
        
        # Check that the search method was called
        retriever._search_doc.assert_called_once()
        
        # Check the results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["doc_id"], "doc1")
        self.assertEqual(results[0]["page_number"], 1)
        self.assertEqual(results[0]["score"], 0.9)
        self.assertEqual(results[1]["doc_id"], "doc1")
        self.assertEqual(results[1]["page_number"], 2)
        self.assertEqual(results[1]["score"], 0.8)


class TestFAISSIndexManager(unittest.TestCase):
    """Test cases for the FAISSIndexManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for indices
        self.temp_dir = tempfile.TemporaryDirectory()
        self.index_dir = Path(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test that the index manager initializes correctly."""
        manager = FAISSIndexManager(
            index_dir=self.index_dir,
            dim=384,
            index_key="Flat"
        )
        
        self.assertEqual(manager.dim, 384)
        self.assertEqual(manager.index_key, "Flat")
        self.assertEqual(manager.mode, "dpr")
        self.assertFalse(manager.use_gpu)
        
        # Check that the index directory was created
        self.assertTrue(self.index_dir.exists())
        
        # Check that the map file was created
        self.assertTrue((self.index_dir / "index_map.json").exists())


if __name__ == "__main__":
    unittest.main() 
