#!/usr/bin/env python
"""
Example script demonstrating how to use the DocRAG retrieval module.
"""

import argparse
from pathlib import Path
from typing import Dict, Any

import torch
from datasets import load_dataset, Dataset
from PIL import Image

from docrag.retrieval import Retriever


def parse_args():
    parser = argparse.ArgumentParser(description="DocRAG Retrieval Example")
    parser.add_argument(
        "--dataset_path", 
        type=str,
        default="docvqa",
        help="HuggingFace dataset path or local dataset directory"
    )
    parser.add_argument(
        "--index_dir", 
        type=str, 
        default="./indices",
        help="Directory to store FAISS indices"
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=5,
        help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--text_model", 
        type=str, 
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Text embedding model name"
    )
    parser.add_argument(
        "--image_model", 
        type=str, 
        default="openai/clip-vit-base-patch32",
        help="Image embedding model name"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        default="dpr",
        choices=["dpr", "li"],
        help="Retrieval mode: 'dpr' (one vector per page) or 'li' (multiple vectors per page)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for embedding models"
    )
    parser.add_argument(
        "--query_text", 
        type=str, 
        default=None,
        help="Text query for retrieval"
    )
    parser.add_argument(
        "--query_image", 
        type=str, 
        default=None,
        help="Path to image query for retrieval"
    )
    
    return parser.parse_args()


def prepare_sample_dataset() -> Dataset:
    """
    Create a sample dataset for demonstration purposes.
    In a real scenario, you would load your actual document corpus.
    """
    # Create a simple sample dataset with text and image paths
    data = {
        "doc_id": ["doc1", "doc1", "doc1", "doc2", "doc2"],
        "page_number": [1, 2, 3, 1, 2],
        "text": [
            "This is the first page of document 1. It contains information about machine learning.",
            "This is the second page of document 1. It discusses neural networks and deep learning.",
            "This is the third page of document 1. It covers transformers and attention mechanisms.",
            "This is the first page of document 2. It contains information about computer vision.",
            "This is the second page of document 2. It discusses image processing techniques."
        ],
        "image_path": [
            None,  # No image for this page
            None,  # No image for this page
            None,  # No image for this page
            None,  # No image for this page
            None,  # No image for this page
        ]
    }
    
    return Dataset.from_dict(data)


def main():
    args = parse_args()
    
    print(f"Initializing retriever with {args.text_model} and {args.image_model}...")
    retriever = Retriever(
        embedding_model_name=args.text_model,
        image_embedding_model_name=args.image_model,
        index_dir=Path(args.index_dir),
        top_k=args.top_k,
        mode=args.mode,
        device=args.device
    )
    
    # Load or prepare dataset
    print("Preparing sample dataset...")
    try:
        dataset = load_dataset(args.dataset_path)
        if isinstance(dataset, dict):
            # Take the first split if multiple are available
            dataset = next(iter(dataset.values()))
    except Exception:
        print("Could not load dataset from HuggingFace, using sample dataset instead.")
        dataset = prepare_sample_dataset()
    
    # Build index
    print("Building retrieval indices...")
    retriever.build_index(corpus_dataset=dataset)
    print("Index built successfully!")
    
    # Perform retrieval
    if args.query_text:
        print(f"Performing text retrieval with query: '{args.query_text}'")
        results = retriever.retrieve(query_text=args.query_text)
        print_results(results)
    
    if args.query_image:
        try:
            query_image = Image.open(args.query_image)
            print(f"Performing image retrieval with image: {args.query_image}")
            results = retriever.retrieve(query_image=query_image, modality="image")
            print_results(results)
        except Exception as e:
            print(f"Error loading query image: {e}")
    
    # If no query provided, use a default one
    if not args.query_text and not args.query_image:
        default_query = "What is deep learning?"
        print(f"No query provided. Using default text query: '{default_query}'")
        results = retriever.retrieve(query_text=default_query)
        print_results(results)


def print_results(results: list[Dict[str, Any]]):
    """Print retrieval results in a readable format."""
    print("\nRetrieval Results:")
    print("=" * 60)
    
    if not results:
        print("No results found.")
        return
    
    for i, result in enumerate(results, 1):
        print(f"Result #{i}:")
        print(f"  Document ID: {result['doc_id']}")
        print(f"  Page Number: {result['page_number']}")
        print(f"  Score: {result['score']:.4f}")
        
        # Print text content if available
        if result.get("metadata", {}).get("text"):
            text = result["metadata"]["text"]
            # Truncate long text for display
            if len(text) > 100:
                text = text[:97] + "..."
            print(f"  Text: {text}")
        
        # Print image path if available
        if result.get("metadata", {}).get("image_path"):
            print(f"  Image: {result['metadata']['image_path']}")
        
        print("-" * 60)


if __name__ == "__main__":
    main() 
