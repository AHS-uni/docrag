#!/usr/bin/env python
"""
Example script demonstrating a complete RAG pipeline integrating retrieval with generation.
"""

import argparse
from pathlib import Path
from typing import Dict, Any, List

import torch
from datasets import load_dataset, Dataset
from PIL import Image

from docrag.retrieval import Retriever
from docrag.generation import Generator
from docrag.schema.config import GeneratorConfig


def parse_args():
    parser = argparse.ArgumentParser(description="DocRAG Complete Pipeline Example")
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
        default=3,
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
        "--generator_model", 
        type=str, 
        default="qwen2.5-vl",
        help="Generator model name"
    )
    parser.add_argument(
        "--generator_path", 
        type=str, 
        default="Qwen/Qwen2.5-VL-7B",
        help="Path to generator model"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for models"
    )
    parser.add_argument(
        "--query", 
        type=str, 
        default="What is deep learning?",
        help="Query for retrieval and generation"
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
            "This is the first page of document 1. It contains information about machine learning, which is a subset of artificial intelligence.",
            "This is the second page of document 1. It discusses neural networks and deep learning. Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
            "This is the third page of document 1. It covers transformers and attention mechanisms, which are key components of modern deep learning architectures.",
            "This is the first page of document 2. It contains information about computer vision, which is a field of AI that enables computers to derive meaningful information from digital images and videos.",
            "This is the second page of document 2. It discusses image processing techniques and convolutional neural networks (CNNs), which are specialized neural networks for processing visual data."
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


def create_generator_config(args) -> GeneratorConfig:
    """
    Create a generator configuration.
    """
    return GeneratorConfig.model_validate({
        "model": {
            "name": args.generator_model,
            "path": args.generator_path,
            "device": args.device,
            "dtype": "float16",
            "device_map": "auto",
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "attn_implementation": "eager"
        },
        "tokenizer": {},
        "image_processor": {},
        "generation": {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9
        },
        "system_prompt": "You are a helpful assistant that answers questions based on the provided document context.",
        "prompt_template": "Based on the retrieved document pages, please answer the following question: {text}"
    })


def format_context_from_results(results: List[Dict[str, Any]]) -> str:
    """
    Format retrieval results into a context string for the generator.
    """
    context_parts = []
    
    for i, result in enumerate(results, 1):
        doc_id = result["doc_id"]
        page_num = result["page_number"]
        text = result.get("metadata", {}).get("text", "")
        
        if text:
            context_parts.append(f"Document {doc_id}, Page {page_num}:\n{text}")
    
    return "\n\n".join(context_parts)


def main():
    args = parse_args()
    
    print("Initializing retriever...")
    retriever = Retriever(
        embedding_model_name=args.text_model,
        image_embedding_model_name=args.image_model,
        index_dir=Path(args.index_dir),
        top_k=args.top_k,
        device=args.device
    )
    
    print("Initializing generator...")
    generator_config = create_generator_config(args)
    generator = Generator(config=generator_config)
    generator.to(args.device)
    
    # Load or prepare dataset
    print("Preparing dataset...")
    try:
        dataset = load_dataset(args.dataset_path)
        if isinstance(dataset, dict):
            # Take the first split if multiple are available
            dataset = next(iter(dataset.values()))
    except Exception:
        print("Could not load dataset from HuggingFace, using sample dataset instead.")
        dataset = prepare_sample_dataset()
    
    # Build retrieval index
    print("Building retrieval indices...")
    retriever.build_index(corpus_dataset=dataset)
    print("Index built successfully!")
    
    # Process the query
    query = args.query
    print(f"\nQuery: {query}")
    
    # Step 1: Retrieve relevant documents
    print("\nStep 1: Retrieving relevant documents...")
    retrieval_results = retriever.retrieve(query_text=query, top_k=args.top_k)
    
    if not retrieval_results:
        print("No relevant documents found.")
        return
    
    # Print retrieval results
    print("\nRetrieval Results:")
    print("=" * 60)
    for i, result in enumerate(retrieval_results, 1):
        print(f"Result #{i}:")
        print(f"  Document ID: {result['doc_id']}")
        print(f"  Page Number: {result['page_number']}")
        print(f"  Score: {result['score']:.4f}")
        
        if result.get("metadata", {}).get("text"):
            text = result["metadata"]["text"]
            # Truncate long text for display
            if len(text) > 100:
                text = text[:97] + "..."
            print(f"  Text: {text}")
        
        print("-" * 60)
    
    # Step 2: Format context from retrieval results
    context = format_context_from_results(retrieval_results)
    
    # Step 3: Generate answer using the retrieved context
    print("\nStep 3: Generating answer based on retrieved documents...")
    
    # Prepare images if available (none in this example)
    images = []
    
    # Generate answer
    # Combine context and query
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    answer = generator.generate(images=images, text=prompt)
    
    print("\nGenerated Answer:")
    print("=" * 60)
    print(answer)
    print("=" * 60)


if __name__ == "__main__":
    main() 
