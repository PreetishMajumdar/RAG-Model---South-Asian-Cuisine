"""
Phase B.1.3: Vectorisation and Embedding
Description: Converts chunked text into embeddings using BGE-small 
and builds an HNSW FAISS index for high-speed approximate retrieval.
"""
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

def build_advanced_vector_store(json_filepaths, index_save_path, metadata_save_path):
    print("1. Loading 3 corpus files...")
    corpus_data = []
    
    # Loop through the list of 3 files
    for filepath in json_filepaths:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                corpus_data.extend(json.load(f))
        else:
            print(f"⚠️ Warning: {filepath} not found.")

    texts = [item["text"] for item in corpus_data]
    
    print(f"\n2. Initializing BGE Embedding Model for {len(texts)} chunks...")
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')

    print("3. Vectorizing the text...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    print("\n4. Building the FAISS HNSW Index...")
    dimension = embeddings.shape[1]
    
    index = faiss.IndexHNSWFlat(dimension, 32)
    index.hnsw.efConstruction = 40 
    
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    print("\n5. Saving the Index and Metadata to disk...")
    faiss.write_index(index, index_save_path)
    with open(metadata_save_path, 'w', encoding='utf-8') as f:
        json.dump(corpus_data, f, indent=4, ensure_ascii=False)
        
    print(f"✅ Advanced Vector Store Ready! Saved to {index_save_path}")

if __name__ == "__main__":
    # Pass the three separate files here
    INPUT_FILES = [
        "wikipedia_corpus.json",
        "wikibooks_corpus.json",
        "blog_corpus.json"
    ]
    build_advanced_vector_store(INPUT_FILES, "recipe_faiss_index.bin", "recipe_metadata.json")