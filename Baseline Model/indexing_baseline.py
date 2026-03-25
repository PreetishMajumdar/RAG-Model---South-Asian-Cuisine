"""
Phase B.1.3: Vectorisation and Embedding (BASELINE)
Responsibility: Yash
Description: Converts flat 800-char chunks into embeddings using MiniLM 
and builds a standard Flat Inner-Product FAISS index.
"""
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

def build_baseline_vector_store(json_filepaths, index_save_path, metadata_save_path):
    print("1. Loading baseline corpus files...")
    all_corpus_data = []
    
    for filepath in json_filepaths:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_corpus_data.extend(data)
            print(f"   - Loaded {len(data)} chunks from {filepath}")
        else:
            print(f"   - ⚠️ Warning: {filepath} not found.")

    texts = [item["text"] for item in all_corpus_data]

    print("\n2. Initializing the Baseline Embedding Model (MiniLM)...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    print("3. Vectorizing the text (this might take a minute)...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    print("\n4. Building the Baseline FAISS Index (Flat IP)...")
    dimension = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print("\n5. Saving the Index and Metadata to disk...")
    faiss.write_index(index, index_save_path)
    with open(metadata_save_path, 'w', encoding='utf-8') as f:
        json.dump(all_corpus_data, f, indent=4, ensure_ascii=False)

    print(f"✅ Baseline Vector Store saved to: {index_save_path}")

if __name__ == "__main__":
    # Pointing to the baseline outputs from ingestion_baseline.py
    INPUT_JSONS = [
        "wikipedia_corpus_baseline.json",  
        "wikibooks_corpus_baseline.json",        
        "blog_corpus_baseline.json"        
    ]
    build_baseline_vector_store(INPUT_JSONS, "baseline_faiss_index.bin", "baseline_metadata.json")