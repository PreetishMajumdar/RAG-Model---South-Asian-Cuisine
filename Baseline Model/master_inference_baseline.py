"""
Phase B.1.4: Prompting and Generation (BASELINE)
Responsibility: Arunima & Yash
Description: Standard Single-Stage Retrieval (MiniLM) feeding 800-char chunks
directly into Qwen2.5-0.5B-Instruct.
"""
import os
import json
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

class BaselineRAG:
    def __init__(self, index_path, metadata_path, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        print("Loading Baseline Embedding Model (MiniLM)...")
        self.embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        print("Loading Baseline FAISS Index...")
        self.index = faiss.read_index(index_path)

        print("Loading Metadata...")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata_store = json.load(f)
            
        print(f"Loading LLM ({model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        print("✅ Baseline System Ready!")

    def retrieve(self, query, final_k=5):
        # Single-stage retrieval (No Cross-Encoder, No Parent Deduplication)
        query_vector = self.embed_model.encode([query])
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), final_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:
                chunk = self.metadata_store[idx].copy()
                chunk['similarity_score'] = float(dist)
                results.append(chunk)
        return results

    def generate(self, query, retrieved_chunks):
        # Format the context directly from the 800-char chunks
        context_block = ""
        for i, chunk in enumerate(retrieved_chunks, 1):
            topic = chunk['metadata'].get('topic', 'Unknown')
            context_block += f"--- Source {i} ({topic}) ---\n{chunk['text']}\n\n"

        system_instruction = (
            "You are a specialized AI culinary assistant. "
            "Use ONLY the provided context sources to answer the user's question. "
            "If the answer is not in the sources, say you don't know."
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"CONTEXT:\n{context_block}\n\nQUESTION: {query}"}
        ]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=1024, temperature=0.1, do_sample=False)
        response_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        return self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0].strip()

if __name__ == "__main__":
    rag = BaselineRAG("baseline_faiss_index.bin", "baseline_metadata.json")
    
    while True:
        user_query = input("\nAsk a baseline question (or 'quit'): ")
        if user_query.lower() in ['quit', 'exit']: break
        if not user_query.strip(): continue

        chunks = rag.retrieve(user_query)
        answer = rag.generate(user_query, chunks)
        
        print("\n" + "="*50)
        print(f"🤖 Baseline AI:\n{answer}")
        print("="*50)