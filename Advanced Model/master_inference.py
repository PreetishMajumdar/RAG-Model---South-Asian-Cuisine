"""
Phase B.1.4: Prompting and Generation (Master Pipeline)
Description: Executes the Two-Stage Retrieval (FAISS + Cross-Encoder) 
and feeds context into Qwen2.5-0.5B-Instruct for grounded generation.
"""
import os
import json
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer

# Disable Windows symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

class SouthAsianCulinaryRAG:
    def __init__(self, index_path, metadata_path, llm_name="Qwen/Qwen2.5-0.5B-Instruct"):
        print("Loading BGE Embedding Model...")
        self.embed_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        
        print("Loading Cross-Encoder Re-ranker...")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        print("Loading FAISS Index & Metadata...")
        self.index = faiss.read_index(index_path)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata_store = json.load(f)
            
        print(f"Loading LLM ({llm_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.model = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype="auto", device_map="auto")
        print("✅ System Ready!")

    def retrieve(self, query, final_k=5):
        # STAGE 1: Broad Search
        query_vector = self.embed_model.encode([query])
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), 20)

        candidates = [self.metadata_store[idx].copy() for idx in indices[0] if idx != -1]

        # STAGE 2: Re-ranking
        cross_scores = self.cross_encoder.predict([[query, c['text']] for c in candidates])
        for i, c in enumerate(candidates):
            c['rerank_score'] = float(cross_scores[i])
        candidates = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)

        # STAGE 3: Parent Deduplication
        final_results = []
        seen_parents = set()
        for chunk in candidates:
            uid = chunk['metadata'].get('parent_id') or chunk['text']
            if uid not in seen_parents:
                seen_parents.add(uid)
                chunk['text'] = chunk['metadata'].get('parent_text', chunk['text'])
                final_results.append(chunk)
            if len(final_results) == final_k: break
        return final_results

    def generate(self, query, contexts):
        context_block = "".join([f"--- Source {i+1} ({c['metadata'].get('topic', 'Unknown')}) ---\n{c['text']}\n\n" for i, c in enumerate(contexts)])
        
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
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=False)
        response_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        return self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0].strip()

if __name__ == "__main__":
    rag_system = SouthAsianCulinaryRAG("recipe_faiss_index.bin", "recipe_metadata.json")
    
    while True:
        user_query = input("\nAsk a recipe question (or 'quit'): ")
        if user_query.lower() in ['quit', 'exit']: break
        if not user_query.strip(): continue

        chunks = rag_system.retrieve(user_query)
        answer = rag_system.generate(user_query, chunks)
        
        print("\n" + "="*50)
        print(f"🤖 AI Chef:\n{answer}")
        print("="*50)