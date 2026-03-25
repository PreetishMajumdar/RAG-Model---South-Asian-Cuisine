"""
Phase B.1.5: Baseline Evaluation Pipeline
Responsibility: Preetish
Description: Evaluates the Baseline system to generate the output payload 
for Deliverable 5 and for comparison against the Advanced system.
"""
import json
# Import the Baseline class
from master_inference_baseline import BaselineRAG 

def run_baseline_generation(input_json_path, output_json_path):
    print("Initializing Baseline RAG System...")
    rag = BaselineRAG("baseline_faiss_index.bin", "baseline_metadata.json")
    
    print(f"Loading queries from: {input_json_path}")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
        
    output_payload = {"results": []}
    
    for item in input_data["queries"]:
        q_id = item["query_id"]
        query_text = item["query"]
        print(f"Processing Query [{q_id}]: {query_text}")
        
        # 1. Retrieve and Generate
        chunks = rag.retrieve(query_text, final_k=5)
        generated_answer = rag.generate(query_text, chunks)
        
        # 2. Format the retrieved context for the University Schema
        formatted_context = []
        for i, chunk in enumerate(chunks):
            formatted_context.append({
                "doc_id": str(i).zfill(3), 
                "text": chunk["text"]
            })
        
        # 3. Build Result Object
        result_obj = {
            "query_id": q_id,
            "query": query_text,
            "response": generated_answer,
            "retrieved_context": formatted_context
        }
        
        output_payload["results"].append(result_obj)
        
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_payload, f, indent=4, ensure_ascii=False)
        
    print(f"\n🎉 Success! Baseline output payload saved to: {output_json_path}")

if __name__ == "__main__":
    # Uses the same input file, but creates a specific baseline output file
    run_baseline_generation("input_payload.json", "output_payload_baseline.json")