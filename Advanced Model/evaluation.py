import json
# Import the RAG class we finalized earlier
from master_inference import SouthAsianCulinaryRAG 

def run_payload_generation(input_json_path, output_json_path):
    print("Initializing South Asian Culinary RAG System...")
    rag = SouthAsianCulinaryRAG("recipe_faiss_index.bin", "recipe_metadata.json")
    
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
        
        # 2. Format the retrieved context to match the required schema
        formatted_context = []
        for i, chunk in enumerate(chunks):
            formatted_context.append({
                # Formats the index as "000", "001", "002" etc.
                "doc_id": str(i).zfill(3), 
                "text": chunk["text"]
            })
        
        # 3. Build the specific result object the markers want
        result_obj = {
            "query_id": q_id,
            "query": query_text,
            "response": generated_answer,
            "retrieved_context": formatted_context
        }
        
        output_payload["results"].append(result_obj)
        
    # 4. Save the final payload
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_payload, f, indent=4, ensure_ascii=False)
        
    print(f"\n🎉 Success! Output payload saved to: {output_json_path}")

if __name__ == "__main__":
    run_payload_generation("input_payload.json", "output_payload.json")