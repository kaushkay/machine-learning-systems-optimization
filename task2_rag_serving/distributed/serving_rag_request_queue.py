import torch
torch.cuda.empty_cache()
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

import queue
import threading
import time

app = FastAPI()

# Example documents in memory
documents = [
    "Cats are small furry carnivores that are often kept as pets.",
    "Dogs are domesticated mammals, not natural wild animals.",
    "Hummingbirds can hover in mid-air by rapidly flapping their wings."
]

# 1. Load embedding model
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)

# Basic Chat LLM
chat_pipeline = pipeline("text-generation", model="facebook/opt-125m")

request_queue = queue.Queue()

response_dict = {}
response_lock = threading.Lock()



def get_embedding(text: str) -> np.ndarray:
    """Compute a simple average-pool embedding."""
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Precompute document embeddings
doc_embeddings = np.vstack([get_embedding(doc) for doc in documents])

### You may want to use your own top-k retrieval method (task 1)
def retrieve_top_k(query_emb: np.ndarray, k: int = 2) -> list:
    """Retrieve top-k docs via dot-product similarity."""
    sims = doc_embeddings @ query_emb.T
    top_k_indices = np.argsort(sims.ravel())[::-1][:k]
    return [documents[i] for i in top_k_indices]

def rag_pipeline(query: str, k: int = 2, request_id: str=None) -> str:
    # Step 1: Input embedding
    query_emb = get_embedding(query)
    
    # Step 2: Retrieval
    retrieved_docs = retrieve_top_k(query_emb, k)
    
    # Construct the prompt from query + retrieved docs
    context = "\n".join(retrieved_docs)
    prompt = f"Question: {query}\nContext:\n{context}\nAnswer:"
    
    # Step 3: LLM Output
    generated = chat_pipeline(prompt, max_length=50, do_sample=True)[0]["generated_text"]
    
    if request_id:
        with response_lock:
            response_dict[request_id] = {
                "query": query,
                "result": generated
            }
    
    return generated

def process_queue():
    while True:
        try:
            request_id, query, k = request_queue.get(block=True)
            rag_pipeline(query, k, request_id)
            request_queue.task_done()
        except Exception as e:
            print(f"Error Processing the Request: {e}")


worker_thread = threading.Thread(target=process_queue, daemon=True)
worker_thread.start()

# Define request model
class QueryRequest(BaseModel):
    query: str
    k: int = 2
    

@app.post("/rag")
async def predict(payload: QueryRequest):
    request_id = f"req_{int(time.time() * 1000)}"
    
    request_queue.put((request_id, payload.query, payload.k))
    
    return {
        "request_id": request_id,
        "status": "processing",
    }

@app.get("/result/{request_id}")
async def get_result(request_id:str):
    with response_lock:
        if request_id in response_dict:
            result = response_dict[request_id]
            del response_dict[request_id]
            return result
        else:
            return {"status": "processing or not found"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)