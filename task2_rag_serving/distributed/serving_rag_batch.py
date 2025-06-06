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

MAX_BATCH_SIZE = 8
MAX_WAITING_TIME = 0.1  #in seconds

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


# def get_embedding(text: str) -> np.ndarray:
#     """Compute a simple average-pool embedding."""
#     inputs = embed_tokenizer(text, return_tensors="pt", truncation=True)
#     with torch.no_grad():
#         outputs = embed_model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Precompute document embeddings
# doc_embeddings = np.vstack([get_embedding(doc) for doc in documents])

def batch_get_embeddings(queries: list):
    inputs = embed_tokenizer(queries, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = embed_model(**inputs)
    
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

doc_embeddings = batch_get_embeddings(documents)

### You may want to use your own top-k retrieval method (task 1)
def retrieve_top_k(query_emb: np.ndarray, k: int = 2) -> list:
    """Retrieve top-k docs via dot-product similarity."""
    sims = doc_embeddings @ query_emb.T
    top_k_indices = np.argsort(sims.ravel())[::-1][:k]
    return [documents[i] for i in top_k_indices]


def processing_batch(batch_items):
    if not batch_items:
        return
    
    request_ids = [item[0] for item in batch_items]
    queries = [item[1] for item in batch_items]
    ks = [item[2] for item in batch_items]
    
    print(f"Processing batch of {len(queries)} requests")
    
    try:
        query_embeddings = batch_get_embeddings(queries)
        
        for i, (request_id, query, k) in enumerate(zip(request_ids, queries, ks)):
            query_emb = query_embeddings[i].reshape(1,-1)
            
            retrieved_docs = retrieve_top_k(query_emb, k)
            context = "\n".join(retrieved_docs)
            prompt = f"Question: {query}\nContext:\n{context}\nAnswer:"
            
            generated = chat_pipeline(prompt, max_length=50, do_sample=True)[0]["generated_text"]
            
            # Store result
            with response_lock:
                response_dict[request_id] = {
                    "query": query,
                    "result": generated
                }
                
    except Exception as e:
        with response_lock:
            for request_id in request_ids:
                if request_id not in response_dict:
                    response_dict[request_id] = {
                        "status": "error",
                        "message": f"Error processing request: {str(e)}"
                    }
                    
def batch_worker():
    while True:
        try:
            batch = []
            
            try:
                first_request = request_queue.get(block=True, timeout=1.0)
                batch.append(first_request)
                request_queue.task_done()
                
            except queue.Empty:
                continue
            
            batch_start_time = time.time()
            
            while len(batch) < MAX_BATCH_SIZE:
                try:
                    elapsed_time = time.time() - batch_start_time
                    remaining_wait = max(0, MAX_WAITING_TIME - elapsed_time)
                    
                    if remaining_wait <=0:
                        break
                    
                    next_request = request_queue.get(block=True, timeout=remaining_wait)
                    batch.append(next_request)
                    request_queue.task_done()
                    
                except queue.Empty:
                    break
                
            processing_batch(batch)
           
        except Exception as e:
            print(f"Error in batch worker: {e}")
            

batch_thread = threading.Thread(target=batch_worker, daemon=True)
batch_thread.start()


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