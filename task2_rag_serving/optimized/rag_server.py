import torch

torch.cuda.empty_cache()
import argparse
import json
import os
import queue
import shutil
import threading
import time

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer, pipeline

app = FastAPI()

MAX_BATCH_SIZE = 8
MAX_WAITING_TIME = 0.1  # in seconds

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()
SERVER_ID = str(args.port)[3]

print(f"Server ID: {SERVER_ID}")

# JSON file to store results
RESULTS_FILE = f"results_server_{SERVER_ID}.json"

# Load existing results if available
response_dict = {}
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "w") as f:
        json.dump(response_dict, f)
else:
    with open(RESULTS_FILE, "r") as f:
        try:
            response_dict = json.load(f)
        except json.JSONDecodeError:
            response_dict = {}

response_lock = threading.RLock()

# Example documents
documents = [
    "Cats are small furry carnivores that are often kept as pets.",
    "Dogs are domesticated mammals, not natural wild animals.",
    "Hummingbirds can hover in mid-air by rapidly flapping their wings."
]

# Load embedding model
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)

# Basic Chat LLM
chat_pipeline = pipeline("text-generation", model="facebook/opt-125m")

request_queue = queue.Queue()


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


# Save results to JSON file
def save_results():
    temp_file = f"{RESULTS_FILE}.tmp"
    try:
        print(f"Saving {len(response_dict)} results to {RESULTS_FILE}")
        with open(temp_file, "w") as f:
            json.dump(response_dict, f)
        shutil.move(temp_file, RESULTS_FILE)
        print(f"Successfully saved results to {RESULTS_FILE}")
    except Exception as e:
        print(f"ERROR saving results: {str(e)}")

def processing_batch(batch_items):
    if not batch_items:
        return

    request_ids = [item[0] for item in batch_items]
    queries = [item[1] for item in batch_items]
    ks = [item[2] for item in batch_items]

    print(f"Processing batch of {len(queries)} requests")

    try:
        query_embeddings = batch_get_embeddings(queries)

        batch_results = {}

        for i, (request_id, query, k) in enumerate(zip(request_ids, queries, ks)):
            query_emb = query_embeddings[i].reshape(1,-1)

            retrieved_docs = retrieve_top_k(query_emb, k)
            context = "\n".join(retrieved_docs)
            prompt = f"Question: {query}\nContext:\n{context}\nAnswer:"

            generated = chat_pipeline(prompt, max_length=50, do_sample=True)[0]["generated_text"]
            print("Processing Batch")
            batch_results[request_id] = {
                "query": query,
                "result": generated
            }
            print(f"Print Batch Results: {batch_results[request_id]}")

        # Store result
        print("Storing Batch")
        print(f"BEFORE UPDATE: response_dict has {len(response_dict)} items")
        with response_lock:
            print("Acquired response_lock")
            old_count = len(response_dict)
            response_dict.update(batch_results)
            new_count = len(response_dict)
            print(f"Updated response_dict: {old_count} -> {new_count} items")

            print("Calling save_results()")
            save_results()
            print("Returned from save_results()")

    except Exception as e:
        error_message = f"Error processing request: {str(e)}"
        print(error_message)
        with response_lock:
            for request_id in request_ids:
                if request_id not in response_dict:
                    response_dict[request_id] = {
                        "status": "error",
                        "message": error_message
                    }
            save_results()

def batch_worker():
    batch_count = 0
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
            batch_count += 1
            if batch_count % 10 == 0:
                torch.cuda.empty_cache()

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
            save_results()
            return result
        else:
            return {"status": "processing or not found"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
