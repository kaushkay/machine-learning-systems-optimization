import asyncio
import itertools

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException

app = FastAPI()
import json
import os

# List of RAG servers (replace with actual URLs or IPs)
rag_servers = {
    "http://localhost:8001": "results_server_1.json",
    "http://localhost:8002": "results_server_2.json",
    "http://localhost:8003": "results_server_3.json",
}

# Round-robin iterator
server_iterator = itertools.cycle(rag_servers.keys())

# Store health status of servers
server_health = {server: True for server in rag_servers}


async def check_server_health():
    """Periodically checks the health of RAG servers asynchronously."""
    async with httpx.AsyncClient() as client:
        while True:
            for server in rag_servers.keys():
                try:
                    response = await client.get(f"{server}/health", timeout=2)
                    server_health[server] = response.status_code == 200
                except Exception:
                    server_health[server] = False
            await asyncio.sleep(10)  # Check health every 10 seconds


@app.on_event("startup")
async def startup_event():
    """Start background health check task."""
    asyncio.create_task(check_server_health())


@app.get("/query")
async def query_rag(query: str, k: int):
    """Load balances the query across healthy RAG servers asynchronously."""
    async with httpx.AsyncClient() as client:
        for _ in range(len(rag_servers)):  # Try all servers until a healthy one is found
            server = next(server_iterator)
            if server_health.get(server, False):  # Ensure server is healthy
                try:
                    payload = {"query": query, "k": k}
                    response = await client.post(f"{server}/rag", json=payload, timeout=60)
                    return response.json()
                except Exception:
                    server_health[server] = False  # Mark server as unhealthy
        raise HTTPException(status_code=503, detail="No healthy RAG servers available")

# @app.get("/fetch_result/{request_id}")
# async def fetch_result(request_id: str):
#     """Fetches the result for a given request_id from any RAG server."""
#     async with httpx.AsyncClient() as client:
#         for server in rag_servers:
#             try:
#                 response = await client.get(f"{server}/result/{request_id}", timeout=10)
#                 result = response.json()
#                 if result.get("status") != "processing or not found":
#                     return result  # Return result if found
#             except Exception:
#                 continue  # Try the next server if one fails
#     return {"status": "processing or not found"}

@app.get("/fetch_result/{request_id}")
async def fetch_result(request_id: str):
    """Fetches the result from JSON files instead of querying servers."""
    for json_file in rag_servers.values():
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                try:
                    results = json.load(f)
                    if request_id in results:
                        return results[request_id]
                except json.JSONDecodeError:
                    continue  # Skip corrupt files
    return {"status": "processing or not found"}

@app.get("/health")
async def health_check():
    """Health check endpoint for the load balancer."""
    return {"status": "ok", "healthy_servers": [s for s in server_health if server_health[s]]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
