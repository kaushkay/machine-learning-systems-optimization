import argparse
import concurrent.futures
import json
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import requests


class RAGServiceTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.query_endpoint = f"{base_url}/query"
        self.result_endpoint = f"{base_url}/fetch_result"
        self.questions = [
            "What animals can hover in the air?",
            "Tell me about cats as pets",
            "Are dogs wild animals?",
            "What makes hummingbirds special?",
            "How do cats compare to dogs as pets?",
            "What animals make good pets?",
            "Tell me about birds that can hover",
            "What is special about domesticated animals?",
            "Tell me about carnivores that people keep as pets",
            "What animals flap their wings rapidly?"
        ]
        self.results = defaultdict(list)

    def send_request(self, query_idx, k=2, timeout=60):
        query = self.questions[query_idx % len(self.questions)]
        start_time = time.time()
        try:
            response = requests.get(f"{self.query_endpoint}?query={query}&k={k}", timeout=timeout)
            if response.status_code != 200:
                return {"status": "error", "query": query, "response_time": time.time() - start_time, "result": f"HTTP {response.status_code}"}
            request_id = response.json().get("request_id")
            result = self.poll_result(request_id, timeout)
            return {"status": "success", "query": query, "response_time": time.time() - start_time, "result": result} if result else {"status": "error", "query": query, "response_time": time.time() - start_time, "result": "Polling Timeout!"}
        except requests.exceptions.Timeout:
            return {"status": "error", "query": query, "response_time": time.time() - start_time, "error": "Request Timeout"}
        except Exception as e:
            return {"status": "error", "query": query, "response_time": time.time() - start_time, "error": str(e)}

    def poll_result(self, request_id, timeout=60, poll_interval=0.2):
        max_polls = int(timeout / poll_interval)
        for _ in range(max_polls):
            try:
                response = requests.get(f"{self.result_endpoint}/{request_id}", timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    if result.get("status") != "processing or not found":
                        return result
            except Exception:
                pass
            time.sleep(poll_interval)
        return None

    def run_concurrent_test(self, num_requests=10, concurrency=1, timeout=60):
        print(f"Running test with {num_requests} requests at concurrency {concurrency}")
        start_time = time.time()
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(self.send_request, i, timeout=timeout) for i in range(num_requests)]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        total_time = time.time() - start_time
        success_count = sum(1 for r in results if r["status"] == "success")
        error_count = sum(1 for r in results if r["status"] == "error")
        successful_times = [r["response_time"] for r in results if r["status"] == "success"]
        avg_latency = np.mean(successful_times) if successful_times else 0
        p95_latency = np.percentile(successful_times, 95) if successful_times else 0
        p99_latency = np.percentile(successful_times, 99) if successful_times else 0
        throughput = num_requests / total_time if total_time > 0 else 0
        self.results["concurrency"].append(concurrency)
        self.results["throughput"].append(throughput)
        self.results["avg_latency"].append(avg_latency)
        self.results["p95_latency"].append(p95_latency)
        self.results["p99_latency"].append(p99_latency)
        self.results["success_rate"].append(success_count / num_requests if num_requests else 0)
        print(f"Test Completed: Success={success_count}, Errors={error_count}, Avg Latency={avg_latency:.2f}s, Throughput={throughput:.2f} req/s")
        return self.results

    def plot_results(self, output_file="rag_performance.png"):
        concurrency = self.results["concurrency"]
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs[0, 0].plot(concurrency, self.results["throughput"], 'o-', linewidth=2)
        axs[0, 0].set_title('Throughput vs Concurrency')
        axs[0, 1].plot(concurrency, self.results["avg_latency"], 'o-', linewidth=2)
        axs[0, 1].set_title('Average Latency vs Concurrency')
        axs[1, 0].plot(concurrency, self.results["p95_latency"], 'o-', linewidth=2, label='P95')
        axs[1, 0].plot(concurrency, self.results["p99_latency"], 's-', linewidth=2, label='P99')
        axs[1, 1].plot(concurrency, self.results["success_rate"], 'o-', linewidth=2)
        axs[1, 1].set_title('Success Rate vs Concurrency')
        axs[1, 1].set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Results plot saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RAG Load Balancer.")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="Base URL of Load Balancer")
    parser.add_argument("--requests", type=int, default=10, help="Number of requests per concurrency level")
    parser.add_argument("--concurrency", type=str, default="1,2,5,10", help="Comma-separated list of concurrency levels")
    args = parser.parse_args()
    tester = RAGServiceTester(base_url=args.url)
    for level in map(int, args.concurrency.split(",")):
        tester.run_concurrent_test(num_requests=args.requests, concurrency=level)
    tester.plot_results()
