import requests
import time
import json
import concurrent.futures
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class RAGServiceTester:
    def __init__(self, base_url="https://localhost:8000"):
        self.base_url = base_url
        self.rag_endpoint = f"{base_url}/rag"
        self.result_endpoint = f"{base_url}/result"
        self.questions=[
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
        
        self.results=defaultdict(list)
        
    
    def send_single_request(self, query_idx, k=2, timeout=60, optimized=False):
        query = self.questions[query_idx % len(self.questions)]
        
        start_time = time.time()
        try:
            payload = {
                "query":query,
                "k":k
            }
            
            response = requests.post(self.rag_endpoint, json=payload, timeout=timeout)
            # end_time = time.time()
            
            if response.status_code != 200:
                end_time = time.time()
                return {
                    "status": "error",
                    "query": query,
                    "response_time": end_time - start_time,
                    "result": f"HTTP Error: {response.status_code}"
                }
                
            if optimized:
                request_id = response.json().get("request_id")
                result = self.poll_result(request_id, timeout)
                end_time = time.time()
                
                if result:
                    return{
                        "status": "success",
                        "query": query,
                        "response_time": end_time - start_time,
                        "result": result
                    }
                else:
                    return{
                        "status": "error",
                        "query": query,
                        "response_time": end_time - start_time,
                        "result": f"Polling Error or Timeout!"
                    }
                
            else:
                end_time = time.time()
                return {
                    "status":"success",
                    "query": query,
                    "response_time": end_time - start_time,
                    "result": response.json()
                }
            
                       
        except requests.exceptions.Timeout:
            end_time = time.time()
            return {
                "status": "error",
                    "query": query,
                    "response_time": end_time - start_time,
                    "error": "Request Timeout"
            }
            
        except Exception as e:
            end_time = time.time()
            return {
                "status": "error",
                "query": query,
                "response_time": end_time - start_time,
                "error": str(e)
            }
            
    def poll_result(self, request_id, timeout=60, poll_interval=0.2):
        max_polls = int(timeout/poll_interval)
        
        for _ in range(max_polls):
            try:
                response = requests.get(f"{self.result_endpoint}/{request_id}")
                
                if response.status_code != 200:
                    time.sleep(poll_interval)
                    continue
            
                result = response.json()
                if "status" not in result or result["status"] != "processing or not found":
                    return result
                
                time.sleep(poll_interval)
            except Exception as e:
                print(f"Error polling for result: {e}")
                time.sleep(poll_interval)
        
        return None
            
    
    def run_concurrent_test(self, num_requests=10, concurrency=1, timeout=60, optimized=False):
        """Run a test with specified concurrency level."""
        print(f"Running test with {num_requests} requests at concurrency level {concurrency}")
        
        start_time=time.time()
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(self.send_single_request, i, timeout=timeout, optimized=optimized) for i in range(num_requests)]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                
        end_time = time.time()
        total_time = end_time - start_time
        
        success_count = sum(1 for r in results if r["status"] == "success")
        error_count = sum(1 for r in results if r["status"] == "error")
        
        if success_count > 0:
            successful_times = [r["response_time"] for r in results if r["status"] == "success"]
            avg_response_time = sum(successful_times) / len(successful_times)
            min_response_time = min(successful_times)
            max_response_time = max(successful_times)
            p50_response_time = np.percentile(successful_times, 50)
            p95_response_time = np.percentile(successful_times, 95)
            p99_response_time = np.percentile(successful_times, 99)
        else:
            avg_response_time = 0
            min_response_time = 0
            max_response_time = 0
            p50_response_time = 0
            p95_response_time = 0
            p99_response_time = 0
            
        throughput = num_requests / total_time if total_time > 0 else 0
        
        test_result = {
            "concurrency": concurrency,
            "num_requests": num_requests,
            "success_count": success_count,
            "error_count": error_count,
            "total_time": total_time,
            "throughput": throughput,
            "avg_response_time": avg_response_time,
            "min_response_time": min_response_time,
            "max_response_time": max_response_time,
            "p50_response_time": p50_response_time,
            "p95_response_time": p95_response_time,
            "p99_response_time": p99_response_time
        }
        
        self.results["concurrency"].append(concurrency)
        self.results["throughput"].append(throughput)
        self.results["avg_latency"].append(avg_response_time)
        self.results["p95_latency"].append(p95_response_time)
        self.results["p99_latency"].append(p99_response_time)
        self.results["success_rate"].append(success_count / num_requests if num_requests > 0 else 0)
        
        # Print the summary
        print(f"\nTest Results (Concurrency: {concurrency}):")
        print(f"Total requests: {num_requests}")
        print(f"Successful requests: {success_count}")
        print(f"Failed requests: {error_count}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Throughput: {throughput:.2f} requests/second")
        print(f"Average response time: {avg_response_time:.2f} seconds")
        print(f"Min response time: {min_response_time:.2f} seconds")
        print(f"Max response time: {max_response_time:.2f} seconds")
        print(f"P50 response time: {p50_response_time:.2f} seconds")
        print(f"P95 response time: {p95_response_time:.2f} seconds")
        print(f"P99 response time: {p99_response_time:.2f} seconds")
        
        return test_result
    

    def run_load_test(self, num_requests=10, concurrency_levels=None, timeout=60, optimized=False):
        if concurrency_levels is None:
            concurrency_levels = [1,2,5,10]
        
        results = {}
        
        for concurrency in concurrency_levels:
            print(f"\n{'='*50}")
            print(f"Testing with concurrency level: {concurrency}")
            print(f"{'='*50}")
            result = self.run_concurrent_test(num_requests, concurrency, timeout, optimized)
            results[f"concurrency_{concurrency}"] = result
            
            time.sleep(2)

        return results
    
    def plot_results(self, output_file="rag_performance.png"):
        """Plot the test results"""
        concurrency = self.results["concurrency"]
        
        fig, axs = plt.subplots(2, 2, figsize = (12,10))
        
        # Throughput vs Concurrency
        axs[0, 0].plot(concurrency, self.results["throughput"], 'o-', linewidth=2)
        axs[0, 0].set_title('Throughput vs Concurrency')
        axs[0, 0].set_xlabel('Concurrency')
        axs[0, 0].set_ylabel('Throughput (req/s)')
        axs[0, 0].grid(True)
        
        # Average Latency vs Concurrency
        axs[0, 1].plot(concurrency, self.results["avg_latency"], 'o-', linewidth=2)
        axs[0, 1].set_title('Average Latency vs Concurrency')
        axs[0, 1].set_xlabel('Concurrency')
        axs[0, 1].set_ylabel('Latency (s)')
        axs[0, 1].grid(True)
        
        # P95 and P99 Latency vs Concurrency
        axs[1, 0].plot(concurrency, self.results["p95_latency"], 'o-', linewidth=2, label='P95')
        axs[1, 0].plot(concurrency, self.results["p99_latency"], 's-', linewidth=2, label='P99')
        axs[1, 0].set_title('P95 and P99 Latency vs Concurrency')
        axs[1, 0].set_xlabel('Concurrency')
        axs[1, 0].set_ylabel('Latency (s)')
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        
        # Success Rate vs Concurrency
        axs[1, 1].plot(concurrency, self.results["success_rate"], 'o-', linewidth=2)
        axs[1, 1].set_title('Success Rate vs Concurrency')
        axs[1, 1].set_xlabel('Concurrency')
        axs[1, 1].set_ylabel('Success Rate')
        axs[1, 1].set_ylim(0, 1.05)
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Results plot saved to {output_file}")
        
        return fig
    
    def save_results_to_json(self, output_file="rag_test_results.json"):
        """Save the test results to a JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to {output_file}")   
    
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a RAG service.")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="Base URL of the RAG service")
    parser.add_argument("--requests", type=int, default=10, help="Number of requests per concurrency level")
    parser.add_argument("--concurrency", type=str, default="1,2,5,10", help="Comma-separated list of concurrency levels")
    parser.add_argument("--timeout", type=int, default=60, help="Requests timeout in seconds")
    parser.add_argument("--output", type=str, default="test_results", help="Base name for output files")
    parser.add_argument("--optimized", action="store_true",default=False, help="Set if testing an optimized implementation")
    
    
    
    args = parser.parse_args()
    
    concurrency_levels = [int(c) for c in args.concurrency.split(",")]
    
    print(f"Testing RAG service at {args.url}")
    print(f"Using {'Optimized' if args.optimized else 'Baseline'} implementation")
    
    tester = RAGServiceTester(base_url=args.url)
    results=tester.run_load_test(
        num_requests=args.requests,
        concurrency_levels=concurrency_levels,
        timeout=args.timeout,
        optimized=args.optimized
    )
    
    tester.plot_results(f"{args.output}.png")
    tester.save_results_to_json(f"{args.output}.json")