
import asyncio
import sys
import os
import argparse
from pathlib import Path

# Add backend to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.evaluation.service import EvaluationService

async def run_benchmark(dataset="redial", sample_size=50, model="cohere"):
    print(f"Initializing EvaluationService...")
    service = EvaluationService()
    
    print(f"Starting Benchmark: Dataset={dataset}, Sample={sample_size}, Model={model}")
    print("-" * 60)
    
    try:
        result = await service.evaluate(
            dataset=dataset,
            sample_size=sample_size,
            start_index=0,
            n_sample=20, # Retrieval candidates
            top_k=5,     # Reranked top-k
            model=model
        )
        
        print("\n" + "="*30)
        print(f"BENCHMARK RESULTS ({model.upper()})")
        print("="*30)
        print(result["message"])
        print(f"Avg Recall@5 (Final):      {result['avg_recall']:.4f}")
        print(f"Avg Recall (Retrieval):    {result['avg_recall_retrieval']:.4f}")
        print(f"Avg Recall (Semantic):     {result['avg_recall_semantic']:.4f}")
        print(f"Avg Recall (Content):      {result['avg_recall_content']:.4f}")
        print(f"Avg Recall (Collab):       {result['avg_recall_collab']:.4f}")
        print("-" * 30)
        
    except Exception as e:
        print(f"Benchmark Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Evaluation Benchmark')
    parser.add_argument('--model', type=str, default='cohere', choices=['llm', 'cohere'], help='Reranking model')
    parser.add_argument('--size', type=int, default=50, help='Sample size')
    args = parser.parse_args()
    
    asyncio.run(run_benchmark(dataset="redial", sample_size=args.size, model=args.model))
