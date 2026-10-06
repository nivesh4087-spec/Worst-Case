"""
Benchmark Execution Suite for RAG Worst-Case Study
Runs HAT, Flat DPR, RAPTOR, and HNSW across Datasets A, B, C, D, and E.
Saves JSON results to `results/benchmark_results.json`.
"""

import json
import os
import time
import numpy as np
from src.algorithms import HierarchicalAbstractTree, FlatDPRSearch, RAPTORSearch, HNSWGraphSearch
from src.dataset_generator import (
    generate_ideal_dataset,
    generate_semantic_drift_worst_case,
    generate_low_confidence_worst_case,
    generate_cluster_skew_worst_case,
    generate_scaling_corpora
)

def evaluate_retrieval(retrieved: list, ground_truth: list, k: int = 10) -> tuple:
    """Calculates Recall@k, nDCG@k, and Hit@k"""
    if not ground_truth:
        return 0.0, 0.0, 0.0
    
    gt_set = set(ground_truth)
    hits = [1 if p in gt_set else 0 for p in retrieved[:k]]
    
    recall = sum(hits) / float(len(gt_set))
    hit_rate = 1.0 if sum(hits) > 0 else 0.0
    
    # nDCG calculation
    dcg = sum([h / np.log2(idx + 2) for idx, h in enumerate(hits)])
    idcg = sum([1.0 / np.log2(idx + 2) for idx in range(min(k, len(gt_set)))])
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    return float(recall), float(ndcg), float(hit_rate)


def run_single_dataset_benchmark(dataset: dict, tau: float = 0.5, beam_width: int = 3, k: int = 10) -> dict:
    vectors = dataset["vectors"]
    p_ids = dataset["passage_ids"]
    d_ids = dataset["doc_ids"]
    queries = dataset["queries"]
    gts = dataset["ground_truth"]
    
    # Instantiate models
    hat = HierarchicalAbstractTree(vectors, p_ids, d_ids, beam_width=beam_width, tau=tau)
    flat = FlatDPRSearch(vectors, p_ids, d_ids)
    raptor = RAPTORSearch(vectors, p_ids, d_ids)
    hnsw = HNSWGraphSearch(vectors, p_ids, d_ids)
    
    results = {
        "dataset_name": dataset["name"],
        "num_passages": len(vectors),
        "algorithms": {}
    }
    
    algos = {
        "HAT": hat,
        "Flat_DPR": flat,
        "RAPTOR": raptor,
        "HNSW": hnsw
    }
    
    for name, algo in algos.items():
        recalls, ndcgs, latencies, ops_list = [], [], [], []
        fallbacks = 0
        
        for q, gt in zip(queries, gts):
            res = algo.search(q, k=k)
            rec, ndcg, _ = evaluate_retrieval(res["retrieved_passages"], gt, k=k)
            
            recalls.append(rec)
            ndcgs.append(ndcg)
            latencies.append(res["latency_ms"])
            ops_list.append(res["ops"])
            if res.get("fallback_triggered", False):
                fallbacks += 1

        results["algorithms"][name] = {
            "avg_recall": float(np.mean(recalls)),
            "avg_ndcg": float(np.mean(ndcgs)),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "avg_ops": float(np.mean(ops_list)),
            "fallback_rate": float(fallbacks / len(queries))
        }
        
    return results


def run_all_benchmarks():
    print("=== Starting RAG Best-Case vs Worst-Case Empirical Benchmarks ===")
    
    ds_a = generate_ideal_dataset()
    ds_b = generate_semantic_drift_worst_case()
    ds_c = generate_low_confidence_worst_case()
    ds_d = generate_cluster_skew_worst_case()
    
    datasets = [ds_a, ds_b, ds_c, ds_d]
    benchmark_outputs = []
    
    for ds in datasets:
        print(f"Running benchmark on: {ds['name']}...")
        out = run_single_dataset_benchmark(ds)
        benchmark_outputs.append(out)

    # Scale benchmark
    scale_sizes = [1000, 2500, 5000, 10000, 20000]
    scale_corpora = generate_scaling_corpora(scale_sizes)
    scaling_results = []
    
    for sc in scale_corpora:
        print(f"Running scaling test for N={sc['scale']}...")
        out = run_single_dataset_benchmark(sc)
        scaling_results.append(out)

    final_payload = {
        "qualitative_benchmarks": benchmark_outputs,
        "scaling_benchmarks": scaling_results
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)
        
    print("Benchmark execution completed! Results saved to `results/benchmark_results.json`.")

if __name__ == "__main__":
    run_all_benchmarks()
