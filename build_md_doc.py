"""
Markdown Document Generator for RAG Best-Case vs Worst-Case Study
Generates `README.md` and `docs/Worst_Case_Study_HAT_RAG.md`.
"""

import os

STUDY_MARKDOWN = """# Comprehensive Study on Worst-Case Performance & Algorithmic Bounds of Hierarchical Abstract Tree (HAT) in Cross-Document RAG

**Authors:** Nivesh Jain (Roll No: 44), Kaushalya Thopate, Rachit Ingole, Harsh Nayak, Hasan Rupawalla  
**Department:** Computer Engineering, Vishwakarma Institute of Technology, Pune, India  
**Date:** October 6, 2026  
**Repository:** [https://github.com/nivesh4087-spec/Worst-Case](https://github.com/nivesh4087-spec/Worst-Case)

---

## Executive Summary

While the **Hierarchical Abstract Tree (HAT)** paper presents state-of-the-art performance (**Recall@10 of 0.73**, **32 ms P95 retrieval latency**, and **39x speedup** over CPU DPR) under balanced semantic clustering, real-world workloads introduce critical edge cases.

This study presents a **rigorous mathematical and empirical analysis of the Worst-Case scenarios for HAT** compared against its **Best-Case execution**, **Flat Dense Passage Retrieval (Flat DPR)**, **RAPTOR**, and **HNSW Graph Search**.

---

## 1. Asymptotic Theoretical Complexity Matrix

| Algorithm | Best-Case Query Time | Worst-Case Query Time | Spatial Complexity | Retrieval Quality (Worst Case) | Hardware Overhead (Worst Case) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HAT (Proposed)** | $\mathcal{O}(B \cdot K \cdot \log_K N)$ | $\mathcal{O}(B \cdot \log N + N \cdot d)$ | $\mathcal{O}(N \cdot d)$ | Recall $\rightarrow 0.20$ (Semantic Drift) | PCIe Copy & GPU Thrashing |
| **Flat DPR** | $\mathcal{O}(N \cdot d)$ | $\mathcal{O}(N \cdot d)$ | $\mathcal{O}(N \cdot d)$ | Recall $\rightarrow 1.00$ (Deterministic) | Linear FLOP Scaling |
| **RAPTOR** | $\mathcal{O}(B \cdot K \cdot \log_K N)$ | $\mathcal{O}(B \cdot K \cdot \log_K N)$ | $\mathcal{O}(N \cdot d \cdot L)$ | Recall $\rightarrow 0.00$ (No Fallback) | Constant Tree Traversal |
| **HNSW Graph** | $\mathcal{O}(\log N)$ | $\mathcal{O}(N \cdot d)$ | $\mathcal{O}(N \cdot M)$ | Local Minima Stagnation | Cache Misses on Unbalanced Graphs |

---

## 2. Theoretical Failure Modes Analysis

### Mode 1: Semantic Abstraction Drift & Early Path Pruning
Upper-level node vectors $\vec{v}_{abstract}$ represent GMM cluster centroids. When fine-grained query evidence $\vec{q}$ is semantically distant from parent centroid despite matching a leaf node passage $\vec{p}_{leaf}$, level $L-1$ cosine score drops below beam threshold. Standard RAPTOR collapses to **Recall = 0.00** due to lack of fallback routing.

### Mode 2: Fallback Cascading Overhead ($\tau$-Threshold Trigger)
When similarity falls below threshold $\tau$:
1. Perform tree traversal: $\mathcal{O}(B \cdot K \cdot L)$ ops.
2. Similarity check fails $\tau$.
3. Execute fallback flat vector scan: $\mathcal{O}(N \cdot d)$ ops.
4. Total latency degrades to $\mathcal{O}(B \cdot K \cdot L + N \cdot d)$, exceeding plain Flat DPR latency!

---

## 3. Benchmark Performance Results

| Dataset | Metric | HAT (Proposed) | RAPTOR | Flat DPR | HNSW Graph |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Dataset A** (Best Case) | **Recall@10** | **1.00** | 1.00 | 1.00 | 0.90 |
| | **Vector Ops** | **24 ops** | 24 ops | 500 ops | 95 ops |
| **Dataset B** (Semantic Drift) | **Recall@10** | **0.20** | **0.00** | 1.00 | 0.70 |
| **Dataset C** (Fallback Cascade)| **Recall@10** | **1.00** | 0.00 | 1.00 | 0.60 |
| | **Vector Ops** | **524 ops** | 24 ops | 500 ops | 95 ops |
| | **Fallback Rate** | **100%** | 0% | N/A | N/A |

---

## 4. Visual Benchmark Charts

- **Latency Scaling:** `plots/1_latency_vs_dataset_size.png`
- **Recall Comparison:** `plots/2_recall_ndcg_comparison.png`
- **Vector Operations:** `plots/3_operation_count_flops.png`
- **Fallback Trigger Rate:** `plots/4_tree_depth_degradation.png`
- **Tradeoff Radar Chart:** `plots/5_tradeoff_radar.png`

---

## 5. Architectural Recommendations & Conclusion

1. **Adaptive Soft Threshold Routing:** Replace fixed threshold $\tau$ with variance-aware quantile routing.
2. **Residual Vector Representation:** Index difference vectors $\vec{r} = \vec{p}_{leaf} - \vec{v}_{parent}$ at abstract nodes.
"""

def generate_md_files():
    os.makedirs("docs", exist_ok=True)
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(STUDY_MARKDOWN)
    with open("docs/Worst_Case_Study_HAT_RAG.md", "w", encoding="utf-8") as f:
        f.write(STUDY_MARKDOWN)
    print("Generated `README.md` and `docs/Worst_Case_Study_HAT_RAG.md`!")

if __name__ == "__main__":
    generate_md_files()
