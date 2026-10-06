"""
Dataset Generator Module for RAG Worst-Case vs Best-Case Empirical Analysis
Generates:
- Dataset A: Ideal Best-Case Corpus (high hierarchy alignment)
- Dataset B: Semantic Drift Worst Case (abstract node drift causing false pruning)
- Dataset C: Low-Confidence Fallback Worst Case (OOD queries triggering full fallback search)
- Dataset D: Cluster Skew Worst Case (heavily imbalanced tree branches)
- Scaled Corpora: Scaling tests from 1K to 100K passages.
"""

import numpy as np
from typing import Dict, Any, Tuple, List

def normalize(vecs: np.ndarray) -> np.ndarray:
    if vecs.ndim == 1:
        return vecs / (np.linalg.norm(vecs) + 1e-9)
    return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)

def generate_ideal_dataset(num_docs: int = 10, passages_per_doc: int = 50, dim: int = 128, seed: int = 42) -> Dict[str, Any]:
    """Dataset A: Best-case scenario with well-separated semantic clusters."""
    np.random.seed(seed)
    N = num_docs * passages_per_doc
    
    doc_centroids = np.random.randn(num_docs, dim)
    doc_centroids = normalize(doc_centroids)
    
    passages = []
    doc_ids = []
    passage_ids = []
    
    p_id = 0
    for d_idx in range(num_docs):
        c = doc_centroids[d_idx]
        noise = np.random.randn(passages_per_doc, dim) * 0.15
        p_vecs = c + noise
        p_vecs = normalize(p_vecs)
        
        passages.append(p_vecs)
        doc_ids.extend([d_idx] * passages_per_doc)
        passage_ids.extend(list(range(p_id, p_id + passages_per_doc)))
        p_id += passages_per_doc
        
    all_passages = np.vstack(passages)
    
    # Generate queries aligned with doc 0
    target_p_idx = 5
    target_vec = all_passages[target_p_idx]
    query_vec = target_vec + np.random.randn(dim) * 0.05
    query_vec = query_vec / np.linalg.norm(query_vec)
    
    return {
        "name": "Dataset A (Best Case - Balanced Hierarchy)",
        "vectors": all_passages,
        "passage_ids": passage_ids,
        "doc_ids": doc_ids,
        "queries": [query_vec],
        "ground_truth": [[target_p_idx]]
    }

def generate_semantic_drift_worst_case(num_docs: int = 10, passages_per_doc: int = 50, dim: int = 128, seed: int = 101) -> Dict[str, Any]:
    """Dataset B: Worst-case where leaf passage is highly relevant but abstract summary parent node is drifted."""
    np.random.seed(seed)
    ds = generate_ideal_dataset(num_docs, passages_per_doc, dim, seed)
    vectors = ds["vectors"]
    
    # Select target leaf passage at index 12
    target_idx = 12
    target_vec = vectors[target_idx].copy()
    
    # Perturb all other sibling passages in the same document cluster in opposite direction
    # so that the cluster centroid (abstract node) drifts away from the target leaf & query
    siblings_start = 0
    siblings_end = passages_per_doc
    opposite_direction = -target_vec
    
    for idx in range(siblings_start, siblings_end):
        if idx != target_idx:
            vectors[idx] = normalize(opposite_direction + np.random.randn(dim) * 0.1)
            
    query_vec = normalize(target_vec + np.random.randn(dim) * 0.02)
    
    return {
        "name": "Dataset B (Worst Case - Semantic Drift & Path Miss)",
        "vectors": vectors,
        "passage_ids": ds["passage_ids"],
        "doc_ids": ds["doc_ids"],
        "queries": [query_vec],
        "ground_truth": [[target_idx]]
    }

def generate_low_confidence_worst_case(num_docs: int = 10, passages_per_doc: int = 50, dim: int = 128, seed: int = 202) -> Dict[str, Any]:
    """Dataset C: Out-of-Distribution noise query where similarity < tau at upper levels, forcing Flat Fallback."""
    np.random.seed(seed)
    ds = generate_ideal_dataset(num_docs, passages_per_doc, dim, seed)
    
    # Query orthogonal/unrelated to all doc clusters
    query_vec = np.random.randn(dim)
    query_vec = normalize(query_vec)
    
    # Find nearest passage index in ground truth for verification
    sims = np.dot(ds["vectors"], query_vec)
    nearest_idx = int(np.argmax(sims))
    
    return {
        "name": "Dataset C (Worst Case - Fallback Cascade / Low Confidence)",
        "vectors": ds["vectors"],
        "passage_ids": ds["passage_ids"],
        "doc_ids": ds["doc_ids"],
        "queries": [query_vec],
        "ground_truth": [[nearest_idx]]
    }

def generate_cluster_skew_worst_case(total_passages: int = 500, dim: int = 128, seed: int = 303) -> Dict[str, Any]:
    """Dataset D: Highly imbalanced distribution (80% in single cluster) creating tree skew."""
    np.random.seed(seed)
    
    num_dense = int(0.8 * total_passages)
    num_sparse = total_passages - num_dense
    
    dense_centroid = normalize(np.random.randn(1, dim))
    dense_vecs = normalize(dense_centroid + np.random.randn(num_dense, dim) * 0.02)
    
    sparse_vecs = normalize(np.random.randn(num_sparse, dim))
    
    all_vectors = np.vstack([dense_vecs, sparse_vecs])
    passage_ids = list(range(total_passages))
    doc_ids = [0] * num_dense + list(range(1, num_sparse + 1))
    
    target_idx = 5
    query_vec = normalize(all_vectors[target_idx] + np.random.randn(dim) * 0.01)
    
    return {
        "name": "Dataset D (Worst Case - Degenerate Cluster Skew)",
        "vectors": all_vectors,
        "passage_ids": passage_ids,
        "doc_ids": doc_ids,
        "queries": [query_vec],
        "ground_truth": [[target_idx]]
    }

def generate_scaling_corpora(sizes: List[int] = [1000, 5000, 10000, 25000], dim: int = 128, seed: int = 404) -> List[Dict[str, Any]]:
    """Generates synthetic datasets of varying scale N."""
    datasets = []
    for sz in sizes:
        np.random.seed(seed + sz)
        num_docs = max(5, sz // 50)
        p_per_doc = sz // num_docs
        ds = generate_ideal_dataset(num_docs=num_docs, passages_per_doc=p_per_doc, dim=dim, seed=seed+sz)
        ds["scale"] = sz
        datasets.append(ds)
    return datasets
