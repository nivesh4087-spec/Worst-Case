"""
Base Algorithms & Helper Module for RAG Worst-Case Study
"""
import time
import numpy as np
from typing import List, Dict, Tuple, Any

def cosine_similarity_batch(query: np.ndarray, matrices: np.ndarray) -> Tuple[np.ndarray, int]:
    query_norm = query / (np.linalg.norm(query) + 1e-9)
    matrices_norm = matrices / (np.linalg.norm(matrices, axis=1, keepdims=True) + 1e-9)
    sims = np.dot(matrices_norm, query_norm)
    ops = matrices.shape[0]
    return sims, ops

class TreeNode:
    def __init__(self, node_id: int, level: int, vector: np.ndarray, passage_id: int = None, document_id: int = None):
        self.node_id = node_id
        self.level = level
        self.vector = vector
        self.passage_id = passage_id
        self.document_id = document_id
        self.children: List['TreeNode'] = []
        self.parent: 'TreeNode' = None

class FlatDPRSearch:
    """Exact Flat Vector Search baseline (DPR)"""
    def __init__(self, vectors: np.ndarray, passage_ids: List[int], doc_ids: List[int]):
        self.vectors = vectors
        self.passage_ids = np.array(passage_ids)
        self.doc_ids = np.array(doc_ids)
        self.dim = vectors.shape[1]

    def search(self, query_vector: np.ndarray, k: int = 10) -> Dict[str, Any]:
        start_time = time.perf_counter()
        sims, ops = cosine_similarity_batch(query_vector, self.vectors)
        top_k_indices = np.argsort(sims)[::-1][:k]
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "retrieved_passages": self.passage_ids[top_k_indices].tolist(),
            "retrieved_docs": self.doc_ids[top_k_indices].tolist(),
            "scores": sims[top_k_indices].tolist(),
            "ops": ops,
            "latency_ms": latency_ms,
            "fallback_triggered": False
        }

class HNSWGraphSearch:
    """Simulated HNSW (Hierarchical Navigable Small World) Graph Search"""
    def __init__(self, vectors: np.ndarray, passage_ids: List[int], doc_ids: List[int], ef_construction: int = 64, M: int = 16):
        self.vectors = vectors
        self.passage_ids = np.array(passage_ids)
        self.doc_ids = np.array(doc_ids)
        self.ef_construction = ef_construction
        self.M = M
        self.dim = vectors.shape[1]
        
    def search(self, query_vector: np.ndarray, k: int = 10, ef_search: int = 32) -> Dict[str, Any]:
        start_time = time.perf_counter()
        num_candidates = min(len(self.vectors), int(ef_search * np.log2(len(self.vectors) + 1)))
        np.random.seed(len(self.vectors))
        sampled_indices = np.random.choice(len(self.vectors), size=num_candidates, replace=False)
        sampled_vecs = self.vectors[sampled_indices]
        
        sims, ops = cosine_similarity_batch(query_vector, sampled_vecs)
        top_k_indices = np.argsort(sims)[::-1][:k]
        
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        
        return {
            "retrieved_passages": self.passage_ids[sampled_indices[top_k_indices]].tolist(),
            "retrieved_docs": self.doc_ids[sampled_indices[top_k_indices]].tolist(),
            "scores": sims[top_k_indices].tolist(),
            "ops": ops,
            "latency_ms": latency_ms,
            "fallback_triggered": False
        }
