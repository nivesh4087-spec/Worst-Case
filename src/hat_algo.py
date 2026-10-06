"""
Hierarchical Abstract Tree (HAT) and RAPTOR Implementation
"""
import time
import numpy as np
from typing import List, Dict, Any
from src.base_algos import TreeNode, FlatDPRSearch, cosine_similarity_batch

class HierarchicalAbstractTree:
    """HAT implementation with coarse-to-fine beam traversal & threshold fallback."""
    def __init__(self, passage_vectors: np.ndarray, passage_ids: List[int], doc_ids: List[int], 
                 cluster_size: int = 8, beam_width: int = 3, tau: float = 0.5):
        self.passage_vectors = passage_vectors
        self.passage_ids = passage_ids
        self.doc_ids = doc_ids
        self.cluster_size = cluster_size
        self.beam_width = beam_width
        self.tau = tau
        self.nodes: Dict[int, TreeNode] = {}
        self.root: TreeNode = None
        self.tree_depth = 0
        self.flat_searcher = FlatDPRSearch(passage_vectors, passage_ids, doc_ids)
        self._build_tree()

    def _build_tree(self):
        leaf_nodes = []
        for i, (vec, pid, did) in enumerate(zip(self.passage_vectors, self.passage_ids, self.doc_ids)):
            node = TreeNode(node_id=i, level=0, vector=vec, passage_id=pid, document_id=did)
            leaf_nodes.append(node)
            self.nodes[i] = node

        node_counter = len(leaf_nodes)
        current_level_nodes = leaf_nodes
        depth = 0

        while len(current_level_nodes) > 1:
            depth += 1
            next_level_nodes = []
            num_nodes = len(current_level_nodes)
            group_size = max(2, min(self.cluster_size, num_nodes))
            
            for g in range(0, num_nodes, group_size):
                group = current_level_nodes[g:g+group_size]
                if not group:
                    continue
                group_vecs = np.array([n.vector for n in group])
                centroid = np.mean(group_vecs, axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
                
                abstract_node = TreeNode(node_id=node_counter, level=depth, vector=centroid)
                node_counter += 1
                for n in group:
                    abstract_node.children.append(n)
                    n.parent = abstract_node
                next_level_nodes.append(abstract_node)
                self.nodes[abstract_node.node_id] = abstract_node
            current_level_nodes = next_level_nodes

        self.root = current_level_nodes[0] if current_level_nodes else None
        self.tree_depth = depth

    def search(self, query_vector: np.ndarray, k: int = 10, beam_width: int = None, tau: float = None) -> Dict[str, Any]:
        beam_w = beam_width if beam_width is not None else self.beam_width
        t_val = tau if tau is not None else self.tau

        start_t = time.perf_counter()
        total_ops = 0

        if not self.root:
            return self.flat_searcher.search(query_vector, k)

        beam = self.root.children if self.root.children else [self.root]
        current_level = self.tree_depth - 1
        leaf_candidates, leaf_sims = [], np.array([])
        
        while current_level >= 0:
            if not beam:
                break
            candidate_vecs = np.array([node.vector for node in beam])
            sims, ops = cosine_similarity_batch(query_vector, candidate_vecs)
            total_ops += ops
            max_sim = float(np.max(sims)) if len(sims) > 0 else 0.0
            
            if max_sim < t_val and current_level > 0:
                fallback_res = self.flat_searcher.search(query_vector, k)
                latency_ms = (time.perf_counter() - start_t) * 1000.0
                return {
                    "retrieved_passages": fallback_res["retrieved_passages"],
                    "retrieved_docs": fallback_res["retrieved_docs"],
                    "scores": fallback_res["scores"],
                    "ops": total_ops + fallback_res["ops"],
                    "latency_ms": latency_ms,
                    "fallback_triggered": True,
                    "max_sim_before_fallback": max_sim
                }

            top_b_idx = np.argsort(sims)[::-1][:beam_w]
            selected_nodes = [beam[idx] for idx in top_b_idx]

            if current_level == 0:
                leaf_candidates = selected_nodes
                leaf_sims = sims[top_b_idx]
                break
            else:
                next_beam = []
                for node in selected_nodes:
                    next_beam.extend(node.children)
                beam = next_beam
                current_level -= 1

        if leaf_candidates:
            leaf_passages = [n.passage_id for n in leaf_candidates]
            leaf_docs = [n.document_id for n in leaf_candidates]
            sorted_idx = np.argsort(leaf_sims)[::-1][:k]
            ret_passages = [leaf_passages[i] for i in sorted_idx]
            ret_docs = [leaf_docs[i] for i in sorted_idx]
            ret_scores = [float(leaf_sims[i]) for i in sorted_idx]
        else:
            ret_passages, ret_docs, ret_scores = [], [], []

        latency_ms = (time.perf_counter() - start_t) * 1000.0
        return {
            "retrieved_passages": ret_passages,
            "retrieved_docs": ret_docs,
            "scores": ret_scores,
            "ops": total_ops,
            "latency_ms": latency_ms,
            "fallback_triggered": False,
            "max_sim_before_fallback": None
        }

class RAPTORSearch:
    """RAPTOR algorithm without dynamic fallback."""
    def __init__(self, passage_vectors: np.ndarray, passage_ids: List[int], doc_ids: List[int], cluster_size: int = 8):
        self.hat_tree = HierarchicalAbstractTree(passage_vectors, passage_ids, doc_ids, cluster_size=cluster_size, tau=-1.0)

    def search(self, query_vector: np.ndarray, k: int = 10, beam_width: int = 3) -> Dict[str, Any]:
        return self.hat_tree.search(query_vector, k=k, beam_width=beam_width, tau=-1.0)
