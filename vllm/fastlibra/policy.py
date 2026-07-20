import time
from typing import List, Dict, Any
from .tree import FastLibraNode, UsageDependencyTree

class CostModelSwapper:
    """
    Performance-Driven Cache Swapper
    As described in Section 5 of the FastLibra paper.
    Calculates Eva (Evaluation Score) for each node to determine swap-in/out priority.
    """
    def __init__(self, w1: float = 0.4, w2: float = 0.4, w3: float = 0.2):
        self.w1 = w1 # Weight for Visit Frequency
        self.w2 = w2 # Weight for LRU Time
        self.w3 = w3 # Weight for Swap Cost
        
    def _calculate_swap_cost(self, node: FastLibraNode) -> float:
        # Simulated swap cost based on node type (LoRA blocks are larger than KV blocks usually)
        return 1.5 if node.is_lora else 0.5
        
    def calculate_eva(self, node: FastLibraNode, total_requests: int) -> float:
        """
        Eva_i calculation. Higher Eva means higher priority to keep in HBM.
        """
        # Frequency component (FE_i)
        fe_i = node.visit_frequency / max(1, total_requests)
        
        # LRU component (normalized recentness)
        time_since_use = time.time() - node.last_used_time
        lru_i = 1.0 / (1.0 + time_since_use)
        
        # Swap cost component (higher cost = want to keep in HBM)
        swap_cost_i = self._calculate_swap_cost(node)
        
        eva = (self.w1 * fe_i) + (self.w2 * lru_i) + (self.w3 * swap_cost_i)
        return eva

    def calculate_eva_for_vllm_request(self, request: Any, total_requests: int) -> float:
        """
        Adapts the FastLibra Eva_i calculation for native vLLM Request objects.
        This is used to intercept the vLLM scheduler's preemption logic.
        """
        # Frequency component based on number of computed tokens vs total tokens
        # A request that has been processed more is "visited" more.
        visit_frequency = getattr(request, 'num_computed_tokens', 1)
        fe_i = visit_frequency / max(1, total_requests)
        
        # LRU component based on arrival time (vLLM uses arrival_time for LRU)
        time_since_use = time.time() - getattr(request, 'arrival_time', time.time())
        lru_i = 1.0 / (1.0 + time_since_use)
        
        # Swap cost: KVs are typically leaf nodes. LoRAs are handled at the block manager level.
        swap_cost_i = 0.5
        
        eva = (self.w1 * fe_i) + (self.w2 * lru_i) + (self.w3 * swap_cost_i)
        return eva

    def get_swap_out_decisions(self, tree: UsageDependencyTree, total_requests: int, slots_needed: int) -> List[FastLibraNode]:
        """
        Identify nodes to swap OUT of HBM.
        We only consider leaf nodes currently in HBM (to maintain dependencies).
        We sort them by lowest Eva score (least valuable to keep).
        """
        candidates = tree.get_eviction_candidates()
        
        # Calculate Eva for all candidates
        scored_candidates = [(node, self.calculate_eva(node, total_requests)) for node in candidates]
        
        # Sort by Eva (ascending) - lowest score gets evicted first
        scored_candidates.sort(key=lambda x: x[1])
        
        return [node for node, score in scored_candidates[:slots_needed]]

    def get_swap_in_decisions(self, tree: UsageDependencyTree, total_requests: int, available_slots: int) -> List[FastLibraNode]:
        """
        Identify nodes to swap INTO HBM.
        We only consider root-most missing nodes.
        We sort them by highest Eva score (most valuable to bring in).
        """
        candidates = tree.get_swap_in_candidates()
        
        scored_candidates = [(node, self.calculate_eva(node, total_requests)) for node in candidates]
        
        # Sort by Eva (descending) - highest score gets swapped in first
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [node for node, score in scored_candidates[:available_slots]]
