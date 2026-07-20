from typing import List, Dict, Tuple, Optional
from .tree import UsageDependencyTree, FastLibraNode
from .policy import CostModelSwapper

class UnifiedBlockManager:
    """
    Unified HBM and Main Memory Cache Pool Manager.
    As described in Section 4.3 of the FastLibra paper.
    
    This replaces independent LoRA and KV cache managers by treating them
    as unified memory blocks in a single pool, managed by the Dependency Tree
    and Cost Model Swapper.
    """
    def __init__(self, total_hbm_blocks: int, total_cpu_blocks: int):
        self.total_hbm_blocks = total_hbm_blocks
        self.total_cpu_blocks = total_cpu_blocks
        
        # State tracking
        self.used_hbm_blocks = 0
        self.used_cpu_blocks = 0
        
        self.tree = UsageDependencyTree()
        self.swapper = CostModelSwapper()
        
        self.total_requests = 0
        
    def can_allocate(self, num_blocks: int) -> bool:
        """Check if HBM has space, or if we can free up space."""
        return (self.used_hbm_blocks + num_blocks <= self.total_hbm_blocks) or \
               (self.total_hbm_blocks > 0) # Assumes we can always swap out if > 0

    def allocate_lora(self, lora_id: str):
        self.total_requests += 1
        node = self.tree.add_lora(lora_id)
        if not node.in_hbm:
            self._ensure_space(1)
            node.in_hbm = True
            self.used_hbm_blocks += 1
            if self.used_cpu_blocks > 0:
                self.used_cpu_blocks -= 1

    def allocate_kv(self, lora_id: str, token_id: str, parent_id: Optional[str] = None):
        node = self.tree.add_kv_cache(lora_id, token_id, parent_id)
        if not node.in_hbm:
            self._ensure_space(1)
            node.in_hbm = True
            self.used_hbm_blocks += 1
            if self.used_cpu_blocks > 0:
                self.used_cpu_blocks -= 1
                
    def _ensure_space(self, blocks_needed: int):
        """Free up HBM blocks if necessary by swapping out to CPU."""
        while self.used_hbm_blocks + blocks_needed > self.total_hbm_blocks:
            # 1. Ask the swapper for eviction candidates
            evict_nodes = self.swapper.get_swap_out_decisions(
                self.tree, self.total_requests, slots_needed=1
            )
            
            if not evict_nodes:
                raise RuntimeError("VRAM OOM: No nodes available to evict.")
                
            # 2. Swap out the worst node
            node = evict_nodes[0]
            node.in_hbm = False
            self.used_hbm_blocks -= 1
            self.used_cpu_blocks += 1
            
            # Simulated Async Transfer via CUDA Stream would happen here
            
    def perform_background_swap(self):
        """
        When HBM is idle, swap in valuable nodes from CPU RAM.
        """
        available_slots = self.total_hbm_blocks - self.used_hbm_blocks
        if available_slots > 0:
            swap_in_nodes = self.swapper.get_swap_in_decisions(
                self.tree, self.total_requests, available_slots
            )
            for node in swap_in_nodes:
                node.in_hbm = True
                self.used_hbm_blocks += 1
                self.used_cpu_blocks -= 1
