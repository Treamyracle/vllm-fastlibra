import time
from typing import Dict, Optional, List, Any

class FastLibraNode:
    def __init__(self, node_id: str, is_lora: bool = False, parent=None):
        self.node_id = node_id
        self.is_lora = is_lora
        self.parent = parent
        self.children: Dict[str, 'FastLibraNode'] = {}
        
        # Metadata for Cost Model
        self.visit_frequency: int = 1
        self.last_used_time: float = time.time()
        
        # True if the block is currently in GPU HBM, False if in CPU RAM
        self.in_hbm: bool = False
        
    def touch(self):
        self.visit_frequency += 1
        self.last_used_time = time.time()

class UsageDependencyTree:
    """
    FastLibra Dependency-Aware Cache Manager (Trie Tree)
    As described in Section 4 of the FastLibra paper.
    """
    def __init__(self):
        # Virtual root node connecting subtrees of different LoRAs
        self.root = FastLibraNode("VIRTUAL_ROOT", is_lora=False)
        self.root.in_hbm = True  # Always in memory
        
    def add_lora(self, lora_id: str) -> FastLibraNode:
        """Add a LoRA node to the second layer of the tree."""
        if lora_id not in self.root.children:
            node = FastLibraNode(lora_id, is_lora=True, parent=self.root)
            self.root.children[lora_id] = node
        else:
            self.root.children[lora_id].touch()
        return self.root.children[lora_id]
        
    def add_kv_cache(self, lora_id: str, token_sequence_id: str, parent_sequence_id: Optional[str] = None) -> FastLibraNode:
        """
        Add a KV cache block node below its corresponding LoRA or parent KV cache.
        """
        lora_node = self.add_lora(lora_id)
        
        if parent_sequence_id is None:
            # Direct child of LoRA
            parent_node = lora_node
        else:
            # Find the parent KV cache node
            parent_node = self._find_node(lora_node, parent_sequence_id)
            if not parent_node:
                parent_node = lora_node # Fallback
                
        if token_sequence_id not in parent_node.children:
            node = FastLibraNode(token_sequence_id, is_lora=False, parent=parent_node)
            parent_node.children[token_sequence_id] = node
        else:
            parent_node.children[token_sequence_id].touch()
            
        return parent_node.children[token_sequence_id]
        
    def _find_node(self, current_node: FastLibraNode, target_id: str) -> Optional[FastLibraNode]:
        if current_node.node_id == target_id:
            return current_node
        for child in current_node.children.values():
            result = self._find_node(child, target_id)
            if result:
                return result
        return None

    def get_eviction_candidates(self) -> List[FastLibraNode]:
        """
        Returns leaf nodes that are currently in HBM.
        FastLibra swaps out from the leaves to maintain dependencies.
        """
        candidates = []
        def _dfs(node: FastLibraNode):
            if not node.children and node.in_hbm and node.node_id != "VIRTUAL_ROOT":
                candidates.append(node)
            for child in node.children.values():
                _dfs(child)
        _dfs(self.root)
        return candidates

    def get_swap_in_candidates(self) -> List[FastLibraNode]:
        """
        Returns root-most nodes (LoRAs or highest KV caches) not in HBM.
        FastLibra swaps in starting from the root of each subtree.
        """
        candidates = []
        def _dfs(node: FastLibraNode):
            if not node.in_hbm:
                candidates.append(node)
                # Don't recurse down if parent isn't in HBM yet (maintain dependency)
            else:
                for child in node.children.values():
                    _dfs(child)
        _dfs(self.root)
        return candidates
