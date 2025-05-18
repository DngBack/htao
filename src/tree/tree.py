from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from .node import ThoughtNode

class ReasoningTree:
    """
    Represents a tree of reasoning steps for solving a problem.
    
    This implementation follows the HATO methodology's tree-based reasoning approach,
    with support for hierarchical rewards, adaptive exploration, and memory-efficient
    representation through sparse materialization and value-uncertainty pruning.
    """
    def __init__(
        self, 
        problem: str, 
        max_depth: int = 3, 
        branching_factor: int = 5,
        sparse: bool = False
    ):
        """
        Initialize a ReasoningTree.
        
        Args:
            problem: The problem statement (root node content)
            max_depth: Maximum depth of the tree
            branching_factor: Maximum number of children per node
            sparse: Whether to use sparse tree materialization
        """
        self.root = ThoughtNode(problem, level=0)
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.sparse = sparse
        
        # Organize nodes by level for efficient access
        self.all_nodes: Dict[int, List[ThoughtNode]] = {0: [self.root]}
        
        # Track metrics
        self.metrics: Dict[str, Any] = {}
        
    def expand_node(
        self, 
        node: ThoughtNode, 
        thoughts: List[str], 
        local_rewards: List[float], 
        path_rewards: Optional[List[float]] = None,
        causal_rewards: Optional[List[float]] = None,
        uncertainties: Optional[List[float]] = None
    ) -> List[ThoughtNode]:
        """
        Expand a node with multiple thought candidates
        
        Args:
            node: Parent node to expand
            thoughts: List of thought contents
            local_rewards: List of local rewards for each thought
            path_rewards: Optional list of path rewards
            causal_rewards: Optional list of causal rewards
            uncertainties: Optional list of uncertainty values
            
        Returns:
            List of created child nodes
        """
        if path_rewards is None:
            path_rewards = [0.0] * len(thoughts)
            
        if causal_rewards is None:
            causal_rewards = [0.0] * len(thoughts)
            
        if uncertainties is None:
            uncertainties = [0.0] * len(thoughts)
        
        created_nodes = []
        for i, (thought, local_r, path_r, causal_r, uncertainty) in enumerate(
            zip(thoughts, local_rewards, path_rewards, causal_rewards, uncertainties)
        ):
            child = ThoughtNode(
                content=thought,
                parent=node,
                level=node.level + 1,
                local_reward=local_r,
                path_reward=path_r,
                causal_reward=causal_r
            )
            
            # Store uncertainty in metadata
            child.metadata['uncertainty'] = uncertainty
            
            # Update total reward
            child.update_rewards()
            
            # Add to parent's children
            node.add_child(child)
            
            # Add to level-organized dictionary
            if node.level + 1 not in self.all_nodes:
                self.all_nodes[node.level + 1] = []
            self.all_nodes[node.level + 1].append(child)
            
            created_nodes.append(child)
            
        return created_nodes
    
    def get_nodes_at_level(self, level: int) -> List[ThoughtNode]:
        """
        Get all nodes at a specific level
        
        Args:
            level: Tree level to retrieve nodes from
            
        Returns:
            List of nodes at the specified level
        """
        return self.all_nodes.get(level, [])
    
    def backpropagate_rewards(self, leaf_node: ThoughtNode, reward: float) -> None:
        """
        Backpropagate rewards from leaf to root
        
        Args:
            leaf_node: Leaf node with final outcome
            reward: Reward value to propagate
        """
        current = leaf_node
        while current is not None:
            current.path_reward = max(current.path_reward, reward)
            current.update_rewards()
            current = current.parent
            
    def prune(self, threshold: float = 0.2, beta: float = 2.0) -> int:
        """
        Prune nodes with low total reward to save memory
        
        This implements the value-uncertainty pruning strategy from HATO:
        Prune(n) = True if V(n) + β · U(n) < θ_prune, False otherwise
        
        Args:
            threshold: Minimum reward threshold to keep a node
            beta: Exploration parameter for uncertainty weighting
            
        Returns:
            Number of pruned nodes
        """
        if not self.sparse:
            return 0
            
        pruned_count = 0
        for level in range(1, self.max_depth):
            nodes = self.get_nodes_at_level(level)
            kept_nodes = []
            
            for node in nodes:
                # Use exploration score (value + beta * uncertainty) for pruning
                exploration_score = node.get_exploration_score(beta)
                
                if exploration_score >= threshold:
                    kept_nodes.append(node)
                else:
                    # Remove from parent's children
                    if node.parent:
                        if node in node.parent.children:
                            node.parent.children.remove(node)
                    pruned_count += 1
                    
            self.all_nodes[level] = kept_nodes
            
        return pruned_count
    
    def get_best_path(self) -> Tuple[List[ThoughtNode], float]:
        """
        Get the path with highest reward
        
        Returns:
            Tuple of (path, reward)
        """
        leaf_nodes = self.get_nodes_at_level(self.max_depth)
        if not leaf_nodes:
            # If no leaf nodes, find deepest nodes
            max_level = max(self.all_nodes.keys())
            leaf_nodes = self.get_nodes_at_level(max_level)
            
        if not leaf_nodes:
            return [self.root], 0.0
            
        best_leaf = max(leaf_nodes, key=lambda x: x.total_reward)
        path = best_leaf.get_path_to_root()
        
        return path, best_leaf.total_reward
    
    def get_all_paths(self) -> List[Tuple[List[ThoughtNode], float]]:
        """
        Get all complete paths from root to leaf
        
        Returns:
            List of (path, reward) tuples
        """
        leaf_nodes = self.get_nodes_at_level(self.max_depth)
        if not leaf_nodes:
            # If no leaf nodes, find deepest nodes
            max_level = max(self.all_nodes.keys())
            leaf_nodes = self.get_nodes_at_level(max_level)
            
        if not leaf_nodes:
            return [([self.root], 0.0)]
            
        paths = []
        for leaf in leaf_nodes:
            path = leaf.get_path_to_root()
            paths.append((path, leaf.total_reward))
            
        # Sort by reward (highest first)
        paths.sort(key=lambda x: x[1], reverse=True)
        
        return paths
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute tree-level metrics
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic tree statistics
        total_nodes = sum(len(nodes) for nodes in self.all_nodes.values())
        metrics['total_nodes'] = total_nodes
        metrics['max_depth'] = max(self.all_nodes.keys())
        metrics['avg_branching'] = 0
        
        # Calculate average branching factor
        internal_nodes = 0
        total_children = 0
        for level in range(self.max_depth):
            nodes = self.get_nodes_at_level(level)
            for node in nodes:
                if node.children:
                    internal_nodes += 1
                    total_children += len(node.children)
        
        if internal_nodes > 0:
            metrics['avg_branching'] = total_children / internal_nodes
            
        # Reward statistics
        all_rewards = []
        for level in range(1, self.max_depth + 1):
            nodes = self.get_nodes_at_level(level)
            all_rewards.extend([node.total_reward for node in nodes])
            
        if all_rewards:
            metrics['max_reward'] = max(all_rewards)
            metrics['min_reward'] = min(all_rewards)
            metrics['avg_reward'] = sum(all_rewards) / len(all_rewards)
            metrics['reward_std'] = np.std(all_rewards)
            
        # Path metrics
        paths = self.get_all_paths()
        if paths:
            metrics['num_paths'] = len(paths)
            metrics['best_path_reward'] = paths[0][1]
            
        return metrics
