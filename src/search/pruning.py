import numpy as np
from typing import List, Dict, Any, Tuple
from ..tree import ReasoningTree, ThoughtNode

def value_uncertainty_pruning(
    tree: ReasoningTree,
    threshold: float = 0.2,
    beta: float = 2.0
) -> int:
    """
    Prune nodes with low exploration score to save memory
    
    This implements the value-uncertainty pruning strategy from HATO:
    Prune(n) = True if V(n) + β · U(n) < θ_prune, False otherwise
    
    Args:
        tree: ReasoningTree to prune
        threshold: Minimum exploration score threshold to keep a node
        beta: Exploration parameter for uncertainty weighting
        
    Returns:
        Number of pruned nodes
    """
    if not tree.sparse:
        return 0
        
    pruned_count = 0
    for level in range(1, tree.max_depth):
        nodes = tree.get_nodes_at_level(level)
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
                
        tree.all_nodes[level] = kept_nodes
        
    return pruned_count

def sparse_tree_materialization(
    tree: ReasoningTree,
    threshold: float = 0.2,
    beta: float = 2.0
) -> ReasoningTree:
    """
    Create a sparse materialization of the tree
    
    This implements the sparse tree materialization from HATO:
    T_sparse = {n ∈ T | V(n) + β · U(n) > θ_prune}
    
    Args:
        tree: Original ReasoningTree
        threshold: Minimum exploration score threshold to keep a node
        beta: Exploration parameter for uncertainty weighting
        
    Returns:
        Sparse tree
    """
    # Create a new sparse tree with the same root
    sparse_tree = ReasoningTree(
        problem=tree.root.content,
        max_depth=tree.max_depth,
        branching_factor=tree.branching_factor,
        sparse=True
    )
    
    # Copy nodes that meet the materialization criterion
    for level in range(1, tree.max_depth + 1):
        nodes = tree.get_nodes_at_level(level)
        for node in nodes:
            # Calculate exploration score
            exploration_score = node.get_exploration_score(beta)
            
            # Only materialize nodes with high exploration score
            if exploration_score > threshold:
                # Find parent in sparse tree
                parent_path = node.get_path_to_root()[:-1]  # Exclude the node itself
                if not parent_path:
                    # Should not happen for level > 0
                    continue
                    
                parent_in_sparse = sparse_tree.root
                for i in range(1, len(parent_path)):
                    # Find matching child in sparse tree
                    parent_content = parent_path[i].content
                    found = False
                    for child in parent_in_sparse.children:
                        if child.content == parent_content:
                            parent_in_sparse = child
                            found = True
                            break
                    
                    if not found:
                        # Parent path not materialized, skip this node
                        break
                else:
                    # Parent path fully materialized, add this node
                    sparse_tree.expand_node(
                        parent_in_sparse,
                        [node.content],
                        [node.local_reward],
                        [node.path_reward],
                        [node.causal_reward],
                        [node.get_uncertainty()]
                    )
    
    return sparse_tree
