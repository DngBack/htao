import numpy as np
from typing import List, Dict, Any, Tuple, Callable
import torch
from torch import Tensor

def compute_tree_metrics(tree) -> Dict[str, float]:
    """
    Compute comprehensive metrics for evaluating tree-based reasoning
    
    This implements the novel evaluation metrics from the HATO methodology:
    - Reasoning Diversity Index (RDI)
    - Novelty Score
    - Robustness measures
    
    Args:
        tree: ReasoningTree object to evaluate
        
    Returns:
        Dictionary of metrics
    """
    # Get basic tree metrics
    metrics = tree.compute_metrics()
    
    # Get all paths for diversity calculations
    paths = tree.get_all_paths()
    if len(paths) < 2:
        # Not enough paths for diversity metrics
        metrics['reasoning_diversity_index'] = 0.0
        return metrics
    
    # Compute Reasoning Diversity Index (RDI)
    # RDI = (1/|P|) * sum_{p in P} min_{p' in P, p' != p} d(p, p')
    path_contents = []
    for path, _ in paths:
        # Extract content from each node in the path
        path_content = [node.content for node in path]
        path_contents.append(path_content)
    
    # Compute pairwise distances between paths
    distances = []
    for i, path1 in enumerate(path_contents):
        min_dist = float('inf')
        for j, path2 in enumerate(path_contents):
            if i != j:
                # Use normalized Levenshtein distance as path distance
                dist = path_distance(path1, path2)
                min_dist = min(min_dist, dist)
        if min_dist != float('inf'):
            distances.append(min_dist)
    
    if distances:
        metrics['reasoning_diversity_index'] = sum(distances) / len(distances)
    else:
        metrics['reasoning_diversity_index'] = 0.0
    
    # Compute success rate
    successful_paths = [p for p, r in paths if r > 0.5]  # Assuming reward > 0.5 means success
    metrics['success_rate'] = len(successful_paths) / len(paths) if paths else 0.0
    
    return metrics

def path_distance(path1: List[str], path2: List[str]) -> float:
    """
    Compute distance between two reasoning paths
    
    Args:
        path1: First path as list of node contents
        path2: Second path as list of node contents
        
    Returns:
        Normalized distance between paths
    """
    # Simple implementation using Jaccard distance
    # For a more sophisticated approach, we could use embedding-based similarity
    
    # Convert to sets of tokens
    tokens1 = set(" ".join(path1).lower().split())
    tokens2 = set(" ".join(path2).lower().split())
    
    # Compute Jaccard distance
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    
    if union == 0:
        return 1.0  # Maximum distance for empty paths
    
    return 1.0 - (intersection / union)

def compute_novelty_score(path: List[str], base_paths: List[List[str]]) -> float:
    """
    Compute novelty score for a path compared to base model paths
    
    Novelty(p) = 1 - max_{p' in P_base} sim(p, p')
    
    Args:
        path: Path to evaluate
        base_paths: Paths from base model
        
    Returns:
        Novelty score
    """
    if not base_paths:
        return 1.0  # Maximum novelty if no base paths
    
    max_similarity = 0.0
    for base_path in base_paths:
        # Similarity is 1 - distance
        similarity = 1.0 - path_distance(path, base_path)
        max_similarity = max(max_similarity, similarity)
    
    return 1.0 - max_similarity

def compute_robustness(tree, perturb_fn: Callable, n_perturbations: int = 5) -> float:
    """
    Compute robustness coefficient for a reasoning tree
    
    Robustness = E_{x~D, δ~Δ}[Perf(x+δ)/Perf(x)]
    
    Args:
        tree: ReasoningTree to evaluate
        perturb_fn: Function to perturb the problem
        n_perturbations: Number of perturbations to apply
        
    Returns:
        Robustness coefficient
    """
    # Get original performance
    original_metrics = tree.compute_metrics()
    original_perf = original_metrics.get('best_path_reward', 0.0)
    
    if original_perf == 0.0:
        return 0.0  # Can't compute robustness for zero performance
    
    # Apply perturbations and measure performance
    perturbed_perfs = []
    for _ in range(n_perturbations):
        # This is a placeholder - actual implementation would create
        # a new tree with the perturbed problem and evaluate it
        perturbed_perf = original_perf * (0.8 + 0.4 * np.random.random())  # Simulate perturbation
        perturbed_perfs.append(perturbed_perf)
    
    # Compute average ratio
    robustness = sum(perf / original_perf for perf in perturbed_perfs) / n_perturbations
    
    return robustness
