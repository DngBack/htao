import torch
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Callable
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..tree import ReasoningTree, ThoughtNode
from .uncertainty import UncertaintyEstimator

def adaptive_tree_search(
    tree: ReasoningTree,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    uncertainty_estimator: UncertaintyEstimator,
    temperature: float = 0.7,
    search_algorithm: str = "beam",
    max_nodes_per_level: int = 5,
    device: str = "cuda"
) -> Tuple[ReasoningTree, Dict[str, float]]:
    """
    Perform adaptive tree search using the specified algorithm
    
    Args:
        tree: ReasoningTree to expand
        model: Language model for generating thoughts
        tokenizer: Tokenizer for the language model
        uncertainty_estimator: Uncertainty estimation module
        temperature: Sampling temperature
        search_algorithm: Search algorithm (bfs, dfs, beam)
        max_nodes_per_level: Maximum nodes to keep at each level
        device: Device to run model on
        
    Returns:
        Tuple of (expanded tree, search metrics)
    """
    metrics = {}
    
    if search_algorithm == "beam":
        return beam_search(
            tree, model, tokenizer, uncertainty_estimator, 
            temperature, max_nodes_per_level, device
        )
    elif search_algorithm == "bfs":
        return breadth_first_search(
            tree, model, tokenizer, uncertainty_estimator, 
            temperature, max_nodes_per_level, device
        )
    elif search_algorithm == "dfs":
        return depth_first_search(
            tree, model, tokenizer, uncertainty_estimator, 
            temperature, max_nodes_per_level, device
        )
    else:
        raise ValueError(f"Unknown search algorithm: {search_algorithm}")

def beam_search(
    tree: ReasoningTree,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    uncertainty_estimator: UncertaintyEstimator,
    temperature: float = 0.7,
    beam_width: int = 5,
    device: str = "cuda"
) -> Tuple[ReasoningTree, Dict[str, float]]:
    """
    Perform beam search to expand the reasoning tree
    
    This implements the adaptive exploration strategy from HATO with
    Thompson sampling at the node level:
    P(select n) ∝ exp((V(n) + β · U(n))/τ)
    
    Args:
        tree: ReasoningTree to expand
        model: Language model for generating thoughts
        tokenizer: Tokenizer for the language model
        uncertainty_estimator: Uncertainty estimation module
        temperature: Sampling temperature
        beam_width: Beam width (max nodes per level)
        device: Device to run model on
        
    Returns:
        Tuple of (expanded tree, search metrics)
    """
    metrics = {
        "nodes_generated": 0,
        "nodes_pruned": 0,
        "max_reward": 0.0,
        "avg_reward": 0.0,
    }
    
    # Start with the root node
    current_level = 0
    
    while current_level < tree.max_depth:
        nodes = tree.get_nodes_at_level(current_level)
        
        for node in nodes:
            # Generate candidate thoughts
            thoughts, local_rewards = generate_thoughts(
                node.content, 
                model, 
                tokenizer, 
                tree.branching_factor, 
                temperature,
                device
            )
            
            # Estimate uncertainty for each thought
            uncertainties = uncertainty_estimator.estimate(
                thoughts,
                model,
                tokenizer
            )
            
            # Expand node with thoughts, rewards, and uncertainties
            new_nodes = tree.expand_node(
                node, 
                thoughts, 
                local_rewards, 
                uncertainties=uncertainties
            )
            metrics["nodes_generated"] += len(new_nodes)
        
        # Move to next level
        current_level += 1
        
        # Beam search: keep only top-k nodes at each level
        if current_level < tree.max_depth:
            next_level_nodes = tree.get_nodes_at_level(current_level)
            
            if len(next_level_nodes) > beam_width:
                # Compute exploration scores (value + beta * uncertainty)
                beta = 2.0  # Exploration parameter
                exploration_scores = [
                    node.get_exploration_score(beta) for node in next_level_nodes
                ]
                
                # Sort nodes by exploration score
                sorted_indices = np.argsort(exploration_scores)[::-1]  # Descending
                
                # Keep only top-k nodes
                kept_indices = sorted_indices[:beam_width]
                kept_nodes = [next_level_nodes[i] for i in kept_indices]
                
                # Update tree with pruned nodes
                tree.all_nodes[current_level] = kept_nodes
                metrics["nodes_pruned"] += len(next_level_nodes) - len(kept_nodes)
    
    # Compute final metrics
    all_rewards = []
    for level in range(1, tree.max_depth + 1):
        nodes = tree.get_nodes_at_level(level)
        rewards = [node.total_reward for node in nodes]
        if rewards:
            all_rewards.extend(rewards)
    
    if all_rewards:
        metrics["max_reward"] = max(all_rewards)
        metrics["avg_reward"] = sum(all_rewards) / len(all_rewards)
    
    return tree, metrics

def breadth_first_search(
    tree: ReasoningTree,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    uncertainty_estimator: UncertaintyEstimator,
    temperature: float = 0.7,
    max_nodes_per_level: int = 5,
    device: str = "cuda"
) -> Tuple[ReasoningTree, Dict[str, float]]:
    """
    Perform breadth-first search to expand the reasoning tree
    
    Args:
        tree: ReasoningTree to expand
        model: Language model for generating thoughts
        tokenizer: Tokenizer for the language model
        uncertainty_estimator: Uncertainty estimation module
        temperature: Sampling temperature
        max_nodes_per_level: Maximum nodes to keep at each level
        device: Device to run model on
        
    Returns:
        Tuple of (expanded tree, search metrics)
    """
    metrics = {
        "nodes_generated": 0,
        "nodes_pruned": 0,
        "max_reward": 0.0,
        "avg_reward": 0.0,
    }
    
    # BFS: process level by level
    for current_level in range(tree.max_depth):
        nodes = tree.get_nodes_at_level(current_level)
        
        # Process all nodes at current level
        for node in nodes:
            # Generate candidate thoughts
            thoughts, local_rewards = generate_thoughts(
                node.content, 
                model, 
                tokenizer, 
                tree.branching_factor, 
                temperature,
                device
            )
            
            # Estimate uncertainty for each thought
            uncertainties = uncertainty_estimator.estimate(
                thoughts,
                model,
                tokenizer
            )
            
            # Expand node with thoughts, rewards, and uncertainties
            new_nodes = tree.expand_node(
                node, 
                thoughts, 
                local_rewards, 
                uncertainties=uncertainties
            )
            metrics["nodes_generated"] += len(new_nodes)
        
        # Prune if needed
        next_level = current_level + 1
        if next_level < tree.max_depth:
            next_level_nodes = tree.get_nodes_at_level(next_level)
            
            if len(next_level_nodes) > max_nodes_per_level:
                # Apply value-uncertainty pruning
                pruned_count = tree.prune(threshold=0.2, beta=2.0)
                metrics["nodes_pruned"] += pruned_count
    
    # Compute final metrics
    all_rewards = []
    for level in range(1, tree.max_depth + 1):
        nodes = tree.get_nodes_at_level(level)
        rewards = [node.total_reward for node in nodes]
        if rewards:
            all_rewards.extend(rewards)
    
    if all_rewards:
        metrics["max_reward"] = max(all_rewards)
        metrics["avg_reward"] = sum(all_rewards) / len(all_rewards)
    
    return tree, metrics

def depth_first_search(
    tree: ReasoningTree,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    uncertainty_estimator: UncertaintyEstimator,
    temperature: float = 0.7,
    max_nodes_per_level: int = 5,
    device: str = "cuda"
) -> Tuple[ReasoningTree, Dict[str, float]]:
    """
    Perform depth-first search to expand the reasoning tree
    
    Args:
        tree: ReasoningTree to expand
        model: Language model for generating thoughts
        tokenizer: Tokenizer for the language model
        uncertainty_estimator: Uncertainty estimation module
        temperature: Sampling temperature
        max_nodes_per_level: Maximum nodes to keep at each level
        device: Device to run model on
        
    Returns:
        Tuple of (expanded tree, search metrics)
    """
    metrics = {
        "nodes_generated": 0,
        "nodes_pruned": 0,
        "max_reward": 0.0,
        "avg_reward": 0.0,
    }
    
    # Use a stack for DFS
    stack = [tree.root]
    
    while stack:
        node = stack.pop()
        
        # Skip if we've reached max depth
        if node.level >= tree.max_depth:
            continue
        
        # Generate candidate thoughts
        thoughts, local_rewards = generate_thoughts(
            node.content, 
            model, 
            tokenizer, 
            tree.branching_factor, 
            temperature,
            device
        )
        
        # Estimate uncertainty for each thought
        uncertainties = uncertainty_estimator.estimate(
            thoughts,
            model,
            tokenizer
        )
        
        # Expand node with thoughts, rewards, and uncertainties
        new_nodes = tree.expand_node(
            node, 
            thoughts, 
            local_rewards, 
            uncertainties=uncertainties
        )
        metrics["nodes_generated"] += len(new_nodes)
        
        # Sort children by exploration score for DFS priority
        beta = 2.0  # Exploration parameter
        sorted_children = sorted(
            new_nodes, 
            key=lambda x: x.get_exploration_score(beta),
            reverse=True
        )
        
        # Limit children per node
        if len(sorted_children) > max_nodes_per_level:
            # Keep only top-k children
            kept_children = sorted_children[:max_nodes_per_level]
            
            # Remove pruned children from parent and tree
            for child in sorted_children[max_nodes_per_level:]:
                if child in node.children:
                    node.children.remove(child)
                if child in tree.all_nodes.get(child.level, []):
                    tree.all_nodes[child.level].remove(child)
                    
            metrics["nodes_pruned"] += len(sorted_children) - max_nodes_per_level
            sorted_children = kept_children
        
        # Add children to stack in reverse order (to process highest-scoring first)
        for child in reversed(sorted_children):
            stack.append(child)
    
    # Compute final metrics
    all_rewards = []
    for level in range(1, tree.max_depth + 1):
        nodes = tree.get_nodes_at_level(level)
        rewards = [node.total_reward for node in nodes]
        if rewards:
            all_rewards.extend(rewards)
    
    if all_rewards:
        metrics["max_reward"] = max(all_rewards)
        metrics["avg_reward"] = sum(all_rewards) / len(all_rewards)
    
    return tree, metrics

def generate_thoughts(
    content: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    num_thoughts: int = 5,
    temperature: float = 0.7,
    device: str = "cuda"
) -> Tuple[List[str], List[float]]:
    """
    Generate candidate thoughts using the language model
    
    Args:
        content: Input content to generate thoughts from
        model: Language model
        tokenizer: Tokenizer for the language model
        num_thoughts: Number of thoughts to generate
        temperature: Sampling temperature
        device: Device to run model on
        
    Returns:
        Tuple of (thoughts, local_rewards)
    """
    # Prepare prompt for thought generation
    prompt = f"Problem: {content}\n\nLet's think step by step to solve this problem:"
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    # Move tensors to device if they are torch tensors
    if hasattr(inputs, 'to'):
        inputs = inputs.to(device)
    elif isinstance(inputs, dict):
        for k, v in inputs.items():
            if hasattr(v, 'to'):
                inputs[k] = v.to(device)
    
    # Generate multiple outputs with different seeds
    thoughts = []
    for i in range(num_thoughts):
        # Set seed for reproducibility but different for each thought
        torch.manual_seed(42 + i)
        
        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                num_return_sequences=1,
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the generated thought (remove the prompt)
        thought = generated_text[len(prompt):].strip()
        thoughts.append(thought)
    
    # Compute simple local rewards (placeholder - would be replaced with actual reward model)
    # In a real implementation, this would use a trained reward model
    local_rewards = [0.5 + 0.5 * np.random.random() for _ in range(num_thoughts)]
    
    return thoughts, local_rewards
