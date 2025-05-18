import numpy as np
from typing import List


class CausalRewardModel:
    """
    Model for computing causal rewards based on counterfactual contribution

    This implements the causal credit assignment mechanism from HATO:
    R_causal(n) = E_{p ∈ P_n} [R(p) - R(p \ n)]

    Where:
    - P_n is the set of paths containing node n
    - R(p) is the reward for path p
    - R(p \ n) is the estimated reward if node n were replaced with an alternative
    """

    def __init__(self):
        """Initialize causal reward model"""
        pass

    def compute_reward(self, node, successful_paths: List[List], tree) -> float:
        """
        Compute causal reward for a node based on counterfactual contribution

        Args:
            node: Node to compute causal reward for
            successful_paths: List of successful paths
            tree: ReasoningTree containing the node

        Returns:
            Causal reward value
        """
        # Find paths containing this node
        paths_with_node = []
        for path in successful_paths:
            if node in path:
                paths_with_node.append(path)

        if not paths_with_node:
            return 0.0

        # Estimate counterfactual by comparing with alternative nodes
        siblings = [
            n
            for n in tree.get_nodes_at_level(node.level)
            if n.parent == node.parent and n != node
        ]

        if not siblings:
            return node.path_reward  # No alternatives to compare with

        # Average reward of paths with this node
        avg_reward_with_node = np.mean(
            [path[-1].path_reward for path in paths_with_node]
        )

        # Estimate reward if node were replaced with siblings
        sibling_rewards = []
        for sibling in siblings:
            # Find paths containing sibling
            paths_with_sibling = []
            for path in successful_paths:
                if sibling in path:
                    paths_with_sibling.append(path)

            if paths_with_sibling:
                avg_reward_with_sibling = np.mean(
                    [path[-1].path_reward for path in paths_with_sibling]
                )
                sibling_rewards.append(avg_reward_with_sibling)

        # If no siblings in successful paths, assume they would have zero reward
        avg_reward_without_node = np.mean(sibling_rewards) if sibling_rewards else 0.0

        # Causal contribution is the difference
        causal_reward = float(avg_reward_with_node - avg_reward_without_node)
        return max(0.0, causal_reward)  # Ensure non-negative

    def compute_counterfactual_reward(
        self, node, path: List, alternative_nodes: List
    ) -> float:
        """
        Compute counterfactual reward by estimating path reward with alternative nodes

        Args:
            node: Node to compute counterfactual for
            path: Path containing the node
            alternative_nodes: Alternative nodes that could replace this node

        Returns:
            Counterfactual reward difference
        """
        # Get the reward of the original path
        original_reward = path[-1].path_reward

        # Estimate rewards with alternative nodes
        alternative_rewards = []
        for alt_node in alternative_nodes:
            # In a real implementation, this would use a model to estimate
            # the reward of the path if this node were replaced
            # For now, use a simple heuristic based on the alternative node's reward
            estimated_reward = alt_node.total_reward * original_reward
            alternative_rewards.append(estimated_reward)

        # Average estimated reward with alternatives
        if alternative_rewards:
            avg_alternative_reward = np.mean(alternative_rewards)
        else:
            avg_alternative_reward = 0.0

        # Counterfactual contribution
        counterfactual_reward = original_reward - avg_alternative_reward

        return max(0.0, counterfactual_reward)  # Ensure non-negative
