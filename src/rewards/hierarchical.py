from typing import List, Optional

from .local import LocalRewardModel
from .causal import CausalRewardModel


class HierarchicalRewardModel:
    """
    Hierarchical reward model for tree-based reasoning

    This implements the hierarchical reward modeling framework from HATO:
    R(n) = α · R_local(n) + β · R_path(n) + γ · R_causal(n)

    Where:
    - R_local(n) is the intrinsic reward based on the quality of the reasoning step
    - R_path(n) is the extrinsic reward based on the final outcome of paths containing this node
    - R_causal(n) is the causal contribution of this node to successful paths
    """

    def __init__(
        self,
        alpha: float = 0.4,
        beta: float = 0.5,
        gamma: float = 0.1,
        local_reward_model: Optional["LocalRewardModel"] = None,
        causal_reward_model: Optional["CausalRewardModel"] = None,
    ):
        """
        Initialize hierarchical reward model

        Args:
            alpha: Weight for local reward
            beta: Weight for path reward
            gamma: Weight for causal reward
            local_reward_model: Model for computing local rewards
            causal_reward_model: Model for computing causal rewards
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Initialize reward models
        self.local_reward_model = local_reward_model or LocalRewardModel()
        self.causal_reward_model = causal_reward_model or CausalRewardModel()

    def compute_reward(
        self, local_reward: float, path_reward: float, causal_reward: float
    ) -> float:
        """
        Compute hierarchical reward as weighted sum of components

        Args:
            local_reward: Intrinsic quality of reasoning step
            path_reward: Extrinsic reward from final outcome
            causal_reward: Causal contribution to successful paths

        Returns:
            Combined hierarchical reward
        """
        return (
            self.alpha * local_reward
            + self.beta * path_reward
            + self.gamma * causal_reward
        )

    def compute_local_reward(
        self, parent_content: str, node_content: str, problem: str
    ) -> float:
        """
        Compute local reward for a reasoning step

        Args:
            parent_content: Content of parent node
            node_content: Content of current node
            problem: Original problem statement

        Returns:
            Local reward value
        """
        return self.local_reward_model.compute_reward(
            parent_content=parent_content, node_content=node_content, problem=problem
        )

    def compute_causal_reward(self, node, successful_paths: List[List], tree) -> float:
        """
        Compute causal reward for a node

        Args:
            node: Node to compute causal reward for
            successful_paths: List of successful paths
            tree: ReasoningTree containing the node

        Returns:
            Causal reward value
        """
        return self.causal_reward_model.compute_reward(
            node=node, successful_paths=successful_paths, tree=tree
        )

    def update_rewards(self, tree) -> None:
        """
        Update rewards for all nodes in a tree

        Args:
            tree: ReasoningTree to update rewards for
        """
        # First, compute local rewards for all nodes
        for level in range(1, tree.max_depth + 1):
            nodes = tree.get_nodes_at_level(level)
            for node in nodes:
                # Compute local reward based on coherence and progress
                node.local_reward = self.compute_local_reward(
                    parent_content=node.parent.content,
                    node_content=node.content,
                    problem=tree.root.content,
                )

        # Next, identify successful paths
        successful_paths = []
        leaf_nodes = tree.get_nodes_at_level(tree.max_depth)

        for leaf in leaf_nodes:
            if leaf.path_reward > 0:
                # Reconstruct path from leaf to root
                path = []
                current = leaf
                while current is not None:
                    path.append(current)
                    current = current.parent

                # Reverse to get root-to-leaf order
                path = list(reversed(path))
                successful_paths.append(path)

        # Finally, compute causal rewards if enabled
        if self.gamma > 0 and successful_paths:
            for level in range(1, tree.max_depth):
                nodes = tree.get_nodes_at_level(level)

                for node in nodes:
                    # Compute causal reward
                    node.causal_reward = self.compute_causal_reward(
                        node=node, successful_paths=successful_paths, tree=tree
                    )

                    # Update total reward
                    node.update_rewards(
                        alpha=self.alpha, beta=self.beta, gamma=self.gamma
                    )
