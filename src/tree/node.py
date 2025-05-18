import torch
from typing import Optional, List, Dict, Any

class ThoughtNode:
    """
    Represents a node in the reasoning tree, corresponding to an intermediate reasoning step.
    
    This implementation follows the HATO methodology's hierarchical reward modeling framework,
    where each node has three reward components:
    - local_reward: Intrinsic quality of this reasoning step
    - path_reward: Extrinsic reward based on final outcome of paths containing this node
    - causal_reward: Causal contribution of this node to successful paths
    
    The total reward is a weighted combination of these components:
    R(n) = α · R_local(n) + β · R_path(n) + γ · R_causal(n)
    """
    def __init__(
        self, 
        content: str, 
        parent: Optional['ThoughtNode'] = None,
        level: int = 0,
        local_reward: float = 0.0,
        path_reward: float = 0.0,
        causal_reward: float = 0.0
    ):
        """
        Initialize a ThoughtNode.
        
        Args:
            content: The text content of this reasoning step
            parent: Parent node in the reasoning tree
            level: Level in the tree (0 for root)
            local_reward: Initial local reward value
            path_reward: Initial path reward value
            causal_reward: Initial causal reward value
        """
        self.content = content
        self.parent = parent
        self.children: List['ThoughtNode'] = []
        self.level = level
        
        # Reward components
        self.local_reward = local_reward    # Intrinsic quality of this step
        self.path_reward = path_reward      # Contribution to final outcome
        self.causal_reward = causal_reward  # Counterfactual contribution
        self.total_reward = 0.0             # Combined reward
        
        # Metadata for tracking and analysis
        self.metadata: Dict[str, Any] = {}
        
    def add_child(self, child_node: 'ThoughtNode') -> None:
        """Add a child node to this node"""
        self.children.append(child_node)
        
    def update_rewards(self, alpha: float = 0.4, beta: float = 0.5, gamma: float = 0.1) -> None:
        """
        Update total reward as weighted sum of reward components
        
        Args:
            alpha: Weight for local reward
            beta: Weight for path reward
            gamma: Weight for causal reward
        """
        self.total_reward = (
            alpha * self.local_reward + 
            beta * self.path_reward + 
            gamma * self.causal_reward
        )
        
    def get_path_to_root(self) -> List['ThoughtNode']:
        """
        Get the path from this node to the root
        
        Returns:
            List of nodes from root to this node
        """
        path = [self]
        current = self.parent
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))
    
    def get_uncertainty(self) -> float:
        """
        Get the uncertainty value from metadata if available
        
        Returns:
            Uncertainty value or 0.0 if not available
        """
        return self.metadata.get('uncertainty', 0.0)
    
    def get_exploration_score(self, beta: float = 2.0) -> float:
        """
        Calculate exploration score based on reward and uncertainty
        
        Args:
            beta: Exploration parameter controlling uncertainty weight
            
        Returns:
            Exploration score: reward + beta * uncertainty
        """
        uncertainty = self.get_uncertainty()
        return self.total_reward + beta * uncertainty
    
    def __repr__(self) -> str:
        """String representation of the node"""
        return f"ThoughtNode(level={self.level}, reward={self.total_reward:.3f}, content='{self.content[:30]}...')"
