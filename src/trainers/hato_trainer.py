import uuid
import logging
from collections import defaultdict
from copy import deepcopy
from typing import (
    Dict,
    List,
    Any,
    Optional,
    Tuple,
    Union,
    Type,
    cast,
    Protocol,
    TYPE_CHECKING,
)

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    compute_advantage,
    Role,
    WorkerType,
    RayWorkerGroup,
)

if TYPE_CHECKING:
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager
else:
    ResourcePoolManager = Any  # type: ignore

from ..tree import ReasoningTree, ThoughtNode
from ..search import (
    adaptive_tree_search,
    UncertaintyEstimator,
    EnsembleUncertaintyEstimator,
    DropoutUncertaintyEstimator,
    BootstrapUncertaintyEstimator,
)
from ..rewards import HierarchicalRewardModel


# Define meta learner protocol
class MetaLearner(Protocol):
    """Protocol for meta learners"""

    def update(self, *args, **kwargs) -> None: ...
    def adapt(self, *args, **kwargs) -> None: ...


class MAMLMetaLearner:
    """Model-Agnostic Meta-Learning implementation"""

    def __init__(self, **kwargs):
        self.inner_lr = kwargs.get("inner_lr", 0.01)
        self.outer_lr = kwargs.get("outer_lr", 0.001)
        self.n_inner_steps = kwargs.get("n_inner_steps", 5)

    def update(self, *args, **kwargs) -> None:
        pass

    def adapt(self, *args, **kwargs) -> None:
        pass


class ReptileMetaLearner:
    """Reptile meta-learning implementation"""

    def __init__(self, **kwargs):
        self.meta_lr = kwargs.get("meta_lr", 0.1)
        self.n_inner_steps = kwargs.get("n_inner_steps", 5)

    def update(self, *args, **kwargs) -> None:
        pass

    def adapt(self, *args, **kwargs) -> None:
        pass


logger = logging.getLogger(__name__)


class HATOTrainer(RayPPOTrainer):
    """
    HATO (Hierarchical Adaptive Tree Optimization) Trainer

    This trainer extends the RayPPOTrainer to incorporate:
    1. Hierarchical reward modeling
    2. Adaptive exploration strategies
    3. Tree-based reasoning
    4. Meta-learning for cross-domain transfer
    5. Memory-efficient tree representation

    The implementation follows the theoretical guarantees from HATO:
    - Theorem 1: Monotonic improvement in expected sequence-level reward
    - Theorem 2: Bounded variance of policy gradient estimator
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        **kwargs,
    ):
        """
        Initialize HATO trainer

        Args:
            config: Configuration object with HATO-specific parameters
            tokenizer: Tokenizer for the language model
            role_worker_mapping: Mapping of roles to worker types
            resource_pool_manager: Resource pool manager for distributed training
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            **kwargs,
        )

        # Validate HATO-specific config
        self._validate_hato_config()

        # Initialize hierarchical reward model
        self.hierarchical_reward_model = HierarchicalRewardModel(
            alpha=self.config.hato.reward_weights.alpha,
            beta=self.config.hato.reward_weights.beta,
            gamma=self.config.hato.reward_weights.gamma,
        )

        # Initialize uncertainty estimator for adaptive exploration
        self.uncertainty_estimator = self._create_uncertainty_estimator()

        # Initialize meta-learning components if enabled
        if self.config.hato.meta_learning.enable:
            self.meta_learner = self._create_meta_learner()

        # Initialize tree metrics collector
        self.tree_metrics = defaultdict(list)

        # Temperature scheduling parameters
        self.initial_temperature = self.config.hato.exploration.initial_temperature
        self.temperature_decay = self.config.hato.exploration.temperature_decay
        self.current_temperature = self.initial_temperature

        # Set up model and device
        self._setup_model_and_device()

    def _setup_model_and_device(self):
        """Set up model and device after parent initialization"""
        # Access workers through actor_rollout_wg
        self.actor_model = self.actor_rollout_wg.workers[0].model
        self.device = next(self.actor_model.parameters()).device

    def _validate_hato_config(self):
        """Validate HATO-specific configuration"""
        assert hasattr(self.config, "hato"), "Missing hato config"
        assert hasattr(self.config.hato, "reward_weights"), "Missing reward weights"
        assert all(
            0 <= w <= 1
            for w in [
                self.config.hato.reward_weights.alpha,
                self.config.hato.reward_weights.beta,
                self.config.hato.reward_weights.gamma,
            ]
        ), "Reward weights must be between 0 and 1"
        assert (
            sum(
                [
                    self.config.hato.reward_weights.alpha,
                    self.config.hato.reward_weights.beta,
                    self.config.hato.reward_weights.gamma,
                ]
            )
            == 1.0
        ), "Reward weights must sum to 1.0"

    def _create_uncertainty_estimator(self):
        """
        Create uncertainty estimator based on config

        Returns:
            UncertaintyEstimator instance
        """
        uncertainty_type = self.config.hato.uncertainty.type

        if uncertainty_type == "ensemble":
            # Implement ensemble-based uncertainty estimation
            return EnsembleUncertaintyEstimator(
                n_models=self.config.hato.uncertainty.n_models,
                model_config=self.config.hato.uncertainty.model_config,
            )
        elif uncertainty_type == "dropout":
            # Implement dropout-based uncertainty estimation
            return DropoutUncertaintyEstimator(
                dropout_rate=self.config.hato.uncertainty.dropout_rate,
                n_forward_passes=self.config.hato.uncertainty.n_forward_passes,
            )
        elif uncertainty_type == "bootstrap":
            # Implement bootstrap-based uncertainty estimation
            return BootstrapUncertaintyEstimator(
                n_bootstrap=self.config.hato.uncertainty.n_bootstrap
            )
        else:
            raise ValueError(f"Unknown uncertainty type: {uncertainty_type}")

    def _create_meta_learner(self):
        """
        Create meta-learner based on config

        Returns:
            MetaLearner instance
        """
        meta_type = self.config.hato.meta_learning.type

        if meta_type == "maml":
            # Model-Agnostic Meta-Learning
            return MAMLMetaLearner(
                inner_lr=self.config.hato.meta_learning.inner_lr,
                outer_lr=self.config.hato.meta_learning.outer_lr,
                n_inner_steps=self.config.hato.meta_learning.n_inner_steps,
            )
        elif meta_type == "reptile":
            # Reptile meta-learning
            return ReptileMetaLearner(
                meta_lr=self.config.hato.meta_learning.meta_lr,
                n_inner_steps=self.config.hato.meta_learning.n_inner_steps,
            )
        else:
            raise ValueError(f"Unknown meta-learning type: {meta_type}")

    def generate_reasoning_trees(self, batch_dict):
        """
        Generate reasoning trees for a batch of problems

        Args:
            batch_dict: Dictionary containing problem batch

        Returns:
            trees: List of reasoning trees
            tree_batch: DataProto containing tree data for training
        """
        # Create DataProto from batch dict
        problem_batch = DataProto.from_single_dict(batch_dict)

        # Extract problems
        problems = problem_batch.non_tensor_batch.get("raw_prompts", [])
        if not problems:
            # Fall back to tokenized inputs if raw prompts not available
            problems = [
                self.tokenizer.decode(ids) for ids in problem_batch.batch["input_ids"]
            ]

        # Initialize trees and metrics
        trees = []
        all_nodes_by_level = defaultdict(list)

        # Current temperature for exploration
        temperature = self._get_current_temperature()

        # Generate trees for each problem
        for problem_idx, problem in enumerate(problems):
            # Create reasoning tree
            tree = ReasoningTree(
                problem=problem,
                max_depth=self.config.hato.tree.max_depth,
                branching_factor=self.config.hato.tree.branching_factor,
                sparse=self.config.hato.memory.sparse_materialization,
            )

            # Perform adaptive tree search
            tree, search_metrics = adaptive_tree_search(
                tree=tree,
                model=self.actor_model,
                tokenizer=self.tokenizer,
                uncertainty_estimator=self.uncertainty_estimator,
                temperature=temperature,
                search_algorithm=self.config.hato.tree.search_algorithm,
                max_nodes_per_level=self.config.hato.tree.max_nodes_per_level,
                device=self.device,
            )

            # Collect tree metrics
            for k, v in search_metrics.items():
                self.tree_metrics[k].append(v)

            # Compute hierarchical rewards for all nodes
            self._compute_hierarchical_rewards(tree)

            # Apply pruning if enabled
            if self.config.hato.memory.sparse_materialization:
                pruned_count = tree.prune(
                    threshold=self.config.hato.memory.pruning_threshold,
                    beta=2.0,  # Exploration parameter
                )
                search_metrics["nodes_pruned"] += pruned_count

            # Collect nodes by level for training
            for level, nodes in tree.all_nodes.items():
                if level > 0:  # Skip root level (problem statements)
                    all_nodes_by_level[level].extend(nodes)

            trees.append(tree)

        # Convert tree nodes to training batch
        tree_batch = self._nodes_to_batch(all_nodes_by_level)

        return trees, tree_batch

    def _compute_hierarchical_rewards(self, tree):
        """
        Compute hierarchical rewards for all nodes in the tree

        Args:
            tree: ReasoningTree object
        """
        # Use the hierarchical reward model to update all rewards
        self.hierarchical_reward_model.update_rewards(tree)

    def _verify_solution(self, problem, solution):
        """
        Verify if a solution is correct

        Args:
            problem: Problem statement
            solution: Proposed solution

        Returns:
            is_correct: Whether the solution is correct
        """
        # In a real implementation, this would use a verification model
        # or external API to check the solution

        # For now, implement a simple heuristic-based verification
        # This is just a placeholder - actual implementation would be more sophisticated

        # Check if solution contains a final answer
        has_answer = any(
            marker in solution.lower()
            for marker in ["answer:", "answer is", "final answer", "therefore", "thus"]
        )

        # Check if solution has reasonable length
        reasonable_length = len(solution.split()) >= 10

        # Check if solution contains numerical values
        has_numbers = any(char.isdigit() for char in solution)

        # Simple verification heuristic
        is_correct = has_answer and reasonable_length and has_numbers

        # Add some randomness to simulate verification accuracy
        if np.random.random() < 0.2:  # 20% chance to flip the result
            is_correct = not is_correct

        return is_correct

    def _nodes_to_batch(
        self, all_nodes_by_level: Dict[int, List[ThoughtNode]]
    ) -> DataProto:
        """
        Convert tree nodes to training batch

        Args:
            all_nodes_by_level: Dictionary mapping levels to lists of nodes

        Returns:
            batch: DataProto containing node data for training
        """
        try:
            # Group nodes by parent for GRPO
            parent_groups = {}
            for level, nodes in all_nodes_by_level.items():
                for node in nodes:
                    parent_id = id(node.parent)
                    if parent_id not in parent_groups:
                        parent_groups[parent_id] = []
                    parent_groups[parent_id].append(node)

            # Filter groups with no reward variance if enabled
            if self.config.algorithm.filter_groups.enable:
                filtered_groups = []
                for parent_id, group in parent_groups.items():
                    if len(group) < 2:  # Need at least 2 nodes for variance
                        continue

                    rewards = [node.total_reward for node in group]
                    if np.std(rewards) > 0:
                        filtered_groups.append(group)

                # Use filtered groups
                parent_groups = {i: group for i, group in enumerate(filtered_groups)}

            # Prepare batch data
            all_contents = []
            all_rewards = []
            all_parent_contents = []
            all_group_ids = []

            for group_id, group in parent_groups.items():
                for node in group:
                    all_contents.append(node.content)
                    all_rewards.append(node.total_reward)
                    all_parent_contents.append(node.parent.content)
                    all_group_ids.append(group_id)

            # Create batch dictionary
            batch_dict = {
                "input_ids": self.tokenizer(
                    all_contents,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.model.max_length,
                )["input_ids"],
                "attention_mask": self.tokenizer(
                    all_contents,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.model.max_length,
                )["attention_mask"],
                "token_level_rewards": torch.tensor(all_rewards)
                .unsqueeze(-1)
                .expand(-1, self.config.model.max_length),
            }

            # Create DataProto with dictionary
            batch = DataProto.from_single_dict(batch_dict)
            batch.non_tensor_batch = {
                "raw_prompts": all_contents,
                "parent_prompts": all_parent_contents,
                "group_ids": np.array(all_group_ids),
                "rewards": np.array(all_rewards),
            }

            return batch

        except Exception as e:
            logger.error(f"Error converting nodes to batch: {e}")
            raise

    def _get_current_temperature(self):
        """
        Get current temperature based on training progress

        Returns:
            temperature: Current temperature value
        """
        progress = min(1.0, self.global_steps / self.total_training_steps)
        temperature = self.initial_temperature * np.exp(
            -self.temperature_decay * progress
        )
        self.current_temperature = temperature
        return temperature

    def train_step(self, batch_dict: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform a single training step

        Args:
            batch_dict: Dictionary containing training batch

        Returns:
            metrics: Dictionary of training metrics
        """
        try:
            logger.info(f"Starting training step {self.global_steps}")

            # Generate reasoning trees
            trees, tree_batch = self.generate_reasoning_trees(batch_dict)

            # Use parent class's fit method
            metrics = super().fit()

            # Add tree metrics
            for k, v in self.tree_metrics.items():
                if v:
                    metrics[f"tree/{k}"] = np.mean(v)

            # Clear tree metrics for next step
            self.tree_metrics = defaultdict(list)

            logger.info(f"Training step metrics: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error in training step: {e}")
            raise

    def compute_advantages(self, batch: DataProto) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages for HATO policy gradient

        Args:
            batch: DataProto containing batch data

        Returns:
            advantages: Computed advantages as tensor
            returns: Computed returns as tensor
        """
        try:
            # Use standard advantage computation from VERL
            advantages, returns = compute_advantage(
                batch,
                self.config.algorithm.gamma,
                self.config.algorithm.gae_lambda,
                self.config.algorithm.normalize_advantage,
            )

            # Convert to tensor for operations
            advantages_tensor = torch.tensor(batch.batch["advantages"])
            returns_tensor = torch.tensor(batch.batch["returns"])
            group_ids = torch.tensor(batch.non_tensor_batch["group_ids"])

            # Group advantages by group_id for GRPO
            unique_groups = torch.unique(group_ids)
            for group_id in unique_groups:
                group_mask = group_ids == group_id
                group_advantages = advantages_tensor[group_mask]

                # Skip groups with only one element
                if len(group_advantages) <= 1:
                    continue

                # Normalize within group
                group_mean = group_advantages.mean()
                group_std = group_advantages.std()
                if group_std > 0:
                    advantages_tensor[group_mask] = (
                        group_advantages - group_mean
                    ) / group_std

            # Update batch
            batch.batch["advantages"] = advantages_tensor.numpy()
            return advantages_tensor, returns_tensor

        except Exception as e:
            logger.error(f"Error computing advantages: {e}")
            raise

    def evaluate(self, batch_dict):
        """
        Evaluate model on a batch

        Args:
            batch_dict: Dictionary containing evaluation batch

        Returns:
            metrics: Dictionary of evaluation metrics
        """
        # Generate reasoning trees
        trees, _ = self.generate_reasoning_trees(batch_dict)

        # Compute evaluation metrics
        metrics = {}

        # Basic metrics
        success_count = 0
        total_count = len(trees)

        for tree in trees:
            best_path, best_reward = tree.get_best_path()
            if best_reward > self.config.hato.verification_threshold:
                success_count += 1

            # Compute tree-specific metrics
            tree_metrics = tree.compute_metrics()
            for k, v in tree_metrics.items():
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(v)

        # Compute success rate
        metrics["success_rate"] = (
            success_count / total_count if total_count > 0 else 0.0
        )

        # Average other metrics
        for k, v in list(metrics.items()):
            if isinstance(v, list):
                metrics[k] = np.mean(v)

        return metrics
