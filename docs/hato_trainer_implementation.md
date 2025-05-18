# HATO Trainer Implementation: Adapting DAPO for Hierarchical Adaptive Tree Optimization

This document provides a detailed implementation of the HATO trainer, adapting the DAPO trainer from VERL to incorporate hierarchical rewards, adaptive exploration, and tree-based reasoning. The implementation maintains the technical rigor and efficiency of the original DAPO trainer while integrating the novel components of the HATO framework.

## 1. Core Trainer Class Structure

```python
# hato_ray_trainer.py
import uuid
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Any, Optional, Tuple, Union

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
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, RayPPOTrainer, _timer, apply_kl_penalty, compute_advantage

from .tree import ReasoningTree, ThoughtNode
from .search import adaptive_tree_search
from .rewards import HierarchicalRewardModel


class RayHATOTrainer(RayPPOTrainer):
    """
    HATO (Hierarchical Adaptive Tree Optimization) Trainer
    
    This trainer extends the RayPPOTrainer to incorporate:
    1. Hierarchical reward modeling
    2. Adaptive exploration strategies
    3. Tree-based reasoning
    4. Meta-learning for cross-domain transfer
    5. Memory-efficient tree representation
    """

    def __init__(self, config):
        super().__init__(config)
        
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
        
    def _create_uncertainty_estimator(self):
        """Create uncertainty estimator based on config"""
        uncertainty_type = self.config.hato.uncertainty.type
        
        if uncertainty_type == "ensemble":
            # Implement ensemble-based uncertainty estimation
            return EnsembleUncertaintyEstimator(
                n_models=self.config.hato.uncertainty.n_models,
                model_config=self.config.hato.uncertainty.model_config
            )
        elif uncertainty_type == "dropout":
            # Implement dropout-based uncertainty estimation
            return DropoutUncertaintyEstimator(
                dropout_rate=self.config.hato.uncertainty.dropout_rate,
                n_forward_passes=self.config.hato.uncertainty.n_forward_passes
            )
        elif uncertainty_type == "bootstrap":
            # Implement bootstrap-based uncertainty estimation
            return BootstrapUncertaintyEstimator(
                n_bootstrap=self.config.hato.uncertainty.n_bootstrap
            )
        else:
            raise ValueError(f"Unknown uncertainty type: {uncertainty_type}")
            
    def _create_meta_learner(self):
        """Create meta-learner based on config"""
        meta_type = self.config.hato.meta_learning.type
        
        if meta_type == "maml":
            # Model-Agnostic Meta-Learning
            return MAMLMetaLearner(
                inner_lr=self.config.hato.meta_learning.inner_lr,
                outer_lr=self.config.hato.meta_learning.outer_lr,
                n_inner_steps=self.config.hato.meta_learning.n_inner_steps
            )
        elif meta_type == "reptile":
            # Reptile meta-learning
            return ReptileMetaLearner(
                meta_lr=self.config.hato.meta_learning.meta_lr,
                n_inner_steps=self.config.hato.meta_learning.n_inner_steps
            )
        else:
            raise ValueError(f"Unknown meta-learning type: {meta_type}")
```

## 2. Tree Generation and Processing

```python
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
            problems = [self.tokenizer.decode(ids) for ids in problem_batch.batch["input_ids"]]
        
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
                branching_factor=self.config.hato.tree.branching_factor
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
                device=self.device
            )
            
            # Collect tree metrics
            for k, v in search_metrics.items():
                self.tree_metrics[k].append(v)
            
            # Compute hierarchical rewards for all nodes
            self._compute_hierarchical_rewards(tree)
            
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
        # First, compute local rewards for all nodes
        for level in range(1, tree.max_depth + 1):
            nodes = tree.get_nodes_at_level(level)
            for node in nodes:
                # Compute local reward based on coherence and progress
                node.local_reward = self.hierarchical_reward_model.compute_local_reward(
                    parent_content=node.parent.content,
                    node_content=node.content,
                    problem=tree.root.content
                )
        
        # Next, compute path rewards for leaf nodes
        leaf_nodes = tree.get_nodes_at_level(tree.max_depth)
        for leaf in leaf_nodes:
            # Verify if the leaf node contains a correct solution
            is_correct = self._verify_solution(tree.root.content, leaf.content)
            
            # Assign path reward based on correctness
            path_reward = 1.0 if is_correct else 0.0
            
            # Backpropagate path reward
            tree.backpropagate_rewards(leaf, path_reward)
        
        # Finally, compute causal rewards if enabled
        if self.config.hato.reward_weights.gamma > 0:
            self._compute_causal_rewards(tree)
            
    def _compute_causal_rewards(self, tree):
        """
        Compute causal rewards for nodes based on counterfactual contribution
        
        Args:
            tree: ReasoningTree object
        """
        # Identify successful paths
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
        
        # For each node in successful paths, estimate counterfactual contribution
        for level in range(1, tree.max_depth):
            nodes = tree.get_nodes_at_level(level)
            
            for node in nodes:
                # Skip nodes not in any successful path
                in_successful_path = False
                for path in successful_paths:
                    if node in path:
                        in_successful_path = True
                        break
                        
                if not in_successful_path:
                    continue
                
                # Estimate counterfactual value by replacing this node
                causal_reward = self._estimate_counterfactual_contribution(tree, node, successful_paths)
                node.causal_reward = causal_reward
                
                # Update total reward
                node.update_rewards(
                    alpha=self.config.hato.reward_weights.alpha,
                    beta=self.config.hato.reward_weights.beta,
                    gamma=self.config.hato.reward_weights.gamma
                )
                
    def _estimate_counterfactual_contribution(self, tree, node, successful_paths):
        """
        Estimate counterfactual contribution of a node to successful paths
        
        Args:
            tree: ReasoningTree object
            node: ThoughtNode to evaluate
            successful_paths: List of successful paths
            
        Returns:
            causal_reward: Estimated causal contribution
        """
        # Find paths containing this node
        paths_with_node = []
        for path in successful_paths:
            if node in path:
                paths_with_node.append(path)
                
        if not paths_with_node:
            return 0.0
            
        # Estimate counterfactual by comparing with alternative nodes
        siblings = [n for n in tree.get_nodes_at_level(node.level) if n.parent == node.parent and n != node]
        
        if not siblings:
            return node.path_reward  # No alternatives to compare with
            
        # Average reward of paths with this node
        avg_reward_with_node = np.mean([path[-1].path_reward for path in paths_with_node])
        
        # Estimate reward if node were replaced with siblings
        sibling_rewards = []
        for sibling in siblings:
            # Find paths containing sibling
            paths_with_sibling = []
            for path in successful_paths:
                if sibling in path:
                    paths_with_sibling.append(path)
                    
            if paths_with_sibling:
                avg_reward_with_sibling = np.mean([path[-1].path_reward for path in paths_with_sibling])
                sibling_rewards.append(avg_reward_with_sibling)
                
        # If no siblings in successful paths, assume they would have zero reward
        avg_reward_without_node = np.mean(sibling_rewards) if sibling_rewards else 0.0
        
        # Causal contribution is the difference
        causal_reward = avg_reward_with_node - avg_reward_without_node
        
        return max(0.0, causal_reward)  # Ensure non-negative
```

## 3. Batch Processing and Training

```python
    def _nodes_to_batch(self, all_nodes_by_level):
        """
        Convert tree nodes to training batch
        
        Args:
            all_nodes_by_level: Dictionary mapping levels to lists of nodes
            
        Returns:
            batch: DataProto containing node data for training
        """
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
            "contents": all_contents,
            "rewards": all_rewards,
            "parent_contents": all_parent_contents,
            "group_ids": all_group_ids
        }
        
        # Tokenize contents
        tokenized_inputs = self.tokenizer(
            all_contents,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.model.max_length
        )
        
        # Create DataProto
        batch = DataProto()
        batch.batch = {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "token_level_rewards": torch.tensor(all_rewards).unsqueeze(-1).expand(-1, tokenized_inputs["input_ids"].shape[1])
        }
        
        batch.non_tensor_batch = {
            "raw_prompts": all_contents,
            "parent_prompts": all_parent_contents,
            "group_ids": np.array(all_group_ids),
            "rewards": np.array(all_rewards)
        }
        
        return batch
        
    def _get_current_temperature(self):
        """Get current temperature based on training progress"""
        progress = min(1.0, self.global_steps / self.total_training_steps)
        temperature = self.initial_temperature * np.exp(-self.temperature_decay * progress)
        self.current_temperature = temperature
        return temperature
        
    def _verify_solution(self, problem, solution):
        """
        Verify if a solution is correct for a given problem
        
        Args:
            problem: Problem statement
            solution: Proposed solution
            
        Returns:
            is_correct: Boolean indicating correctness
        """
        # Use reward function to verify solution
        combined_text = f"Problem: {problem}\n\nSolution: {solution}"
        inputs = self.tokenizer(combined_text, return_tensors="pt")
        
        with torch.no_grad():
            if self.use_rm:
                # Use reward model if available
                reward = self.rm_wg.compute_rm_score(inputs)
                is_correct = reward.item() > self.config.hato.verification_threshold
            else:
                # Use reward function
                reward_result = self.reward_fn(inputs, return_dict=True)
                is_correct = reward_result["reward_tensor"].sum().item() > self.config.hato.verification_threshold
                
        return is_correct
```

## 4. Main Training Loop

```python
    def fit(self):
        """
        The training loop of HATO.
        Extends the PPO training loop with tree-based reasoning and hierarchical rewards.
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # Load checkpoint before doing anything
        self._load_checkpoint()

        # Perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            print(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # Add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # Start from step 1
        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        num_trees_in_batch = 0
        num_gen_batches = 0
        
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # Generate reasoning trees
                    with _timer("tree_generation", timing_raw):
                        trees, tree_batch = self.generate_reasoning_trees(batch_dict)
                        num_trees_in_batch += len(trees)
                        num_gen_batches += 1
                    
                    # Apply meta-learning if enabled
                    if self.config.hato.meta_learning.enable:
                        with _timer("meta_learning", timing_raw):
                            tree_batch = self.meta_learner.adapt(tree_batch)
                    
                    # Process batch for training
                    if not self.config.algorithm.filter_groups.enable:
                        batch = tree_batch
                    else:
                        # Check if we have enough trees
                        tree_bsz = self.config.data.train_batch_size
                        if num_trees_in_batch < tree_bsz:
                            print(f"{num_trees_in_batch=} < {tree_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}. Generated too many. "
                                    "Please check if your data are too difficult. "
                                    "You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            batch = tree_batch
                    
                    # Balance the number of valid tokens on each dp rank
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)
                    
                    # Compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    
                    # Compute old log probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)
                    
                    if self.use_reference_policy:
                        # Compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)
                    
                    # Compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)
                    
                    with _timer("adv", timing_raw):
                        # Compute advantages with group-based normalization
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        
                        # For HATO, we use group-based advantage computation
                        batch = self._compute_hierarchical_advantage(
                            batch,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo
                        )
                    
                    # Update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)
                    
                    # Implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # Update actor
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                    
                    # Validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)
                    
                    # Save checkpoint
                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()
                
                # Collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                
                # Add tree metrics
                for k, v in self.tree_metrics.items():
                    if v:
                        metrics[f"tree/{k}"] = np.mean(v)
                self.tree_metrics = defaultdict(list)  # Clear tree metrics
                
                # Add exploration metrics
                metrics["exploration/temperature"] = self.current_temperature
                
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # Clear timing
                
                metrics["train/num_gen_batches"] = num_gen_batches
                metrics["train/num_trees"] = num_trees_in_batch
                
                batch = None
                num_trees_in_batch = 0
                num_gen_batches = 0
                
                # Log metrics
                logger.log(data=metrics, step=self.global_steps)
                
                if is_last_step:
                    print(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
                
                progress_bar.update(1)
                self.global_steps += 1
```

## 5. Hierarchical Advantage Computation

```python
    def _compute_hierarchical_advantage(self, batch, gamma, lam, norm_adv_by_std_in_grpo=True):
        """
        Compute advantages with hierarchical structure and group-based normalization
        
        Args:
            batch: DataProto containing batch data
            gamma: Discount factor
            lam: GAE lambda parameter
            norm_adv_by_std_in_grpo: Whether to normalize advantages by std in each group
            
        Returns:
            batch: Updated batch with advantages
        """
        # Extract group IDs
        group_ids = batch.non_tensor_batch.get("group_ids", None)
        
        if group_ids is None or not norm_adv_by_std_in_grpo:
            # Fall back to standard advantage computation if no groups
            return compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=gamma,
                lam=lam,
                num_repeat=1,  # No repeats in HATO
                norm_adv_by_std_in_grpo=False
            )
        
        # Compute raw advantages first
        batch = compute_advantage(
            batch,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=gamma,
            lam=lam,
            num_repeat=1,
            norm_adv_by_std_in_grpo=False
        )
        
        # Extract raw advantages
        advantages = batch.batch["advantages"].clone()
        
        # Normalize advantages within each group
        unique_groups = np.unique(group_ids)
        normalized_advantages = torch.zeros_like(advantages)
        
        for group_id in unique_groups:
            group_mask = (group_ids == group_id)
            group_indices = np.where(group_mask)[0]
            
            if len(group_indices) > 1:  # Need at least 2 samples for normalization
                group_advantages = advantages[group_indices]
                group_mean = group_advantages.mean()
                group_std = group_advantages.std()
                
                if group_std > 0:
                    normalized_group_advantages = (group_advantages - group_mean) / (group_std + 1e-8)
                else:
                    normalized_group_advantages = torch.zeros_like(group_advantages)
                
                normalized_advantages[group_indices] = normalized_group_advantages
        
        # Update batch with normalized advantages
        batch.batch["advantages"] = normalized_advantages
        
        return batch
```

## 6. Configuration

```yaml
# config/hato_config.yaml
hato:
  # Tree configuration
  tree:
    max_depth: 3
    branching_factor: 5
    search_algorithm: "beam"  # Options: bfs, dfs, beam
    max_nodes_per_level: 5
  
  # Reward weights
  reward_weights:
    alpha: 0.4  # Weight for local reward
    beta: 0.5   # Weight for path reward
    gamma: 0.1  # Weight for causal reward
  
  # Exploration parameters
  exploration:
    initial_temperature: 0.7
    temperature_decay: 2.0
  
  # Uncertainty estimation
  uncertainty:
    type: "ensemble"  # Options: ensemble, dropout, bootstrap
    n_models: 5
    dropout_rate: 0.1
    n_forward_passes: 10
    n_bootstrap: 5
    model_config:
      hidden_size: 128
      n_layers: 2
  
  # Meta-learning configuration
  meta_learning:
    enable: true
    type: "reptile"  # Options: maml, reptile
    inner_lr: 0.01
    outer_lr: 0.001
    meta_lr: 0.1
    n_inner_steps: 5
  
  # Verification threshold
  verification_threshold: 0.7
```

## 7. Integration with VERL

### 7.1 Main Script

```python
# main_hato.py
import os
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from verl.trainer import RayTrainer

from hato_ray_trainer import RayHATOTrainer
from tree import ReasoningTree, ThoughtNode
from search import adaptive_tree_search
from rewards import HierarchicalRewardModel

def main():
    # Load configurations
    with open("config/model_config.yaml", "r") as f:
        model_config = yaml.safe_load(f)
    
    with open("config/training_config.yaml", "r") as f:
        training_config = yaml.safe_load(f)
    
    with open("config/hato_config.yaml", "r") as f:
        hato_config = yaml.safe_load(f)
    
    # Combine configurations
    config = {
        "model": model_config,
        "trainer": training_config,
        "hato": hato_config,
        # Add standard VERL configurations
        "algorithm": {
            "adv_estimator": "grpo",
            "gamma": 0.99,
            "lam": 0.95,
            "norm_adv_by_std_in_grpo": True,
            "use_kl_in_reward": False,
            "filter_groups": {
                "enable": True,
                "metric": "rewards",
                "max_num_gen_batches": 5
            }
        },
        "actor_rollout_ref": {
            "actor": {
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.28,
                "loss_agg_mode": "token-mean"
            }
        },
        "data": {
            "gen_batch_size": 32,
            "train_batch_size": 8
        }
    }
    
    # Initialize trainer
    trainer = RayHATOTrainer(config)
    
    # Start training
    trainer.train()
    
    # Save final model
    trainer.save_model(os.path.join(config["trainer"]["output_dir"], "final_model"))

if __name__ == "__main__":
    main()
```

### 7.2 Run Script

```bash
#!/bin/bash
# run_hato.sh

# Set environment variables
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="0,1,2,3"  # Adjust based on available GPUs

# Prepare data if needed
if [ ! -f "data/math_train.jsonl" ]; then
    echo "Preparing data..."
    bash scripts/prepare_data.sh
fi

# Run training
python main_hato.py

echo "Training complete!"
```

## 8. Key Differences from Standard DAPO

The HATO trainer implementation differs from the standard DAPO trainer in several key ways:

1. **Tree-Based Processing**: Instead of processing sequences directly, HATO generates and processes reasoning trees with hierarchical structure.

2. **Hierarchical Rewards**: HATO implements a three-component reward system (local, path, causal) instead of the standard sequence-level rewards.

3. **Adaptive Exploration**: HATO incorporates uncertainty estimation and temperature scheduling for more effective exploration.

4. **Group-Based Advantage**: While DAPO uses group-based advantage estimation, HATO extends this to work with tree structures and parent-child relationships.

5. **Meta-Learning**: HATO adds meta-learning components for cross-domain transfer, which is not present in standard DAPO.

6. **Memory Efficiency**: HATO implements sparse tree materialization and value-uncertainty pruning for computational efficiency.

7. **Advanced Metrics**: HATO tracks tree-specific metrics like reasoning diversity and novelty scores.

## 9. Implementation Considerations

### 9.1 Memory Management

The tree-based approach can be memory-intensive, especially for deep trees with high branching factors. The implementation includes several optimizations:

1. **Sparse Tree Materialization**: Only promising nodes are fully materialized.
2. **Batch Processing**: Nodes are processed in batches to limit memory usage.
3. **Pruning Strategies**: Value-uncertainty pruning removes unpromising branches early.

### 9.2 Computational Efficiency

To ensure computational efficiency:

1. **Distributed Search**: Tree search is parallelized across multiple GPUs.
2. **Caching**: Intermediate computations are cached to avoid redundant processing.
3. **Adaptive Branching**: Branching factor is adjusted based on node promise.

### 9.3 Integration with Existing VERL Components

The implementation carefully integrates with existing VERL components:

1. **DataProto Compatibility**: All tree data is converted to VERL's DataProto format.
2. **Worker Group Integration**: Tree generation and processing leverage VERL's worker groups.
3. **Metric Tracking**: Tree-specific metrics are integrated with VERL's logging system.

## 10. Conclusion

This implementation adapts the DAPO trainer from VERL to incorporate the novel components of the HATO framework, including hierarchical rewards, adaptive exploration, and tree-based reasoning. The implementation maintains the technical rigor and efficiency of the original DAPO trainer while adding significant new capabilities for enhanced reasoning in language models.

By following this implementation guide, researchers and practitioners can leverage the HATO framework to improve reasoning capabilities in small language models, particularly for mathematical reasoning tasks.
