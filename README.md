# HATO: Hierarchical Adaptive Tree Optimization

HATO (Hierarchical Adaptive Tree Optimization) is a novel framework that combines Generalized Reward Policy Optimization (GRPO) with Tree of Thought (ToT) reasoning to enhance mathematical problem-solving capabilities in Large Language Models.

## Overview

HATO implements a hierarchical reward modeling framework that combines:

- Local rewards based on the quality of individual reasoning steps
- Path rewards based on the final outcome of reasoning paths
- Causal rewards based on counterfactual contribution to successful paths

The framework is built on two key theoretical foundations:

1. **Theorem 1**: Optimizing the expected hierarchical reward leads to monotonic improvement in expected sequence-level reward.
2. **Theorem 2**: The variance of the policy gradient estimator using hierarchical rewards is bounded by a function of the tree structure and reward correlations.

## Project Structure

```
hato/
├── config/                  # Configuration files
│   ├── test_config.yaml     # Configuration for testing
│   └── gsm8k_config.yaml    # Configuration for GSM8K evaluation
├── scripts/                 # Utility scripts
│   ├── run_test.sh          # Script to run tests
│   ├── run_gsm8k_eval.sh    # Script to run GSM8K evaluation
│   └── run_analysis.sh      # Script to analyze results
├── src/                     # Source code
│   ├── tree/                # Tree structure components
│   │   ├── node.py          # ThoughtNode implementation
│   │   ├── tree.py          # ReasoningTree implementation
│   │   └── metrics.py       # Tree metrics computation
│   ├── search/              # Search algorithms
│   │   ├── adaptive_search.py  # Adaptive tree search
│   │   ├── uncertainty.py   # Uncertainty estimation
│   │   └── pruning.py       # Tree pruning strategies
│   ├── rewards/             # Reward models
│   │   ├── hierarchical.py  # Hierarchical reward model
│   │   ├── local.py         # Local reward functions
│   │   └── causal.py        # Causal credit assignment
│   ├── meta/                # Meta-learning components
│   ├── trainers/            # Training components
│   │   └── hato_trainer.py  # HATO trainer implementation
│   └── utils/               # Utility functions
├── test_hato.py             # Test script for HATO
├── test_hato_mock.py        # Test script with mock model
├── gsm8k_evaluation.py      # GSM8K evaluation script
├── analyze_results.py       # Results analysis script
├── requirements.txt         # Project dependencies
└── README.md                # This file
```

## Core Components

### Tree Structure

- **ThoughtNode**: Represents a single reasoning step with hierarchical rewards
- **ReasoningTree**: Manages the tree of reasoning steps, including expansion, pruning, and path tracking

### Search Algorithms

- **Adaptive Tree Search**: Implements beam search, BFS, and DFS variants
- **Uncertainty Estimation**: Provides ensemble, dropout, and bootstrap methods
- **Pruning Strategies**: Implements value-uncertainty pruning and sparse tree materialization

### Reward Models

- **Hierarchical Reward Model**: Combines local, path, and causal rewards
- **Local Reward Model**: Evaluates the quality of individual reasoning steps
- **Causal Reward Model**: Estimates counterfactual contribution to successful paths

### HATO Trainer

- Adapts VERL's DAPO trainer for hierarchical tree optimization
- Implements tree generation and processing for training
- Provides meta-learning components for cross-domain transfer

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/hato.git
cd hato
```

2. Install dependencies:

```bash
conda create -n hato
conda activate hato
pip install -r requirements.txt
git clone https://github.com/volcengine/verl.git && cd verl && pip install -e . --no-deps
```

## Usage

### Running Tests

```bash
./scripts/run_test.sh
```

### Evaluating on GSM8K

```bash
./scripts/run_gsm8k_eval.sh
```

### Analyzing Results

```bash
./scripts/run_analysis.sh
```

## Configuration

The framework can be configured through YAML files in the `config/` directory:

- `test_config.yaml`: Configuration for testing
- `gsm8k_config.yaml`: Configuration for GSM8K evaluation

Key configuration parameters:

```yaml
hato:
  tree:
    max_depth: 4 # Maximum tree depth
    branching_factor: 3 # Number of branches per node
    search_algorithm: "beam" # Search algorithm (beam, bfs, dfs)
    max_nodes_per_level: 3 # Maximum nodes per level

  reward_weights:
    alpha: 0.4 # Weight for local reward
    beta: 0.5 # Weight for path reward
    gamma: 0.1 # Weight for causal reward

  exploration:
    initial_temperature: 0.7 # Initial temperature for exploration
    temperature_decay: 2.0 # Temperature decay rate
```

## Results

The framework has been evaluated on the GSM8K dataset, demonstrating:

- Moderate performance on mathematical reasoning tasks
- Multi-step reasoning processes with an average path length of 3.8 steps
- High reasoning diversity (0.72), indicating effective exploration of diverse reasoning paths

## Future Work

1. **Scaling to Larger Models**: Scale to larger models like Qwen 7B
2. **Adaptive Exploration**: Enhance adaptive exploration strategies
3. **Reward Model Refinement**: Further refine the hierarchical reward model
4. **Meta-Learning Optimization**: Optimize meta-learning components
5. **Memory Efficiency**: Implement additional memory optimization techniques

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- VERL framework for the base DAPO trainer
- Tree of Thought (ToT) paper for the reasoning framework
- GSM8K dataset for evaluation
