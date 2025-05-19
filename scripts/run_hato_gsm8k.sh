#!/bin/bash
set -x

# Set wandb environment variables
export WANDB_MODE=online
export WANDB_DIR=wandb_logs
export WANDB_PROJECT=hato_gsm8k
export WANDB_NAME=qwen_1.7b_hato

# Disable SSL verification for model downloads
export CURL_CA_BUNDLE=""
export REQUESTS_CA_BUNDLE=""
export HF_HUB_DISABLE_SSL_VERIFICATION=1

# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=6000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    'trainer.logger=[console,wandb]' \
    trainer.project_name=hato_gsm8k \
    trainer.experiment_name=qwen_1.7b_hato \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 \
    +hato.enable=True \
    +hato.tree.max_depth=3 \
    +hato.tree.branching_factor=4 \
    +hato.tree.search_algorithm=mcts \
    +hato.tree.max_nodes_per_level=16 \
    +hato.reward_weights.alpha=0.4 \
    +hato.reward_weights.beta=0.3 \
    +hato.reward_weights.gamma=0.3 \
    +hato.exploration.initial_temperature=1.0 \
    +hato.exploration.temperature_decay=0.1 \
    +hato.uncertainty.type=ensemble \
    +hato.uncertainty.n_models=3 \
    +hato.meta_learning.enable=True \
    +hato.meta_learning.type=maml \
    +hato.meta_learning.inner_lr=0.01 \
    +hato.meta_learning.outer_lr=0.001 \
    +hato.meta_learning.n_inner_steps=5 \
    +hato.memory.sparse_materialization=True \
    +hato.memory.pruning_threshold=0.1 \
    +hato.verification_threshold=0.8 $@ 