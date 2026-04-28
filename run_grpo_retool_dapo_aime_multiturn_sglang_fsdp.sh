set -x
ulimit -n 65535

ray stop --force 2>/dev/null || true
ipcrm -a 2>/dev/null || true

cd /root/pbrs-rlvr/verl

python3 -m verl.trainer.main_ppo \
    --config-path="/root/pbrs-rlvr/verl/examples/sglang_multiturn/config" \
    --config-name='gsm8k_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=/root/autodl-tmp/datasets/DAPO-Math-17k/retool_dapo/train.parquet \
    data.val_files=/root/autodl-tmp/datasets/AIME-2024/retool_aime2024/train.parquet \
    data.train_batch_size=4 \
    data.max_prompt_length=2048 \
    data.max_response_length=6144 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.custom_cls.path=recipe/retool/retool.py \
    data.custom_cls.name=CustomRLHFDataset \
    reward.custom_reward_function.path=recipe/retool/retool.py \
    reward.custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=8 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="/root/pbrs-rlvr/sandbox_fusion_tool_config.yaml" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.project_name='grpo_retool' \
    trainer.experiment_name='qwen2.5_1.5b_dapo_aime' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.save_freq=30 \
    trainer.test_freq=5 \
    trainer.log_val_generations=10 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard","file"]' \
    trainer.default_local_dir=/root/autodl-tmp/checkpoints/grpo_retool_qwen2.5_1.5b_dapo_aime