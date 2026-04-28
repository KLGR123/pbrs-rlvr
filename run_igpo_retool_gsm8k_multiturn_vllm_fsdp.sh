set -x
ulimit -n 65535

#set vllm v1 env
export VLLM_USE_V1=1

cd /root/pbrs-rlvr/verl # for config relative path

python3 -m verl.trainer.main_ppo \
    --config-path="/root/pbrs-rlvr/verl/examples/sglang_multiturn/config" \
    --config-name='gsm8k_multiturn_grpo' \
    actor_rollout_ref.rollout.name=vllm \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    algorithm.use_kl_in_reward=False \
    algorithm.adv_estimator=igpo \
    +algorithm.ig_config.ig_type=log_prob_diff \
    +algorithm.ig_norm_mode=separate \
    +algorithm.ig_reward_weight=1.0 \
    +algorithm.outcome_reward_weight=1.0 \
    +algorithm.ig_config.ig_source=policy \
    algorithm.gamma=1.0 \
    trainer.critic_warmup=0 \
    trainer.project_name='igpo_retool' \
    trainer.experiment_name='qwen2_1.5b_gsm8k' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=5 \
    trainer.log_val_generations=10 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.logger='["console","tensorboard","file"]' \
    trainer.default_local_dir=/root/autodl-tmp/checkpoints/igpo_retool_qwen2_1.5b_gsm8k \
    data.train_files=/root/autodl-tmp/datasets/gsm8k/multiturn/train.parquet \
    data.val_files=/root/autodl-tmp/datasets/gsm8k/multiturn/test.parquet \
    trainer.total_epochs=2 \
    actor_rollout_ref.rollout.trace.token2text=False \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="/root/pbrs-rlvr/sandbox_fusion_tool_config.yaml" \
    actor_rollout_ref.rollout.free_cache_engine=True