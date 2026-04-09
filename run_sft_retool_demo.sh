set -x

experiment_name=multiturn-sft-qwen-2.5-1.5b-instruct
TRAIN_DATA=/root/autodl-tmp/datasets/ReTool-SFT/data/train-00000-of-00001.parquet
EVAL_DATA=/root/autodl-tmp/datasets/ReTool-SFT/data/train-00000-of-00001.parquet
MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct
SAVE_PATH=/root/autodl-tmp/checkpoints/$experiment_name

# add data.max_token_len_per_gpu=16384 for use_remove_padding=True

torchrun --standalone --nnodes=1 --nproc_per_node=1 \
    -m verl.trainer.sft_trainer \
    data.train_files=$TRAIN_DATA \
    data.val_files=$EVAL_DATA \
    data.messages_key=messages \
    data.micro_batch_size_per_gpu=2 \
    data.max_length=16384 \
    data.max_token_len_per_gpu=16384 \
    +data.multiturn.enable=true \
    +data.multiturn.messages_key=messages \
    +data.multiturn.tools_key=tools \
    +data.filter_overlong_prompts=True \
    model.path=$MODEL_PATH \
    model.use_remove_padding=True \
    engine=fsdp \
    engine.ulysses_sequence_parallel_size=1 \
    optim=fsdp \
    optim.lr=1e-5 \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=multiturn-sft-math \
    trainer.experiment_name=$experiment_name \
    trainer.logger='["console"]' \
    trainer.total_epochs=3
    