time=$(date '+%Y-%m-%d-%H%M%S')
start_time=$(date +%s)

source activate trl_dev


id=3
time=$(date '+%Y-%m-%d-%H%M%S')
/bin/echo "${id} ANO"
export CUDA_VISIBLE_DEVICES=$id
# CUDA_VISIBLE_DEVICES=$id 
echo "Starting ANO with Reduced Batch Size (8)..."
accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    --num_processes 1 \
    examples/scripts/ano/ano_tldr.py \
    --num_sample_generations 0 \
    --output_dir models/minimal/ano_tldr \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --total_episodes 1000000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --local_rollout_forward_batch_size 8 \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --dataset_name "TRL-Lib/tldr" > train_ANO_$time.txt 2>&1


    #!/bin/bash



end_time=$(date +%s)
duration=$((end_time - start_time))

# 计算小时、分钟、秒
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

printf "脚本运行时间: "
if [ $hours -gt 0 ]; then
    printf "%02d小时 " $hours
fi
if [ $minutes -gt 0 ] || [ $hours -gt 0 ]; then
    printf "%02d分钟 " $minutes
fi
printf "%02d秒\n" $seconds
