time=$(date '+%Y-%m-%d-%H%M%S')
start_time=$(date +%s)

source activate ano_trl


id=3

time=$(date '+%Y-%m-%d-%H%M%S')
/bin/echo "${id} GRPO"
export CUDA_VISIBLE_DEVICES=$id
# CUDA_VISIBLE_DEVICES=$id 
echo "Starting GRPO with Reduced Batch Size (8)..."
accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    --num_processes 1 \
    examples/scripts/grpo/grpo_tldr.py \
    --output_dir models/minimal/grpo_tldr_${time} \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 2 \
    --num_generations 4 \
    --gradient_accumulation_steps 8 \
    --max_steps 15625 \
    --model_name_or_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --dataset_name "TRL-Lib/tldr" \
    --beta 0.05 \
    --max_prompt_length 512 \
    --max_completion_length 53 > train_GRPO_${time}.txt 2>&1


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
