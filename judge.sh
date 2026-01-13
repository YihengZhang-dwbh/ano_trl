time=$(date '+%Y-%m-%d-%H%M%S')

source activate ano_trl

export CUDA_VISIBLE_DEVICES=0

python judge.py \
    --model_a_path "models/minimal/ano_0.2_0.05_tldr_2026-01-13-115400/checkpoint-1000" \
    --model_b_path "models/minimal/ppo_r" \
    --num_examples 100 \
    --batch_size 16
