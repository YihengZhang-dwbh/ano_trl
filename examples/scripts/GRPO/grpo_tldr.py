# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil

import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import ModelConfig, ScriptArguments, get_kbit_device_map, get_peft_config, get_quantization_config
# [Change 1] 替换 PPO 引用为 GRPO
from trl import GRPOConfig, GRPOTrainer 

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Load Policy Model
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path if training_args.sft_model_path else model_args.model_name_or_path, 
        trust_remote_code=model_args.trust_remote_code, 
        **model_kwargs
    )
    
    # [Change 3] GRPO 不需要显式加载 ref_policy 和 value_model
    # 只需要加载 Reward Model 用于计算奖励
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        script_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1, **model_kwargs
    )
    # 将 Reward Model 移动到正确的设备 (如果不是自动 device_map)
    if quantization_config is None:
        reward_model = reward_model.to(policy.device)

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    train_dataset = dataset[script_args.dataset_train_split]
    eval_dataset = dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None

    # [Change 4] 数据处理逻辑变更：
    # PPO 需要 tokenize 成 input_ids，但 GRPO 需要原始 prompt 进行 generate。
    # 我们保留过滤逻辑，但直接基于 prompt 文本长度过滤。
    
    def filter_long_prompts(example):
        # 简单估算 token 数，或者用 tokenizer 实际跑一下
        return len(tokenizer(example["prompt"])["input_ids"]) <= 512

    with PartialState().local_main_process_first():
        # 只需要过滤，不需要 tokenize
        train_dataset = train_dataset.filter(filter_long_prompts, num_proc=training_args.dataset_num_proc)
        if eval_dataset is not None:
            eval_dataset = eval_dataset.filter(filter_long_prompts, num_proc=training_args.dataset_num_proc)

    ################
    # Reward Function
    ################
    # [Change 5] 定义 GRPO 所需的 Reward Function
    # GRPO 接受一个函数列表，我们把加载好的 reward_model 包装进去
    def reward_func(prompts, completions, **kwargs):
        # 拼接 prompt 和 completion，这是 Reward Model 预期的输入格式
        inputs = [p + c for p, c in zip(prompts, completions)]
        encoding = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
        
        # 移动到 reward_model 所在的设备
        input_ids = encoding.input_ids.to(reward_model.device)
        attention_mask = encoding.attention_mask.to(reward_model.device)
        
        with torch.no_grad():
            outputs = reward_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
        # 假设是标量输出，展平返回
        return logits.flatten()

    ################
    # Training
    ################
    # [Change 6] 实例化 GRPOTrainer
    trainer = GRPOTrainer(
        model=policy,
        reward_funcs=[reward_func], # 传入包装好的奖励函数
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer, # GRPOTrainer 使用 processing_class 来处理 padding
        peft_config=get_peft_config(model_args),
    )
    
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
