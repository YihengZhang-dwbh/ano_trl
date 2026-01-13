import torch
import gc
from dataclasses import dataclass, field
from tqdm import tqdm
from datasets import load_dataset
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForCausalLM

# 尝试导入我们在 judges.py 中定义的 DeepSeek 裁判
try:
    from judges import DeepSeekPairwiseJudge, HfPairwiseJudge, OpenAIPairwiseJudge
except ImportError:
    # 兼容 TRL 官方路径结构
    from trl.experimental.judges import DeepSeekPairwiseJudge, HfPairwiseJudge, OpenAIPairwiseJudge

@dataclass
class ScriptArguments:
    model_a_path: str = field(
        default="",
        metadata={"help": "Path to Model A (ANO)."}
    )
    model_b_path: str = field(
        default="",
        metadata={"help": "Path to Model B (PPO)."}
    )
    judge_model: str = field(
        default="deepseek",
        metadata={"help": "Judge model type: 'deepseek', 'gpt', or 'hf_model_id'."}
    )
    num_examples: int | None = field(
        default=100, 
        metadata={"help": "Number of examples to evaluate."}
    )
    batch_size: int = field(
        default=8,
        metadata={"help": "Batch size for generation."}
    )

def generate_responses(model_path, prompts, args, model_name="Model"):
    """
    使用用户验证过的方式加载模型并生成
    """
    print(f"\n🔄 [{model_name}] Loading from: {model_path} ...")
    
    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # 生成时必须左填充

    # 2. 加载模型 (完全按照用户提供的实测代码)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" # 自动分配 GPU
    )
    model.eval()
    print(f"✅ [{model_name}] Loaded successfully.")

    completions = []
    print(f"🚀 [{model_name}] Generating {len(prompts)} responses...")
    
    # 分批生成
    for i in tqdm(range(0, len(prompts), args.batch_size)):
        batch_prompts = prompts[i : i + args.batch_size]
        
        # 构造输入
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=53, # 与训练时的 response_length 保持一致
                do_sample=True,    # 开启采样以获得更自然的结果
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码，只取生成的回复部分
        batch_texts = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        completions.extend([t.strip() for t in batch_texts])

    # 3. 显存清理
    print(f"🧹 [{model_name}] Cleaning up memory...")
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return completions

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    file = open('evaluation_log.txt', 'a', encoding='utf-8')

    # 1. 准备数据
    strt = f"📚 Loading Dataset (Validation Split, {script_args.num_examples} samples)..."
    print(strt)
    file.write(strt)
    dataset = load_dataset("trl-lib/tldr", split="validation")
    if script_args.num_examples:
        dataset = dataset.select(range(script_args.num_examples))
    
    prompts = dataset["prompt"]

    # 2. 生成 Model A (ANO)
    completions_a = generate_responses(
        script_args.model_a_path, prompts, script_args, model_name="ANO"
    )

    # 3. 生成 Model B (PPO)
    completions_b = generate_responses(
        script_args.model_b_path, prompts, script_args, model_name="PPO"
    )

    # 4. 初始化裁判
    strt = f"\n⚖️  Initializing Judge: {script_args.judge_model}"
    print(strt)
    file.write(strt)
    if "deepseek" in script_args.judge_model.lower():
        judge = DeepSeekPairwiseJudge()
    elif "gpt" in script_args.judge_model.lower():
        judge = OpenAIPairwiseJudge(script_args.judge_model)
    else:
        judge = HfPairwiseJudge(script_args.judge_model)

    # 5. 开始评判
    strt = "\n⚡️ Judging pairs..."
    print(strt)
    file.write(strt)
    # 配对数据 [ANO结果, PPO结果]
    completion_pairs = [[c_a, c_b] for c_a, c_b in zip(completions_a, completions_b)]
    
    # best_idxs: 0 代表前一个(ANO)赢, 1 代表后一个(PPO)赢
    best_idxs = judge.judge(prompts, completion_pairs, shuffle_order=True)

    # 6. 统计结果
    wins_ano = best_idxs.count(0)
    wins_ppo = best_idxs.count(1)
    ties = best_idxs.count(-1)
    total_valid = len(best_idxs) - ties

    strt = "\n" + "="*50 + "\n" 
    strt += f"📊  Evaluation Results (N={len(best_idxs)})\n"
    strt += f"Model A (ANO): {script_args.model_a_path}\n"
    strt += f"Model B (PPO): {script_args.model_b_path}\n"
    strt += "-" * 50 + "\n"
    strt += f"ANO Wins : {wins_ano:<5} ({wins_ano/total_valid:.1%})\n"
    strt += f"PPO Wins : {wins_ppo:<5} ({wins_ppo/total_valid:.1%})\n"
    strt += f"Ties/Err : {ties}\n"
    strt += "-" * 50 + "\n"
    print(strt)
    file.write(strt)
    
    if total_valid > 0:
        # 计算 PPO 相对 ANO 的胜率
        ppo_win_rate = wins_ppo / total_valid
        strt = f"\n🏆 PPO Win Rate vs ANO: {ppo_win_rate * 100:.2f}%\n"
        print(strt)
        file.write(strt)
    strt = "="*50
    print(strt)
    file.write(strt + "\n")
    file.close()
