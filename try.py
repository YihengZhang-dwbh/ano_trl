import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 设置你的 checkpoint 路径 (根据截图路径)
# 注意：虽然文件夹里有个 global_step2000，但我们要读的是当前这一层
# model_path = "/home/yiheng/trl/models/minimal/ano_0.2_0_tldr/checkpoint-500/"
model_path = "/home/yiheng/trl/models/minimal/ppo_tldr/"
# model_path = "/home/yiheng/trl/models/minimal/ano0.2_tldr/checkpoint-1500/"
# model_path = "/home/yiheng/trl/models/minimal/ano_0.2_0.03_tldr/checkpoint-1500/"

print(f"🔄 正在加载模型: {model_path} ...")

# 2. 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 3. 加载模型 (bf16精度，防止显存溢出)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16, 
    device_map="cuda:0"  # 放到 GPU 0 上
)
print("✅ 模型加载成功！")

# 4. 准备测试 Prompt (TL;DR 风格)
# 找一个 Reddit 风格的长文本
# article = """
# I've been dating my girlfriend for 3 years. She is amazing and we get along great. 
# However, recently she has been spending a lot of time with her "work husband". 
# They text late at night and go to lunch every day. 
# I confronted her and she said I'm being insecure and controlling. 
# I found messages where she calls him "babe" but she says it's a joke.
# I don't know if I should break up with her or try to fix it.
# """

article = """
SUBREDDIT: r/relationships  TITLE: Boyfriend(26M) leaving me(24F) to go to Antarctica.  POST: I'll make this short and sweet. We've been together 8 months. Things moved fast for us. We live together and we are both in love. I'm in nursing school, and he's working on his Master's. He just landed an amazing opportunity: he has a job offer to go to the South Pole for research. He'll be gone for a year. I really don't want to stop him from going, and this opportunity is too amazing to pass up. He states that he wants to do long distance with me, but this will obviously come with some complications. I feel...torn. I really want him to go but I feel like I will be missing out on a huge chunk of his life. I'm also so afraid that he'll meet someone else or something else will happen that will tear him away from me. Please help!  TL;DR: 
"""


# 加上 TRL/TLDR 数据集特有的 Prompt 格式
prompt = article + "\nTL;DR:"





article2 = """
[Step 820] SUBREDDIT: r/relationships  TITLE: I [26 M] am married, but have developed an innocent crush on a girl and feel guilty about it  POST: I'm not at all worried that I'm actually going to be unfaithful. I'm happy in my marriage, and wouldn't even think of doing that to my wife. Plus, I doubt this girl is even attracted to me, anyway.   But I can't help but feel super attracted to this girl who I hardly know; she's pretty, friendly, dresses well, and seems intelligent. If I got to know her, I'd probably find something not to like about her. But I really have no reason to get to know her. She works in a different part of the building, and her job is unrelated to mine.  I haven't had a crush like this in years. Even though I haven't done anything, I feel guilty about it, and I don't know how to shake it.  TL;DR: I am married, but have developed an innocent crush on a girl and feel guilty about it. How do I shake it without ruining everything?
"""

with open("ano_live_log.txt", 'r') as file:
    lines = file.readlines()
    last_line = lines[-1] if lines else ''
article2 = last_line.strip()

for i in range(0, 20):
    if article2[i] == "]":
        step = article2[0:i+1]
        article2 = article2[i+1:].strip()
        ano = article2.split(" TL;DR:")[1].strip()
        prompt = article2.split(" TL;DR:")[0].strip() + "\nTL;DR:"
        break

file = open('analysis.txt', 'w', encoding='utf-8')

strt = f"{step} {prompt}\nANO: {ano}\nPPO:\n"
print(strt)
file.write(strt)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

# 5. 生成结果 (不做采样，用 greedy search 看它最想说什么)
strt = "\n🤖 模型生成结果 (Greedy):\n" + "-" * 30 + "\n"
print(strt)
file.write(strt)
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=50, 
        do_sample=False,  # 关掉随机性，看模型最确信的输出
        pad_token_id=tokenizer.eos_token_id
    )

output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# 只打印 TL;DR 之后的部分
strt = output_text.split("TL;DR:")[-1].strip() + "\n" + "-" * 30 + "\n"
print(strt)
file.write(strt)

# 6. 生成结果 (开启采样，看看多样性)
strt = "\n🎲 模型生成结果 (Sampling Temp=0.7):\n" + "-" * 30 + "\n"
print(strt)
file.write(strt)
with torch.no_grad():
    outputs_sample = model.generate(
        **inputs, 
        max_new_tokens=50, 
        do_sample=True, 
        temperature=0.7,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )

output_text_sample = tokenizer.decode(outputs_sample[0], skip_special_tokens=True)
strt = output_text_sample.split("TL;DR:")[-1].strip() + "\n" + "-" * 30 + "\n"
print(strt)
file.write(strt)
file.close()
