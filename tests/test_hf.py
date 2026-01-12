from huggingface_hub import HfApi, DatasetCard

# api = HfApi()
card = DatasetCard.load("openai/gsm8k")   # dataset id
# text = card.text

# print(card)
# print(text[:1000])   # README 前 1000 字

# 将 card 内容完整保存到本地文件，先转成字符串
with open("/mnt/DataFlow/scy/One-Eval/cache/gsm8k_dataset_card.md", "w", encoding="utf-8") as f:
    f.write(str(card))   # 使用 str(card) 查看完整内容
print("已保存 card 到 gsm8k_dataset_card.md")
