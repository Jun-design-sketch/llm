from transformers import AutoTokenizer, AutoModelForCausalLM

# model / tokenizerを呼び出す
model_id = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)
# 質問
messages = [
    {"role": "user", "content": "第100代の内閣総理大臣は誰ですか？"}
]
# モデル入力に変換
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
# 返答生成
outputs = model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=False,
)
# 返答部分のみ出力
answer_tokens = outputs[0][inputs.input_ids.shape[1]:]
print(tokenizer.decode(answer_tokens, skip_special_tokens=True))
