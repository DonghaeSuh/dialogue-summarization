from transformers import AutoTokenizer, BartForConditionalGeneration

model_name = "alaggung/bart-r3f"
max_length = 64
num_beams = 5
length_penalty = 1.2
dialogue = ["밥 ㄱ?", "고고고고 뭐 먹을까?", "어제 김치찌개 먹어서 한식말고 딴 거", "그럼 돈까스 어때?", "오 좋다 1시 학관 앞으로 오셈", "ㅇㅋ"]

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
model.eval()

inputs = tokenizer("[BOS]" + "[SEP]".join(dialogue) + "[EOS]", return_tensors="pt")
outputs = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    num_beams=num_beams,
    length_penalty=length_penalty,
    max_length=max_length,
    use_cache=True,
)
summarization = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(summarization)
# 어제 김치찌개를 먹어서 한식 말고 돈가스를 먹기로 했다.