import sys
import os

# 프로젝트의 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoTokenizer, BartForConditionalGeneration
import json
import pandas as pd
from src.utils import compute_metrics

model_name = "alaggung/bart-r3f"
max_length = 512
num_beams = 10
length_penalty = 1.2

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='../cache')
model = BartForConditionalGeneration.from_pretrained(model_name, cache_dir='../cache')
model.eval()

# 'inputs', 'labels', 'preds', 'rouge', 'bleu'
df = pd.DataFrame()

def make_data(path):
    inputs = []
    labels = []

    with open(path, "r") as f:
        data = json.load(f)

    def make_chat(inp):
        for cvt in inp['conversation']:
            chats = []
            speaker = cvt['speaker']
            utterance = cvt['utterance']
            chats.append(f"화자{speaker}: {utterance}")
        
        chat = "[BOS]" + "[SEP]".join(chats)  + "[EOS]"
        return chat

    for example in data:
        chat = make_chat(example["input"])
        
        inputs.append(tokenizer(chat, truncation=True, padding=True, max_length=max_length, return_tensors="pt"))
        labels.append(example["output"])

    return inputs, labels

inputs, labels = make_data("resource/data/train.json")

print('## data total length : ', len(inputs), len(labels))

preds = []

for idx in range(len(inputs)):
    try:
        input_ids = inputs[idx]["input_ids"]
        attention_mask = inputs[idx]["attention_mask"]

        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            num_beams=num_beams,
            length_penalty=length_penalty,
            max_length=max_length,
            use_cache=True,
        )
        
    except:
        raise ValueError(f"{idx} row에서 에러 발생 : {len(input_ids[0])}")
    
    preds.append(tokenizer.decode(output[0], skip_special_tokens=True))

metrics = compute_metrics(preds, labels)

results = pd.DataFrame()
results['preds'] = preds
results['labels'] = labels

results.to_csv(f"./bart_result1_rouge_{metrics['rouge-1']}_bert_{metrics['bert-score']}_bleu_{metrics['bleurt']}.csv", index=False)