import evaluate
import numpy as np
import re
import json
from transformers import EvalPrediction, AutoTokenizer
import torch
import random
from typing import Dict
import argparse

config_name = input("## input utils config file name ##\n")

print("## utils Config File : ", config_name)
# change abspath to run in ipynb file
# with open(f'/mnt/g/내 드라이브/국립국어원_일상대화요약/korean_dialog/dialogue-summarization/configs/{config_name}', 'r') as f:
#     config = json.load(f)

with open(f'configs/{config_name}', 'r') as f:
    config = json.load(f)

rouge = evaluate.load('rouge')
bert_score = evaluate.load('bertscore')
bleurt = evaluate.load('bleurt', 'bleurt-large-512', module_type="metric")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    return preds, labels

def preprocess_logits_for_metrics(logits, labels):
      """
      Original Trainer may have a memory leak. 
      This is a workaround to avoid storing too many tensors that are not needed.
      """
      pred_ids = torch.argmax(logits, dim=-1)
      return pred_ids, labels

def compute_metrics(eval_pred: EvalPrediction):
    tokenizer = AutoTokenizer.from_pretrained(config["arch"]["model_id"])

    # compute Rouge-1 F1 score
    labels = eval_pred.label_ids # (batch_size, seq_len)
    predictions = eval_pred.predictions[0].reshape(labels.shape[0],-1) # (batch_size, seq_len)

    # Replace -100 with pad_token_id
    mask  = np.where(labels == -100)
    labels[mask] = tokenizer.pad_token_id
    predictions[mask] = tokenizer.pad_token_id

    # Decoding
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Simple postprocessing
    predictions, labels = postprocess_text(predictions, labels)

    rouge_scores = rouge.compute(predictions=predictions, references=labels, rouge_types=["rouge1"])
    # rouge_scores = rouge.get_scores(predictions, labels, avg=True)
    bert_scores = bert_score.compute(predictions=predictions, references=labels, lang="ko")
    bleurt_scores = bleurt.compute(predictions=predictions, references=labels)

    bertScore = sum(bert_scores['f1']) / len(labels)
    bleurtScore = sum(bleurt_scores['scores']) / len(labels)

    rouge1 = rouge_scores['rouge1']
    total = (bertScore + bleurtScore + rouge1) / 3

    return {"total" : round(total, 4), "rouge1" : round(rouge1, 4), "BERTScore" : round(bertScore, 4), "BLEURT": round(bleurtScore, 4)}


## Preprocess functions ##
def remove_empty_utterance(data:json):
    """
    Remove empty utterances from the data
    """
    for example in data:
        example['input']['conversation'] = [cvt for cvt in example['input']['conversation'] if cvt['utterance'] != '']
    return data


def correct_wrong_output(data:json):
    """
    Correct wrong speakers in outputs of train samples 'train-000401', 'train-000402'
    """
    data[400]['output'] = data[400]['output'].replace('SD2100504','SD2110504')
    data[401]['output'] = data[401]['output'].replace('SD2110503','SD2100503')

    return data


def file_preprocess(data:json, is_train=False):
    data = remove_empty_utterance(data)

    if is_train == True:
        data = correct_wrong_output(data)

    return data


"""
불용어 처리
- name1, name2..
- name1이가, name3은 ...
- 뒤에 물결이 붙는 경우 ("음~", "아~")
- 그, 뭐, 어, 인제, 막, 아, 음, 읍, 오, 으
- 한 글자가 두번 이상 반복되는 경우 ("또 또", "그 그")
- 의미가 적다고 판단되는 빈출부사 (좀, 또)
"""

stopwords_pattern = [r'name[0-9]\S*', r'\w~', r'\b으\b', r'\b그\b', r'\b뭐\b', r'\b어\b',  r'\b인제\b', r'\b이제\b', r'\b막\b', r'\b아\b', r'\b음\b', r'\b읍\b', r'\b오\b', r'\b으\b', r'\b이\b', r'\b먹\b', r'\b있\b', r'\b좀\b', r'\b또\b', r'딱\b']
stopwords = ['x', '쪼금', '그러면 그런', '약간 조금']

def remove_stopwords(text):
    for pattern in stopwords_pattern:
        text = re.sub(pattern, '', text)
    
    # 두 번 이상 반복되는 경우
    text = re.sub(r'\b(\w)\s+\1\b', r'\1', text)
    text = re.sub(r'\b([가-힣a-zA-Z0-9_]+)\s+\1\b', r'\1', text)

    # name으로 시작하는 인칭 대명사 제거
    re.sub(r'name[0-9]\S*', '', text)

    # stopwords 제거
    for stopword in stopwords:
        text = re.sub(stopword, '', text)

    # 공백 두 번 이상 연속 -> 1개로
    text = re.sub(r'\s{2,}', ' ', text)
    return text


# stopwords + 반복 어구 제거
def text_preprocess(text):
    text = remove_stopwords(text)
    
    return text



def set_seed(config: Dict):
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# argparse에서 boolean인자 받기
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')