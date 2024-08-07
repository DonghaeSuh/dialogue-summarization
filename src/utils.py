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


# 이상치 output 처리
def correct_wrong_output(data:json, is_train=False):
    """
    1. Correct wrong speakers in outputs of train samples 'train-000401', 'train-000402, 'train-000111'
    2. Add dot(.) at the end of the last sentence in outputs of train samples 'train-000130'
    4. Replace speaker name 'SSD' with 'SD' in outputso of 'train-000030', 'train-000193' and 'dev-000085'
    5. Remove duplicate sentences in outputs of dev samples 'dev-000093'.
    """
    if is_train == True:
        # Correct wrong speakers
        data[400]['output'] = data[400]['output'].replace('SD2100504','SD2110504')
        data[401]['output'] = data[401]['output'].replace('SD2110503','SD2100503')
        data[110]['output'] = data[110]['output'].replace('SD20010813','SD2001083')
        # Add dot(.) at the end of the last sentence
        data[129]['output'] = data[129]['output'] + '.'
        # Replace speaker name
        data[29]['output'] = data[29]['output'].replace('SSD', 'SD')
        data[192]['output'] = data[192]['output'].replace('SSD', 'SD')

    else:
        # Replace speaker name
        data[84]['output'] = data[84]['output'].replace('SSD', 'SD')
        # Remove duplicate sentences
        data[92]['output'] = '.'.join(data[92]['output'].split('.')[1:]).strip()

    return data


# total summary(output의 맨 첫 번째 문장) 형식 통일을 위한 이상치 output 처리
def change_weird_output(data:json, is_train=False):
    """
    Standardize the type of the output of train-000032, train-000418, dev-000074, dev-000093
    """
    # Standardize the type of outputs
    if is_train == True:
        # train-000032 : total_summary 교체
        output = data[31]['output'].split('.')
        total_summary = "두 화자는 이 대화에서 진로 관련 고민에 대해 이야기했습니다. "
        data[31]['output'] = total_summary + '.'.join(output[1:])

        # train-000418 : total_summary 추가
        total_summary = "두 화자는 이 대화에서 다이어트에 대해 이야기했습니다. "
        data[417]['output'] = total_summary + data[417]['output']

    else:
        # dev-000074 : total_summary 수정
        data[73]['output'] = "두 화자는 "+ data[73]['output'] # 이 대화에서 -> 두 화자는 이 대화에서

        # dev-000093 : total_summary 추가
        total_summary = "두 화자는 이 대화에서 엔시티와 방탄소년단에 대해 이야기 했습니다. "
        data[92]['output'] = total_summary + data[92]['output']
    
    return data



# output에 SD가 예외적으로 들어간 경우 처리
def remove_sd_in_total_summary(data:json, is_train=False):
    """
    Remove 'SD' in total_summary of train-000020 and train-000176
    """
    if is_train == True:
        # train-000020 : total_summary 수정
        data[19]['output'] = data[19]['output'].replace('SD2000039의 꿈인 ','')

        # train-000176 : total_summary '.' 가 빠져있던 것을 수정
        output = data[175]['output']
        data[175]['output'] = re.sub(r'(장단점에 대해 말했습니다)\s+(SD\d{7}(?:은|는))', r'\1. \2', output)

    return data



# utterance와 output에서는 '.' 뒤에 공백이 무조건 존재하는 형태로 통일 / 문장 맨 마지막의 경우는 '.'으로 통일
def add_space_after_period_and_remove_control_characters(data:json):
    """
    Add space after period if there is no space after period
    text = re.sub(r'\.(?=\S)', '. ', text)
    """
    # Add space after period in utterances
    for example in data:
        example['input']['conversation'] = [{'speaker': cvt['speaker'], 'utterance': re.sub(r'\.(?=\S)', '. ', cvt['utterance']).strip()} for cvt in example['input']['conversation']]

    # Remove_control_characters and Add space after period in outputs
    for example in data:
        output = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', example['output'])
        example['output'] = re.sub(r'\.(?=\S)', '. ', output).strip()

    return data



# total summary(output의 맨 첫 번째 문장) 형식을 "두 화자는 이 대화에서"로 통일
def total_summary_generalization(data:json):
    """
    Standardize the format of the total summary in the first sentence of the output 
    to start with "두 화자는 이 대화에서".
    """
    types = ["두 화자는", "화자들은" ,"두 사람은", "이 대화에서는"] # "두 화자는 이 대화에서"
    types2 = r"SD\d{7}(?:와|과).*SD\d{7}(?:은|는)"

    for example in data:
        output = example['output']
        total_summary = output.split('.')[0]

        if "두 화자는 이 대화에서" in total_summary:
            continue
        elif re.search(types2, total_summary):
            total_summary = re.sub(r'(.*)'+types2, '두 화자는 이 대화에서', total_summary)+'.'
            example['output'] = total_summary+'.'.join(output.split('.')[1:])
        else:
            for type in types:
                if type in total_summary:
                    total_summary = re.sub(r'(.*)'+type, '두 화자는 이 대화에서', total_summary)+'.'
                    example['output'] = total_summary+'.'.join(output.split('.')[1:])
                    break
    
    return data


def file_preprocess(data:json, is_train=False):
    """
    Preprocess the data
    - correct_wrong_output 
        : correct wrong speakers in outputs of train, dev samples
    - change_weird_output 
        : change output of train, dev samples for standardization
    - remove_sd_in_total_summary 
        : remove 'SD' in total_summary of train, dev samples
    - add_space_after_period_and_remove_control_characters 
        : add space after period if there is no space after period and remove control characters
    - total_summary_generalization 
        : standardize the format of the total summary in the first sentence of the output
    """
    print("## file_preprocess start ...")
    data = remove_empty_utterance(data)
    data = correct_wrong_output(data, is_train)
    data = change_weird_output(data, is_train)
    data = remove_sd_in_total_summary(data, is_train)
    data = add_space_after_period_and_remove_control_characters(data, is_train)
    data = total_summary_generalization(data, is_train)
    print("## file_preprocess done ...")

    return data


"""
불용어 처리

## hyperstella2 ##
- name1, name2..
- 뒤에 물결이 붙는 경우 ("음~", "아~")
- 그, 뭐, 어, 인제, 막, 아, 음, 읍, 오, 으
- 한 글자가 두번 이상 반복되는 경우 ("또 또", "그 그")


## nova ##
- name 그대로 유지
- 뒤에 물결이 붙는 경우 ("음~", "아~")
- 그, 뭐, 어, 인제, 막, 아, 음, 읍, 오, 으
- 단어가 두 번 이상 반복되는 경우 제거 ( r'\b([가-힣a-zA-Z0-9_]+)\s+\1\b')


## nova3, hypernova ##
- name 그대로 유지
- 뒤에 물결이 붙는 경우 ("음~", "아~")
- 그, 뭐, 어, 인제, 막, 아, 음, 읍, 오, 으
- 단어가 두 번 이상 반복되는 경우 제거 ( r'\b([가-힣a-zA-Z0-9_]+)\s+\1\b')
- x를 포함한 단어 제거 (r'\b[가-힣a-zA-Z]*[xX][가-힣a-zA-Z]*\b')

## hypernova2 ##
- name 그대로 유지
- 뒤에 물결이 붙는 경우 ("음~", "아~")
- 그, 뭐, 어, 인제, 막, 아, 음, 읍, 오, 으, 좀
- 단어가 두 번 이상 반복되는 경우 제거 re.sub(r'\b(\w+)\b(?:\s+\1\b)+', r'\1', text)
- x를 포함한 단어 제거 (r'\b[가-힣a-zA-Z]*[xX][가-힣a-zA-Z]*\b')
- 전처리 이후 빈 utterance 제거

## cosmos ##
- hypernova기반
- output의 맨 첫 번째 문장인 total summary 형식을 "두 화자는 이 대화에서"로 통일
- output 이상치 추가 수정 
    (train) train-000111 / train-000130, train-000030, train-000193 / train-000032, train-000418 / train-000020, train-000176
    (dev)   dev-000085, dev-000093 / dev-000074, dev-000093 / 


"""

stopwords_pattern = [r'\w~', r'\b으\b', r'\b그\b', r'\b뭐\b', r'\b어\b',  r'\b인제\b', r'\b이제\b', r'\b막\b', r'\b아\b', r'\b음\b', r'\b읍\b', r'\b오\b', r'\b으\b'] # r'name[0-9]\S*'

def remove_stopwords(text):
    # 커스텀 불용어 제거
    for pattern in stopwords_pattern:
        text = re.sub(pattern, '', text)
    
    # x를 포함한 단어 제거
    text = re.sub(r'\b[가-힣a-zA-Z]*[xX][가-힣a-zA-Z]*\b', '', text)

    # 단어가 두 번 이상 반복되는 경우 -> 1개로
    # text = re.sub(r'\b(\w)\s+\1\b', r'\1', text)
    text = re.sub(r'\b([가-힣a-zA-Z0-9_]+)\s+\1\b', r'\1', text)

    # 공백 두 번 이상 연속 -> 1개로
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.strip()
    
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