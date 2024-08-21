import evaluate
import numpy as np
import re
import json
from transformers import EvalPrediction, AutoTokenizer
import torch
import random
from typing import Dict
from tqdm import tqdm
import pickle

## 바꿔줘야 해!!
config_name = "config_bllossom.json"
print("## utils Config File : ", config_name)
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
def correct_wrong_output(data:json, path:str):
    """
    1. Correct wrong speakers in outputs of train samples 'train-000401', 'train-000402, 'train-000111'
    2. Add dot(.) at the end of the last sentence in outputs of train samples 'train-000130'
    3. Replace speaker name 'SSD' with 'SD' in outputso of 'train-000030', 'train-000193' and 'dev-000085'
    4. Remove duplicate sentences in outputs of dev samples 'dev-000093'.
    5. Change '말했습니다,' to '말했습니다.' in outputs of train samples 'train-000044'
    """
    if 'train' in path:
        # Correct wrong speakers
        data[400]['output'] = data[400]['output'].replace('SD2100504','SD2110504')
        data[401]['output'] = data[401]['output'].replace('SD2110503','SD2100503')
        data[110]['output'] = data[110]['output'].replace('SD20010813','SD2001083')
        # Add dot(.) at the end of the last sentence
        data[129]['output'] = data[129]['output'] + '.'
        # Replace speaker name
        data[29]['output'] = data[29]['output'].replace('SSD', 'SD')
        data[192]['output'] = data[192]['output'].replace('SSD', 'SD')
        # Change '말했습니다,' to '말했습니다.'
        data[43]['output'] = data[43]['output'].replace('말했습니다,','말했습니다.')


    elif 'dev' in path:
        # Replace speaker name
        data[84]['output'] = data[84]['output'].replace('SSD', 'SD')
        # Remove duplicate sentences
        data[92]['output'] = '.'.join(data[92]['output'].split('.')[1:]).strip()

    return data


# total summary(output의 맨 첫 번째 문장) 형식 통일을 위한 이상치 output 처리
def change_weird_output(data:json, path:str):
    """
    Standardize the type of the output of train-000032, train-000418, dev-000074, dev-000093
    """
    # Standardize the type of outputs
    if 'train' in path:
        # train-000032 : total_summary 교체
        output = data[31]['output'].split('.')
        total_summary = "두 화자는 이 대화에서 진로 관련 고민에 대해 이야기했습니다. "
        data[31]['output'] = total_summary + '.'.join(output[1:])

        # train-000418 : total_summary 추가
        total_summary = "두 화자는 이 대화에서 다이어트에 대해 이야기했습니다. "
        data[417]['output'] = total_summary + data[417]['output']

    elif 'dev' in path:
        # dev-000074 : total_summary 수정
        data[73]['output'] = "두 화자는 "+ data[73]['output'] # 이 대화에서 -> 두 화자는 이 대화에서

        # dev-000093 : total_summary 추가
        total_summary = "두 화자는 이 대화에서 엔시티와 방탄소년단에 대해 이야기 했습니다. "
        data[92]['output'] = total_summary + data[92]['output']
    
    return data



# output에 SD가 예외적으로 들어간 경우 처리
def remove_sd_in_total_summary(data:json, path:str):
    """
    Remove 'SD' in total_summary of train-000020 and train-000176
    """
    if 'train' in path:
        # train-000020 : total_summary 수정
        data[19]['output'] = data[19]['output'].replace('SD2000039의 꿈인 ','')

        # train-000176 : total_summary '.' 가 빠져있던 것을 수정
        output = data[175]['output']
        data[175]['output'] = re.sub(r'(장단점에 대해 말했습니다)\s+(SD\d{7}(?:은|는))', r'\1. \2', output)

    return data


# utterance와 output에서는 '.' 뒤에 공백이 무조건 존재하는 형태로 통일 / 문장 맨 마지막의 경우는 '.'으로 통일
def add_space_after_period_and_remove_control_characters(data:json, path:str):
    """
    Add space after period if there is no space after period
    text = re.sub(r'\.(?=\S)', '. ', text)
    """
    # Add space after period in utterances
    for example in data:
        example['input']['conversation'] = [{'speaker': cvt['speaker'], 'utterance': re.sub(r'\.(?=\S)', '. ', cvt['utterance']).strip()} for cvt in example['input']['conversation']]

    if 'train' or 'dev' in path:
        # Remove_control_characters and Add space after period in outputs
        for example in data:
            output = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', example['output'])
            example['output'] = re.sub(r'\.(?=\S)', '. ', output).strip()

    return data


# total summary(output의 맨 첫 번째 문장) 형식을 "두 화자는 이 대화에서"로 통일
def total_summary_generalization(data:json, path:str):
    """
    Standardize the format of the total summary in the first sentence of the output 
    to start with "두 화자는 이 대화에서".
    """
    types = ["두 화자는", "화자들은" ,"두 사람은", "이 대화에서는"] # "두 화자는 이 대화에서"
    types2 = r"SD\d{7}(?:와|과).*SD\d{7}(?:은|는)"

    if 'train' or 'dev' in path:
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


# output의 형식 통일 후, 중복 단어 제거
def remove_duplicate_output_words(data:json, path:str):
    """
    Remove duplicate words in outputs of train samples 
    (그리고 그리고) 'train-000387', 
    (대화에서 대화에서) 'train-000383', 'train-000451', 'train-000479', 'train-000495'
    (좋은 좋은) 'train-000268'
    (화자 화자) 'train-000092', 'train-000231'
    (할머니가 할머니가) 'train-000128'
    (가도 가도) 'train-000338'
    """
    if 'train' in path:
        # Remove duplicate words
        ids = [387, 383, 451, 479, 495, 268, 92, 231, 128, 338]
        for id in ids:
            output = data[id-1]['output']
            output = re.sub(r'\b(\w+)\b(?:\s+\1\b)+', r'\1', output)
            data[id-1]['output'] = output

    return data


# stopword로 제거하기 전, 예외적인 경우 처리
def remove_stopwords_exception(data:json, path:str):
    """
    manual exception handling for removing stopwords in utterances
    (" 좋 ") : train과 dev에서는 의미없게 단어 사이에 추가된 단어이지만, test에서는 의미있는 단어로 사용되는 경우(좋 은데, 좋 을 것)가 있음
        ex) 'test-000119' : "좋 은데" -> "좋은데"
            'test-000303' : "좋 을 것" -> "좋을 것"
            'test-000348' : "좋 다고" -> "좋다고"
    """
    if 'test' in path:
        # " 좋 " -> " 좋"
        data[118]['input']['conversation'][-1]['utterance'] = data[118]['input']['conversation'][-1]['utterance'].replace(' 좋 ', ' 좋')
        data[302]['input']['conversation'][-2]['utterance'] = data[302]['input']['conversation'][-2]['utterance'].replace(' 좋 ', ' 좋')
        data[347]['input']['conversation'][4]['utterance'] = data[347]['input']['conversation'][4]['utterance'].replace(' 좋 ', ' 좋')

    return data


# SD\d{7} 앞에 '화자' 제거
def remove_hwaja_before_speaker_in_output(data:json, path:str):
    """
    Remove '화자' before 'SD\d{7}' in outputs of train samples
    """
    if 'train' in path:
        for example in data:
            output = example['output']
            output = re.sub(r'화자\s*(SD\d{7})', r'\1', output)
            example['output'] = output

    return data


# SD\d{7} 뒤에 아무런 조사가 붙지 않은 경우 수정
def add_josa_after_speaker_in_output(data:json, path:str):
    """
    <Train>
    - train-243 : SD2002060 또한 -> SD2002060도
    - train-410 : 또 SD2100516 자신은 -> 또 자신은
    - train-441 :  SD2110545 유기견을 -> 또 유기견을 / 또 SD2100546은 -> SD2100546은
    - train-495 :  SD2100589에도 -> SD2100589에게도 / SD2100589 헬스장 -> SD2100589에게 헬스장
    """
    if 'train' in path:
        data[242]['output'] = data[242]['output'].replace('SD2002060 또한', 'SD2002060도')
        data[409]['output'] = data[409]['output'].replace('또 SD2100516 자신은', '또 자신은')
        data[440]['output'] = data[440]['output'].replace('SD2110545 유기견을', '또 유기견을').replace('또 SD2100546은', 'SD2100546은')
        data[494]['output'] = data[494]['output'].replace('SD2100589에도', 'SD2100589에게도').replace('SD2100589 헬스장', 'SD2100589에게 헬스장')
        
    return data


# speaker output 형식 통일
def speaker_summary_generalization(data:json, path:str):
    """
    Standardize the format of the speaker summary in the first sentence of the output 
    to start with "SD\d{7}은(는)".
    """
    if 'train' in path:
        # exception handling 
        # train-000496 "SD2100589가" -> "SD2100589는"
        # train-000476 "SD2100573도" -> "SD2100573은"
        data[495]['output'] = data[495]['output'].replace('SD2100589가', 'SD2100589는')
        data[475]['output'] = data[475]['output'].replace('SD2100573도', 'SD2100573은')
    

    def check_first_speaker_and_first_summary_speaker_is_same(example:json) -> bool:
        """
        Check if the first speaker and the first speaker summary speaker are the same.
        """
        first_speaker = example['input']['conversation'][0]['speaker']
        first_summary_speaker = re.search(r'SD\d{7}', example['output']).group()
        return first_speaker == first_summary_speaker

    def make_speaker_summaries_bullet_point_format(text:str) -> str:
        """
        Make the speaker summaries in bullet point format.
        """
        output = "\n".join([f"- {sentence.strip()}. " for sentence in text.split('.') if sentence.strip()])
        return output


    def find_split_indexes(text: str) -> list[tuple]:
        """
        Find the indexes(strat, end) to split the structured summary.
        """
        # The number of 'SD{7}[은는]{1}'
        num_speakers = len(re.findall(r'SD\d{7}[은는]{1}', text))

        # Split the structured summary based on the number of 'SD{7}[은는]{1}'
        if num_speakers == 2: 
            mathes = re.finditer(r'SD\d{7}[은는]{1}', text)
            return [(match.group(), match.start()) for match in mathes] # [(speaker1, start_id_1), (speaker2, start_id_2)]
        
        elif num_speakers in [0, 1]:
            matches = re.finditer(r'SD\d{7}\w+', text)

            first_match = next(matches)
            first_tuple = (first_match.start(), first_match.group())

            for match in matches:
                if match.group()[:9] == first_tuple[1][:9]: # SD{7}가 같은 경우
                    continue
                return [(first_tuple[1], first_tuple[0]), (match.group(), match.start())]
            
        elif num_speakers >= 3:
            matches = re.finditer(r'SD\d{7}[은는]{1}', text)

            first_match = next(matches)
            first_tuple = (first_match.start(), first_match.group())

            for match in matches:
                if match.group()[:9] == first_tuple[1][:9]: # SD{7}가 같은 경우
                    continue
                return [(first_tuple[1], first_tuple[0]), (match.group(), match.start())]
            

    if 'test' in path:
        for example in data:
            # Find speaker_1 and speaker_2
            speaker_1 = example['input']['conversation'][0]['speaker']

            for speaker in example['input']['conversation']:
                if speaker['speaker'] != speaker_1:
                    speaker_2 = speaker['speaker']
                    break
                
            example['input']['speaker_1'] = speaker_1
            example['input']['speaker_2'] = speaker_2

    elif 'train' or 'dev' in path:
        for example in data:
            output = example['output']

            # Find the indexes to split the structured summary
            split_indexes = find_split_indexes(output) # [(r'speaker1\w+', start_id_1), (r'speaker2\w+', start_id_2)]
            speaker_1, speaker_2 = split_indexes[0][0][:9], split_indexes[1][0][:9] # SD{7}

            # Split the structured summary
            total_summary = output[:split_indexes[0][1]].strip()
            if check_first_speaker_and_first_summary_speaker_is_same(example):
                # The first speaker and the first speaker summary speaker are the same
                example['input']['speaker_1'] = speaker_1
                example['input']['speaker_2'] = speaker_2

                speaker_1_summary = output[split_indexes[0][1]:split_indexes[1][1]].strip()
                speaker_2_summary = output[split_indexes[1][1]:].strip()
            else:
                # The first speaker and the first speaker summary speaker are different
                speaker_1, speaker_2 = speaker_2, speaker_1 # Swap the speakers
                example['input']['speaker_1'] = speaker_1
                example['input']['speaker_2'] = speaker_2

                speaker_1_summary = output[split_indexes[1][1]:].strip()
                speaker_2_summary = output[split_indexes[0][1]:split_indexes[1][1]].strip()


            # speaker_1_summary = make_speaker_summaries_bullet_point_format(speaker_1_summary)
            # speaker_2_summary = make_speaker_summaries_bullet_point_format(speaker_2_summary)

            # Standardize the format of the speaker summary
            output_format = f'''## 전반적인 요약\n{total_summary}\n\n## {speaker_1} 요약\n{speaker_1_summary}\n\n## {speaker_2} 요약\n{speaker_2_summary}'''
            
            example['output'] = output_format
    
    return data


# subject_keyword 반복 단어 제거
def remove_duplicate_subject_keywords(data:json, path:str):
    '''
    Remove duplicate words in subject_keywords of dev samples 'dev-000045', 'dev-000086', 'dev-000087', 'dev-000088', 'dev-000089'
    '''
    if 'dev' in path:
        # Remove duplicate words
        ids = [44, 85, 86, 87, 88]
        for id in ids:
            subject_keywords = data[id]['input']['subject_keyword']
            subject_keywords = list(set(subject_keywords))
            data[id]['input']['subject_keyword'] = subject_keywords

    return data


# speaker1의 utterance 개수가 50개가 넘는 샘플 제거
def remove_samples_with_more_than_50_utterances(data:json, path:str):
    """
    Remove the samples with more than 50 utterances in speaker1
    """
    if 'train' in path:
        # Remove the samples with more than 50 utterances in speaker1
        # indexes = [6, 310, 311, 323, 324, 339, 349, 358, 359, 362, 413, 444, 460, 461, 492] # from utterance_length_eda.ipynb
        
        indexes = [6, 310, 311, 323, 324, 339, 349, 358, 359, 362, 413, 444, 460, 461, 492]
        data = [data[i] for i in range(len(data)) if i not in indexes]

    return data 


# 반복되는 단어 조합 제거
def make_one_repeated_words(data:json, path:str, iter:int=0):
    """
    Replace the repeated words in the text with one word.

    Parameters:
    data (json): Data to be processed.
    path (str): Path to save the processed data.

    Returns:
    data (json): Processed data.
    """
    
    # Function for removing repeated words
    def removeing_repeated_words(data:json, repeated_phrase_indices:dict, mode:str):
    # repeated_phrase_indices = {key : index, value : repeated phrase}

        for idx in tqdm(repeated_phrase_indices.keys(), total=len(repeated_phrase_indices), desc=f'Removing repeated phrases in {mode} data ... (Phase {iter})'):
            repeated_phrases = repeated_phrase_indices[idx]
            for phrase in repeated_phrases:
                pattern = rf'\b{phrase} {phrase}'
                try:
                    for i, turn in enumerate(data[idx]['input']['conversation']):
                        if re.search(pattern, turn['utterance']):
                            data[idx]['input']['conversation'][i]['utterance'] = re.sub(pattern, phrase, turn['utterance'])
                except:
                    pass

    # Remove repeated words in the conversation
    if 'train' in path:
        with open(f'./src/data/train_repeated_phrase_indices_{iter}.pkl', 'rb') as file:
            repeated_phrase_indices = pickle.load(file)
        
        removeing_repeated_words(data, repeated_phrase_indices, mode='train')

    elif 'dev' in path:
        with open(f'./src/data/dev_repeated_phrase_indices_{iter}.pkl', 'rb') as file:
            repeated_phrase_indices = pickle.load(file)
        removeing_repeated_words(data, repeated_phrase_indices, mode='dev')

    elif 'test' in path:
        with open(f'./src/data/test_repeated_phrase_indices_{iter}.pkl', 'rb') as file:
            repeated_phrase_indices = pickle.load(file)
        removeing_repeated_words(data, repeated_phrase_indices, mode='test')

    return data

# # 좀 삭제
# def remove_jom(data:json, path:str):
#     """
#     Remove '좀' in utterances and outputs
#     """

#     # Remove '좀' in utterances and outputs
#     for example in data:
#         example['output'] = re.sub(r'\b좀\b', '', example['output'])
#         # '  ' -> ' '
#         example['output'] = re.sub(r'\s+', ' ', example['output']).strip()

#     return data


# 전반적인 요약 문장이 2개인 경우(506개 중 4개) 두 번째 요약 문장을 제거
def remove_second_summary(data:json, path:str):
    """
    Remove the second summary in the output if there are two summaries in the output
    """

    indexes = [59, 63, 261, 475]
    
    if 'train' in path:
        for idx in indexes:
            output = data[idx]['output']
            output = output.split('.')
            data[idx]['output'] = output[0] + '.' + '.'.join(output[2:])

    return data


# 이어지는 다음 턴 속 반복 문장 제거
def remove_repeated_sentences_in_next_turn(data:json, path:str):
    """
    Remove repeated sentences in the conversation.
    [train]
    {16: ['turn : 1   (current ✓)  prev: 앞으로 먹어 보고 싶은 있 먹 | current: 앞으로 먹어 보고 싶은 먹거리가 있나요?'], 
    162: ['turn : 3   (current ✓)  prev: 그러니까 결혼 생활 중에서 가장 행복했었었던 때 | current: 결혼 생활 중에서 가장 행복했던 때'],
    169: ['turn : 9   (prev ✓)  prev:  이게 뭔가 이게 나중에 어르신들 했을 때 이게 참 이게 뭐가 잘되게 이게 융합 맞은 맞을 거 같아 | current: 이게 뭐가 잘되게 이게 융합 맞은 맞을 거 같아'],
    278: ['turn : 19  (current ✓)   prev: 안 먹으면 이제는 힘이 없으니까 말이 안 나올 거 같고 그냥 | current: 안 먹으면 이제는 힘이 없으니까 말이 안 나올 거 같고 그냥 적게 먹는 게 다이어트 하는 방법인 거 같아요'],
    358: ['turn : 64  (current ✓)  prev: 직장에서? | current: 오빠 직장에서?'],
    381: ['turn : 10  (prev ✓)   prev: 내나 그런 느낌이지 않아? | current: 그런 느낌이지'],
    383: ['turn : 2   (current ✓)  prev: 백두산? | current: 갑자기 백두산?'],
    505: ['turn : 17  (prev ✓)   prev: company-name3 집은 아직 열고 있긴 한데 건너편에 재개발이 되다 보니까 상권들이 다 안 좋아져서 많이들 맛집들이 문을 닫으려고 하는 거 같아 | current: 아직 열고 있긴 한데 건너편에 재개발이 되다 보니까 상권들이 다 안 좋아져서 많이들 맛집들이 문을 닫으려고 하는 거 같아']}
    
    [test]
    {198: ['turn : 16 (prev ✓)   prev: 정말로 뭔가 평이하게 큰소리 한번 나지 않고 그렇게 그런 환경에서 자랄 수 있던 것이 정말로 감사하고 컸다라는 것을 알아가게 되는 것 같습니다 | current: 정말로 뭔가 평이하게 큰소리 한번 나지 않고 그렇게 그런 환경에서 자랄 수 있던 것이 정말로 감사하고 컸다라는 것을 알아가게 되는 것 같습니다'],
    218: ['turn : 15  (current ✓)   prev:  제목이 | current: 제목이 뭐야?'],
    331: ['turn : 3   (current ✓)  prev:  진짜 습하면은 | current: 습하면은 진짜 아무것도 못하겠는 거예요'],
    372: ['turn : 3   (prev ✓)  prev:  당연히 직접 먹는 걸 좋아합니다 | current: 먹는 걸 좋아합니다']}
    """
    train_indexes_and_turns = [(16,1,'current'), 
                             (162,3,'current'),
                             (169,9, 'prev'),
                             (278,19,'current'),
                             (358,64,'current'),
                             (381,10,'prev'),
                             (383,2,'current'),
                             (505,17,'prev')]
    
    test_indexes_and_turns = [(198,16,'prev'),
                            (218,15,'current'),
                            (331,3,'current'),
                            (372,3,'prev')]

    def change_sent(data, indexes_and_turns):
        for idx, turn, mode in indexes_and_turns:
            if mode == 'prev':
                # Remain the previous turn and remove the current turn's first sentence
                current_utterance = data[idx]['input']['conversation'][turn]['utterance']
                if '.' in current_utterance:
                    data[idx]['input']['conversation'][turn]['utterance'] = re.sub(r'^[^.]*\.', '', data[idx]['input']['conversation'][turn]['utterance'])
                else:
                    data[idx]['input']['conversation'][turn]['utterance'] = ''
            elif mode == 'current':
                # Remain the current turn and remove the previous turn's last sentence
                prev_utterance = data[idx]['input']['conversation'][turn-1]['utterance']
                if '.' in prev_utterance:
                    data[idx]['input']['conversation'][turn-1]['utterance'] = '.'.join(prev_utterance.split('.')[:-2]) + '.'
                else:
                    data[idx]['input']['conversation'][turn-1]['utterance'] = ''
        
        return data

    if 'train' in path:
        data = change_sent(data, train_indexes_and_turns)
        
    elif 'test' in path:
        data = change_sent(data, test_indexes_and_turns)

    return data 
    
    
# 의미 없는 " 예.", " 네.", " 응." 제거
def remove_useless_word(data:json, path:str):
    """
    Remove the meaningless words in the conversation. such as " 예.", " 네.", " 응." -> " "
    """
    # Remove the meaningless words in the conversation

    if 'train' or 'dev' or 'test' in path:
        for example in data:
            for i, turn in enumerate(example['input']['conversation']):
                if re.search(r' (예|네|응)\.\s*', turn['utterance']):
                    example['input']['conversation'][i]['utterance'] = re.sub(r' (예|네|응)\.\s*', ' ', turn['utterance'])
    
    return data


# name 토큰 전처리
def name_token_preprocessing(data:json, path:str):
    """
    Preprocess the name tokens in the text.
    """

    # Exception handling for name tokens
    # dev-000078 :'name3 씨와 name2 씨를' -> 'name3와 name2를'
    if 'dev' in path:
        data[77]['input']['conversation'][10]['utterance'] = data[77]['input']['conversation'][10]['utterance'].replace('name3 씨와 name2 씨를', 'name3와 name2를')

    return data

def file_preprocess(data:json, path:str):
    """
    [Preprocess the data]
    - remove_stopwords_exception
        : manual exception handling for removing stopwords in utterances

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
    
    - remove_duplicate_output_words
        : remove duplicate words in outputs of train samples

    - remove_hwaja_before_speaker_in_output
        : remove '화자' before 'SD\d{7}' in outputs of train samples

    - add_josa_after_speaker_in_output
        : add josa after speaker in outputs of train samples
    
    - speaker_summary_generalization
        : standardize the format of the speaker's summary of the output

    - remove_duplicate_subject_keywords
        : remove duplicate words in subject_keywords of dev samples

    - remove_samples_with_more_than_50_utterances
        : remove the samples with more than 50 utterances in speaker1

    - make_one_repeated_words
        : replace the repeated words in the text with one word
    """
    print("file_preprocess start ...")
    data = remove_stopwords_exception(data, path)
    data = correct_wrong_output(data, path)
    data = change_weird_output(data, path)
    data = remove_sd_in_total_summary(data, path)
    data = add_space_after_period_and_remove_control_characters(data, path)
    data = total_summary_generalization(data, path)
    data = remove_second_summary(data, path) 
    data = remove_duplicate_output_words(data, path)
    data = remove_hwaja_before_speaker_in_output(data, path)
    data = remove_useless_word(data, path)
    data = add_josa_after_speaker_in_output(data, path)
    data = speaker_summary_generalization(data, path)
    data = remove_duplicate_subject_keywords(data, path)
    # data = remove_samples_with_more_than_50_utterances(data, path)
    data = remove_empty_utterance(data)
    data = remove_repeated_sentences_in_next_turn(data, path)
    data = name_token_preprocessing(data, path)
    data = text_preprocess(data)
    data = make_one_repeated_words(data, path, iter=0)
    data = make_one_repeated_words(data, path, iter=1)
    
    # data = remove_jom(data, path)

    return data


"""
불용어 처리

## hyperstella2 ##
- name1, name2..
- 뒤에 물결이 붙는 경우 ("음~", "아~")
- 그, 뭐, 어, 인제, 막, 아, 음, 읍, 오, 으
- 한 글자가 두번 반복되는 경우 ("또 또", "그 그")


## nova ##
- name 그대로 유지
- 뒤에 물결이 붙는 경우 ("음~", "아~")
- 그, 뭐, 어, 인제, 막, 아, 음, 읍, 오, 으
- 단어가 두 번 반복되는 경우 제거 ( r'\b([가-힣a-zA-Z0-9_]+)\s+\1\b')


## nova3, hypernova ##
- name 그대로 유지
- 뒤에 물결이 붙는 경우 ("음~", "아~")
- 그, 뭐, 어, 인제, 막, 아, 음, 읍, 오, 으
- 단어가 두 번 반복되는 경우 제거 ( r'\b([가-힣a-zA-Z0-9_]+)\s+\1\b')
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
    (train) train-000130, train-000030, train-000193 / train-000032, train-000418 / train-000020, train-000176
    (dev)   dev-000085, dev-000093 / dev-000074, dev-000093 / 

## cosmos2 ##
- cosmos기반
- output 형식 통일 이후 output 속 중복단어 제거
- 의미없이 끼어있는 ' 좋 ', ' 크 ', ' 스 '
- '. .' 제거
- 문장 맨 앞 '. ' or ' . ' 제거
- output 속 SD\d{7} 앞에 '화자' 제거 <- 입력으로 'SD\d{7} : utterance' 형태로 들어가기 때문에

## galaxy ##
- cosmos2기반
- output의 speaker summary 형식 통일 (+ utterance 시작 speaker == speaker summary 시작 speaker)
- dev의 subject_keyword 중복 단어 제거

## galaxy2 ##
- galaxy기반
- speaker summary 형식을 bullet point로 변경

## blackhole (짱짱미녀 서연, 너무예뻐 서연) ##
- galaxy 기반
- 반복되는 단어 조합 제거

## blackhole2 ##
- blackhole 기반
- 반복되는 단어 조합 제거 Version 2

## blackhole3 ##
- blackhole 기반
- 좀 제거(utterance, output)

"""


def remove_stopwords(text):

    stopwords_pattern = stopwords_pattern = [r'\w~', r'\b으\b', r'\b그\b', r'\b뭐\b', r'\b어\b',  r'\b인제\b', r'\b이제\b', r'\b막\b', r'\b아\b', r'\b음\b', r'\b읍\b', r'\b오\b', 
    r'\b으\b', r'좋 ', r'\b크\b', r'\b스\b', r'\. \.', r'^\s*\.\s{1}',r'\b하\b', r'\b예\b']#, r'\b좀\b'] # r'name[0-9]\S*'

    # 커스텀 불용어 제거
    for pattern in stopwords_pattern:
        text = re.sub(pattern, '', text)
    
    # x를 포함한 단어 제거
    text = re.sub(r'\b[가-힣a-zA-Z]*[xX][가-힣a-zA-Z]*\b', '', text)

    # 단어가 두 번 이상 반복되는 경우 -> 1개로
    # text = re.sub(r'\b(\w)\s+\1\b', r'\1', text)
    # text = re.sub(r'\b([가-힣a-zA-Z0-9_]+)\s+\1\b', r'\1', text)
    text = re.sub(r'\b(\w+)\b(?:\s+\1\b)+', r'\1', text)

    # 공백 두 번 이상 연속 -> 1개로
    text = re.sub(r'\s{2,}', ' ', text)

    # 간단한 후처리
    text = text.strip()
    
    return text

# stopwords + 반복 어구 제거
def text_preprocess(data):
    print("text_preprocess start ...")
    
    # Remove stopwords
    for example in data:
        for cvt in example['input']['conversation']:
            cvt['utterance'] = remove_stopwords(cvt['utterance'])
    
    print("text_preprocess end ...")
    return data



def set_seed(config: Dict):
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)