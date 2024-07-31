
import json
import os
import torch
from torch.utils.data import Dataset
from src.utils import text_preprocess, file_preprocess

# JSON 파일에 데이터를 한 줄씩 추가하는 함수
def save_to_json_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


class CustomDataset(Dataset):
    def __init__(self, fname, tokenizer):
        IGNORE_INDEX=-100
        self.inp = []
        self.label = []

        PROMPT = '''당신은 유능한 AI 어시스턴트 입니다. [대화 내용]과 [대화 키워드]를 보고, [요약문]을 생성해주세요.\n'''

        with open(fname, "r") as f:
            data = json.load(f)

        if fname.split('/')[-1].split('.')[0].split('_')[1] == "train":
            data = file_preprocess(data, is_train=True)
        else:
            data = file_preprocess(data, is_train=False)

        ID_FILE = []

        def make_chat(id, inp):
            chat = [f"[대화 키워드]\n{', '.join(inp['subject_keyword'])}에 대한 대화 내용입니다.\n[대화 내용]"]
            
            # json row로 저장
            # 먼저 나온 speaker를 A로 할당
            id_row = {"id" : id, "speaker_ids" : {inp['conversation'][0]['speaker'] : "<|A|>"}}

            for cvt in inp['conversation']:
                speaker_idx = cvt['speaker']

                # 2번째 발화자 추가 : 뒤에 나온 speaker를 B로 할당
                if speaker_idx not in id_row["speaker_ids"].keys():
                    id_row["speaker_ids"][speaker_idx] = "<|B|>"

                utterance = text_preprocess(cvt['utterance'])

                # 비어있는 문장 제거
                if len(utterance) == 0:
                    continue

                chat.append(f"{id_row['speaker_ids'][speaker_idx]}: {utterance}")
                
                

            chat = "\n".join(chat)

            # speaker dict를 json 파일에 저장
            ID_FILE.append(id_row)

            question = f"[요약문]\n"
            chat = chat + "\n\n" + question

            return chat
        
        for example in data:
            chat = make_chat(example["id"], example["input"])
            message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": chat},
            ]
     
            source = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            target = example["output"]
            if target != "":
                target += tokenizer.eos_token
            target = tokenizer(target,
                      return_attention_mask=False,
                      add_special_tokens=False,
                      return_tensors="pt")
            target["input_ids"] = target["input_ids"].type(torch.int64)

            input_ids = torch.concat((source[0], target["input_ids"][0]))
            labels = torch.concat((torch.LongTensor([IGNORE_INDEX] * source[0].shape[0]), target["input_ids"][0]))
            self.inp.append(input_ids)
            self.label.append(labels)
        
        save_to_json_file(os.path.join("resource/data/", f"ID_{fname.split('/')[-1]}"), ID_FILE)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return self.inp[idx]


class OriginalDataset(Dataset):
    def __init__(self, fname, tokenizer):
        IGNORE_INDEX=-100
        self.inp = []
        self.label = []

        # hypernova와 비슷한 형식, 살짝만 수정함 (07.31)
        PROMPT = '''당신은 유능한 AI 어시스턴트입니다. [대화 내용]을 보고, [대화 키워드]와 연관된 한국어 대화 요약문을 생성해주세요.\n'''

        with open(fname, "r") as f:
            data = json.load(f)

        if fname.split('/')[-1].split('.')[0].split('_')[1] == "train":
            data = file_preprocess(data, is_train=True)
        else:
            data = file_preprocess(data, is_train=False)


        def make_chat(inp):
            chat = [f"[대화 키워드]\n{', '.join(inp['subject_keyword'])}에 대한 대화 내용입니다.\n[대화 내용]"]
            for cvt in inp['conversation']:
                speaker = cvt['speaker']
                utterance = cvt['utterance']

                chat.append(f"{speaker}: {utterance}")

            chat = "\n".join(chat)

            question = f"[요약문]\n"
            chat = chat + "\n\n" + question

            return chat
        
        for example in data:
            chat = make_chat(example["input"])
            message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": chat},
            ]
     
            source = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            target = example["output"]
            if target != "":
                target += tokenizer.eos_token
            target = tokenizer(target,
                      return_attention_mask=False,
                      add_special_tokens=False,
                      return_tensors="pt")
            target["input_ids"] = target["input_ids"].type(torch.int64)

            input_ids = torch.concat((source[0], target["input_ids"][0]))
            labels = torch.concat((torch.LongTensor([IGNORE_INDEX] * source[0].shape[0]), target["input_ids"][0]))
            self.inp.append(input_ids)
            self.label.append(labels)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return self.inp[idx]


class DataCollatorForSupervisedDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(lbls) for lbls in labels], batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
