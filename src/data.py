
import json

import torch
from torch.utils.data import Dataset
from src.utils import text_preprocess, file_preprocess

class CustomDataset(Dataset):
    def __init__(self, fname, tokenizer):
        IGNORE_INDEX=-100
        self.inp = []
        self.label = []

        PROMPT = '''당신은 유능한 AI 어시스턴트 입니다. [대화 내용]과 [대화 키워드]를 보고, 한국어 대화 요약문을 생성해주세요.
        '''

        with open(fname, "r") as f:
            data = json.load(f)

        if fname.split('/')[-1].split('.')[0].split('_')[1] == "train":
            data = file_preprocess(data, is_train=True)
        else:
            data = file_preprocess(data, is_train=False)



        def make_chat(inp):
            chat = [f"[대화 키워드] : {', '.join(inp['subject_keyword'])}에 대한 대화 내용입니다.\n[대화 내용]"]
            for cvt in inp['conversation']:
                speaker = cvt['speaker']
                utterance = text_preprocess(cvt['utterance'])
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

class OriginalDataset(Dataset):
    def __init__(self, fname, tokenizer):
        IGNORE_INDEX=-100
        self.inp = []
        self.label = []

        PROMPT = '''당신은 유능한 AI 어시스턴트 입니다. [대화 내용]과 [대화 키워드]를 보고, 한국어 대화 요약문을 생성해주세요.
        '''

        with open(fname, "r") as f:
            data = json.load(f)

        if fname.split('/')[-1].split('.')[0].split('_')[1] == "train":
            data = file_preprocess(data, is_train=True)
        else:
            data = file_preprocess(data, is_train=False)


        def make_chat(inp):
            chat = [f"[대화 키워드] : {', '.join(inp['subject_keyword'])}에 대한 대화 내용입니다.\n[대화 내용]"]
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
