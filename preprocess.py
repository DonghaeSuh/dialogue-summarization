import json
import re


# Remove exclamation marks from the utterances
EXCLEMATIONS = ["음~","어~","아~","그~"]

def remove_exclamation(utterance):
    for exclamation in EXCLEMATIONS:
        utterance = re.sub(exclamation,'',utterance)
    return utterance.strip().replace('  ',' ')


def preprocess(data:json):
    """
    Preprocess the data to remove exclamation marks from the utterances
    """
    for example in data:
        for cvt in example['input']['conversation']:
            cvt['utterance'] = remove_exclamation(cvt['utterance'])
            print(cvt['utterance'])
    return data


def main():
    with open("resource/data/일상대화요약_train.json", "r",encoding='utf-8') as f:
        data = json.load(f)
    data = preprocess(data)
    with open("resource/data/일상대화요약_train.json", "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    with open("resource/data/일상대화요약_dev.json", "r",encoding='utf-8') as f:
        data = json.load(f)
    data = preprocess(data)
    with open("resource/data/일상대화요약_dev.json", "w",encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    with open("resource/data/일상대화요약_test.json", "r",encoding='utf-8') as f:
        data = json.load(f)
    data = preprocess(data)
    with open("resource/data/일상대화요약_test.json", "w",encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()