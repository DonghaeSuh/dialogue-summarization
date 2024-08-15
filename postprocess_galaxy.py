import json
import re
import argparse

# argument parser
parser = argparse.ArgumentParser(prog="postprocess", description="Prostprocess the data.")

parser.add_argument("--path", type=str, default='./results/result_galaxy.json',help="data file path")
parser.add_argument("--ensemble_path_1", type=str, default=None, help="ensemble data file path")
parser.add_argument("--ensemble_path_2", type=str, default=True, help="ensemble data file path")
parser.add_argument("--model_type", type=str, default='galaxy', help="model type")
parser.add_argument("--output_path", type=str, default='post_result.json',help="output file path")

"""
Example:

python postprocess_galaxy.py --data_path result --remove_special_characters True
"""

def postprocess(data):
    """
    1. Remove '## 전반적인 요약', '## speaker_2 요약', '## speaker_2 요약' from the data
    2. Concatenate the summaries into one string
    """
    for example in data:
        output = example["output"]
        speakers = set()
        for cvt in example["input"]["conversation"]:
            speakers.add(cvt["speaker"])
        speaker_1, speaker_2 = speakers

        output = re.sub(r'## 전반적인 요약', '', output)
        output = re.sub(r'# 전반적인 요약', '', output)
        output = re.sub(r'## ' + speaker_1 + ' 요약', '', output)
        output = re.sub(r'## ' + speaker_2 + ' 요약', '', output)
        output = re.sub(r'-', '', output)
        output = re.sub(r'\s+', ' ', output)
        output = output.strip()

        example["output"] = output

    return data

def ensemble_blackhole_115(data:json, data2_path:str):
    """
    ensemble two json data
    
    exact_repeated_sentences_indexes : {117, 219}
    - [Replace] 117, 219
    
    < Replace >
    [117] -> galaxy
    [219] -> galaxy
    """
    # Load data2
    with open(data2_path, 'r') as f:
        data2 = json.load(f)

    # Replace
    data[117]["output"] = data2[117]["output"]
    data[219]["output"] = data2[219]["output"]
    
    return data


def ensemble_blackhole(data:json, data2_path:str, data3_path:str):
    """
    ensemble two json data
    
    exact_repeated_sentences_indexes : {92, 117, 210, 328, 369, 378}
    - [Replace] 117, 210, 328, 369, 378
    - [Make one] 92
    - [no change] 

    < Replace >
    [117] -> galaxy
    [210] -> galaxy
    [328] -> galaxy
    [369] -> galaxy
    [378] -> exaome

    < Make one >
    [92] -> 1st sentence
    
    """
    def make_duplicatation_one(indexes: set, data: json):
        """
        Make the duplicated sentences one.
        """
        for idx in indexes:
            sentences = data[idx]['output'].split('.')[:-1]
            seen = set()
            unique_sentences = []

            for sentence in sentences:
                if sentence not in seen:
                    seen.add(sentence)
                    unique_sentences.append(sentence)

            data[idx]['output'] = '.'.join(unique_sentences) + '.'
    
        return data
    
    # Load data2, data3
    with open(data2_path, 'r') as f:
        data2 = json.load(f)
    with open(data3_path, 'r') as f:
        data3 = json.load(f)

    # Replace
    data[117]["output"] = data2[117]["output"]
    data[210]["output"] = data2[210]["output"]
    data[328]["output"] = data2[328]["output"]
    data[369]["output"] = data2[369]["output"]
    data[378]["output"] = data3[378]["output"]

    # Make one
    data = make_duplicatation_one({92}, data)
    
    return data

def ensemble_galaxy(data:json, data2_path:str):
    """
    ensemble two json data

    exact_repeated_sentences_indexes : {43, 92, 125, 173, 231, 293, 312}
    - [Replace] 43, 231, 378
    - [Make one] 92, 312(맨 뒤 2 문장 제거)
    - [no change] 125, 173, 293

    high_similarity_repeated_sentences_indexes : {66, 68, 261, 234, 43, 331, 14, 369, 212, 214, 249, 218, 219}

    
    < Replace >
    [43] -> cosmos25
    [231] -> cosmos25
    [378] -> cosmos25

    < Make one >
    [92] -> 1st sentence
    [312] -> remove last 2 sentences
    """
    def make_duplicatation_one(indexes: set, data: json):
        """
        Make the duplicated sentences one.
        """
        for idx in indexes:
            sentences = data[idx]['output'].split('.')[:-1]
            seen = set()
            unique_sentences = []

            for sentence in sentences:
                if sentence not in seen:
                    seen.add(sentence)
                    unique_sentences.append(sentence)

            data[idx]['output'] = '.'.join(unique_sentences) + '.'
    
        return data

    # Load data2
    with open(data2_path, 'r') as f:
        data2 = json.load(f)

    # Replace
    data[43]["output"] = data2[43]["output"]
    data[231]["output"] = data2[231]["output"]
    data[378]["output"] = data2[378]["output"]

    # Make one
    data = make_duplicatation_one({92}, data)
    data[312]["output"] = '.'.join(data[312]["output"].split('.')[:-3]) + '.'

    return data


def main(args):
    with open(args.path, 'r') as f:
        data = json.load(f)
    
    data = postprocess(data)

    if args.ensemble_path_1:
        if args.model_type == 'galaxy':
            data = ensemble_galaxy(data, args.ensemble_path_1)
        elif args.model_type == 'blackhole':
            data = ensemble_blackhole(data, args.ensemble_path_1, args.ensemble_path_2)
        elif args.model_type == 'blackhole_115':
            data = ensemble_blackhole_115(data, args.ensemble_path_1)
    
    with open(args.output_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)