import json
import re
import argparse

# argument parser
parser = argparse.ArgumentParser(prog="postprocess", description="Prostprocess the data.")

parser.add_argument("--path", type=str, default='result.json',help="data file path")
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
        output = re.sub(r'## ' + speaker_1 + ' 요약', '', output)
        output = re.sub(r'## ' + speaker_2 + ' 요약', '', output)
        output = re.sub(r'\s+', ' ', output)
        output = output.strip()

        example["output"] = output

    return data


def main(args):
    with open(args.path, 'r') as f:
        data = json.load(f)
    
    data = postprocess(data)
    
    with open(args.output_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)