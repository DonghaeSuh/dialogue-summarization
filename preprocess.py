import json
import re
import argparse

# argument parser
parser = argparse.ArgumentParser(prog="preprocess", description="Preprocess the data.")

parser.add_argument("--path", type=str, default='resource/data',help="data file path")
parser.add_argument("--remove_exclamation", type=bool, default=True, help="remove exclamation marks from the utterances")
parser.add_argument("--remove_empty_utterance", type=bool, default=True, help="remove empty utterances") 
parser.add_argument("--correct_wrong_output", type=bool, default=True, help="correct wrong outputs")

"""
Example:

python preprocess.py --data_path resource/data --remove_exclamation True --remove_empty_utterance True --correct_wrong_output True
"""

# Remove exclamation marks from the utterances
EXCLEMATIONS = ["음~","어~","아~","그~"]


## Preprocess functions ##

def remove_exclamation(utterance):
    """
    Remove exclamation marks from the utterances
    """
    for exclamation in EXCLEMATIONS:
        utterance = re.sub(exclamation,'',utterance)
    return utterance.strip().replace('  ',' ')


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


## Preprocessing ##

def preprocess(args:argparse.Namespace):
    """
    Preprocess the data 
    """

    types = ['train','dev','test']

    for t in types:

        # Load the data
        with open(f"{args.path}/일상대화요약_{t}.json", "r",encoding='utf-8') as f:
            data = json.load(f)

        # Remove exclamation marks from the utterances
        if args.remove_exclamation:
            for example in data:
                for cvt in example['input']['conversation']:
                    cvt['utterance'] = remove_exclamation(cvt['utterance'])

            print(f"Exclamation marks removed from the utterances in {t} data...")

        # Remove empty utterances
        if args.remove_empty_utterance:
            data = remove_empty_utterance(data)

            print(f"Empty utterances removed from the {t} data...")

        # Correct wrong train outputs
        if args.correct_wrong_output and t == 'train':
            data = correct_wrong_output(data)

            print(f"Wrong outputs corrected in the {t} data...")

        # Save the data
        with open(f"{args.path}/일상대화요약_{t}.json", "w",encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"{t} data saved...",end='\n\n')


    return data

def main(args):
    preprocess(args)

if __name__ == "__main__":
    main(parser.parse_args())