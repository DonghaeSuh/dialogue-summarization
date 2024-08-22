
import argparse
import json
import tqdm

import torch
import numpy
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType
from peft import PeftModel, PeftConfig, get_peft_model

from src.data_2 import CustomDataset


# fmt: off
parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--output", type=str, required=True, help="output filename")
g.add_argument("--model_id", type=str, required=True, help="huggingface model id")
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer")
g.add_argument("--device", type=str, required=True, help="device to load the model")
g.add_argument("--checkpoint_path", type=str, required=True, help="checkpoint path to inference the model")
# fmt: on


def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        low_cpu_mem_usage=True
    )

    if args.tokenizer == None:
        args.tokenizer = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token


    special_tokens_dict = {'additional_special_tokens': ['<A>', '<B>']}

    tokenizer.add_special_tokens(special_tokens_dict)

    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, args.checkpoint_path)
    model.to(dtype = torch.bfloat16)
    model = model.merge_and_unload()
    model.eval()
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    dataset = CustomDataset("resource/data/일상대화요약_test_add_spk.json", tokenizer)

    
    with open("resource/data/일상대화요약_test_add_spk.json", "r") as f:
        result = json.load(f)

    for idx in tqdm.tqdm(range(len(dataset))):
        inp = dataset[idx]

        outputs = model.generate(
            inp.to(args.device).unsqueeze(0),
            max_new_tokens=1024,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            # no_repeat_ngram_size=5,
            # num_beams=4
        )

        result[idx]["output"] = tokenizer.decode(outputs[0][inp.shape[-1]:], skip_special_tokens=True)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    exit(main(parser.parse_args()))