
import argparse
import json
import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.data import CustomDataset
from peft import PeftModel

# fmt: off
parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--output", type=str, required=True, help="output filename")
g.add_argument("--model_id", type=str, default="MLP-KTLim/llama-3-Korean-Bllossom-8B", required=True,help="huggingface model id")
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer")
g.add_argument("--device", type=str, required=True, help="device to load the model")
g.add_argument("--adapter_checkpoint_path", type=str, help="model path where model saved")
g.add_argument("--top_p", type=int, help="top p parameter")
g.add_argument("--top_k", type=int, default=0, help="top k parameter")
g.add_argument("--temperature", type=int, default=1, help="temperature parameter")
# fmt: on


def main(args):
    model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
            low_cpu_mem_usage=True
    )

    model = PeftModel.from_pretrained(model, args.adapter_checkpoint_path)
    model = model.merge_and_unload()
    model.to(dtype = torch.bfloat16)
    model.eval()

    if args.tokenizer == None:
        args.tokenizer = args.model_id
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    dataset = CustomDataset("resource/data/일상대화요약_test.json", tokenizer)

    with open("resource/data/일상대화요약_test.json", "r") as f:
        result = json.load(f)

    for idx in tqdm.tqdm(range(len(dataset))):
        inp = dataset[idx]
        outputs = model.generate(
            inp.to(args.device).unsqueeze(0),
            max_new_tokens=1024,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p = args.top_p,
            top_k = args.top_k,
            temperature = args.temperature
        )
        result[idx]["input"] = dataset[idx]
        result[idx]["output"] = tokenizer.decode(outputs[0][inp.shape[-1]:], skip_special_tokens=True)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    exit(main(parser.parse_args()))