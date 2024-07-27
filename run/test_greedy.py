
import argparse
import json
import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys, os, os.path

# 프로젝트의 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.data import CustomDataset
from peft import PeftModel

from typing import Dict

def main(config: Dict):
    # fmt: off
    parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

    g = parser.add_argument_group("Common Parameter")
    g.add_argument("--output_file_name", type=str, default=config["path"]["output_file_name"], help="output filename")
    g.add_argument("--predict_path", type=str, default=config["path"]["predict_path"], help="predict file path")
    g.add_argument("--model_id", type=str, default=config["arch"]["model_id"], help="which model to use")
    g.add_argument("--device", type=str, default=config["device"], help="device to load the model")
    g.add_argument("--adapter_checkpoint_path", type=str, default=config["path"]["adapter_checkpoint_path"], help="model path where model saved")
    g.add_argument("--do_sample", type=str, default=config["inference"]["do_sample"], help="do_sample setting")
    g.add_argument("--num_beams", type=str, default=config["inference"]["num_beams"], help="num_beams setting")
    g.add_argument("--temperature", type=str, default=config["inference"]["temperature"], help="temperature setting")
    g.add_argument("--top_k", type=str, default=config["inference"]["top_k"], help="top_k setting")
    g.add_argument("--top_p", type=str, default=config["inference"]["top_p"], help="top_p setting")
    g.add_argument("--no_repeat_ngram_size", type=str, default=config["inference"]["no_repeat_ngram_size"], help="no_repeat_ngram_size setting")
    g.add_argument("--is_test", type=bool, default=True, is_required=True, help="dev or test data")
    

    args = parser.parse_args()
    if args.is_test == False:
        print("## Dev inference mode enabled!! ##")
        OUTPUT_FILE_PATH = os.path.join("results/", "dev_" + args.output_file_name)
        dataset = CustomDataset("resource/data/일상대화요약_dev.json", tokenizer)
        print(f"## Dataset length : {len(dataset)} ##")
        with open("resource/data/일상대화요약_dev.json", "r") as f:
            result = json.load(f)
    else:
        print("## Test inference mode ##")
        OUTPUT_FILE_PATH = os.path.join("results/", args.output_file_name)
        dataset = CustomDataset("resource/data/일상대화요약_test.json", tokenizer)
        print(f"## Dataset length : {len(dataset)} ##")
        with open("resource/data/일상대화요약_test.json", "r") as f:
            result = json.load(f)

    print("## model id :", args.model_id)
    print("## adapter_checkpoint_path :", args.adapter_checkpoint_path)
    if os.path.exists(OUTPUT_FILE_PATH):
        raise ValueError("Wrong output name! File already Exitsts!")

    print("## output_file_path :", OUTPUT_FILE_PATH)

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

    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    for idx in tqdm.tqdm(range(len(dataset))):
        inp = dataset[idx]
        outputs = model.generate(
            inp.to(args.device).unsqueeze(0),
            max_new_tokens=1024,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=args.do_sample,
            num_beams=args.num_beams,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            no_repeat_ngram_size=args.no_repeat_ngram_size
        )
        if args.is_test == True:
            # 제출용 : output에 그대로 덮어 씌우기
            result[idx]["output"] = tokenizer.decode(outputs[0][inp.shape[-1]:], skip_special_tokens=True)
        else:
            # dev 테스트용 : inference 별도로 빼서 label과 비교
            result[idx]["inference"] = tokenizer.decode(outputs[0][inp.shape[-1]:], skip_special_tokens=True)

        # 테스트용 예시 10개 뽑기
        if args.is_test == True and idx == 10:
            with open(os.path.join("results/", "test_" + args.output_file_name), "w", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False, indent=4))

    with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    selected_config = input('## input config path ##\n')
    try:
        with open(f'configs/{selected_config}', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"File not found: configs/{selected_config}")
    except json.decoder.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

    main(config=config)