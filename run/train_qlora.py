import sys
import os
import gc
import json

# from transformers.utils.dummy_tf_objects import TFDPRQuestionEncoder

# 프로젝트의 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback

from trl import SFTTrainer, SFTConfig

from src.data import ExaoneDataset, CustomDataset, DataCollatorForSupervisedDataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

from src.utils import compute_metrics, preprocess_logits_for_metrics, set_seed
from datetime import datetime

from typing import Dict

# os.makedirs('../cache', exist_ok=True)

def main(config: Dict):
    # seed 고정
    set_seed(config)

    # fmt: off
    parser = argparse.ArgumentParser(prog="train", description="Training about Conversational Context Inference.")

    g = parser.add_argument_group("Common Parameter")
    g.add_argument("--model_id", type=str, default=config["arch"]["model_id"], help="model file path")
    g.add_argument("--save_dir", type=str, default=config["path"]["chkpoint_save_dir"], help="model save path")
    g.add_argument("--batch_size", type=int, default=config["arch"]["batch_size"], help="batch size (both train and eval)")
    g.add_argument("--gradient_accumulation_steps", type=int, default=config["arch"]["gradient_accumulation_steps"], help="gradient accumulation steps")
    g.add_argument("--eval_accumulation_steps", type=int, default=config["arch"]["eval_accumulation_steps"], help="eval_accumulation_steps")
    g.add_argument("--warmup_steps", type=int, default=config["arch"]["warmup_steps"], help="scheduler warmup steps")
    g.add_argument("--lr", type=float, default=config["arch"]["lr"], help="learning rate")
    g.add_argument("--epoch", type=int, default=config["arch"]["epoch"], help="training epoch")
    g.add_argument("--wandb_run_name", type=str, default=config["wandb"]["wandb_run_name"], help="wandb run name")
    g.add_argument("--resume_path", type=str, default=None, help='resume path' )

    args = parser.parse_args()

    os.environ["WANDB_RUN_ID"] = args.wandb_run_name
    os.environ["WANDB_ENTITY"] = config["wandb"]["wandb_entity_name"]  # name your W&B project
    os.environ["WANDB_PROJECT"] = config["wandb"]["wandb_project_name"]  # name your W&B project
    os.environ["WANDB_LOG_MODEL"] = config["wandb"]["wandb_log_model"]  # log all model checkpoints
    os.environ["WANDB_RESUME"] = 'allow'

    print('### Check Model Arguments ... ###')
    print('model_id : ', args.model_id)
    print('wandb_run_name : ', args.wandb_run_name)


    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        # cache_dir='../cache'
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config["lora_arch"]["r"], 
        lora_alpha=config["lora_arch"]["lora_alpha"], 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
        lora_dropout=config["lora_arch"]["lora_dropout"],
        bias="none", 
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"## trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param} ##"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # 모델 이름으로 판단
    if "EXAONE" in args.model_id == True:
        train_dataset = ExaoneDataset(config["path"]["train_path"], tokenizer, is_train=True, is_dev=False)
        valid_dataset = ExaoneDataset(config["path"]["dev_path"], tokenizer, is_train=False, is_dev=True)
    else:
        train_dataset = CustomDataset(config["path"]["train_path"], tokenizer, is_train=True, is_dev=False)
        valid_dataset = CustomDataset(config["path"]["dev_path"], tokenizer, is_train=False, is_dev=True)


    train_dataset = Dataset.from_dict({
        'input_ids': train_dataset.inp,
        "labels": train_dataset.label,
        })
    valid_dataset = Dataset.from_dict({
        'input_ids': valid_dataset.inp,
        "labels": valid_dataset.label,
        })
    
    print(f"## Train dataset : {len(train_dataset)}, Valid dataset : {len(valid_dataset)} loaded ##")
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    

    training_args = SFTConfig(
        output_dir=config["path"]["chkpoint_save_dir"],
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,

        save_strategy=config["arch"]["strategy"],
        eval_strategy=config["arch"]["strategy"],
        save_steps=config["arch"]["steps"],
        eval_steps=config["arch"]["steps"],
        logging_steps=config["arch"]["steps"],

        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,

        learning_rate=args.lr,
        weight_decay=config["arch"]["weight_decay"],
        num_train_epochs=args.epoch,
        lr_scheduler_type=config["arch"]["lr_scheduler_type"],
        warmup_steps=args.warmup_steps,
        # label_smoothing_factor=0.1, # label smoothing

        log_level="info",
        save_total_limit=2,

        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        max_seq_length=config["arch"]["max_seq_length"],
        packing=True,

        seed=config["arch"]["seed"],

        report_to="wandb",
        run_name=args.wandb_run_name,
        metric_for_best_model =config["arch"]["metric_for_best_model"],
        load_best_model_at_end=True
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=config["arch"]["early_stopping_patience"])],
        preprocess_logits_for_metrics = preprocess_logits_for_metrics)
    
    gc.collect()
    torch.cuda.empty_cache()
    trainer.train(resume_from_checkpoint = args.resume_path)

    now = datetime.now()
    trainer.save_model(os.path.join(config["path"]["model_save_dir"], f"run/model/{args.model_id}_batch_{args.batch_size}_{args.wandb_run_name}_time_{now.strftime('%Y-%m-%d_%H:%M')}"))


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