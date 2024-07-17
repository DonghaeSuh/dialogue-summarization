
import argparse

import torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          BitsAndBytesConfig, 
                          EvalPrediction,
                          EarlyStoppingCallback)
import pandas as pd
import numpy as np
from rouge import Rouge

from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import os

from src.data import CustomDataset, DataCollatorForSupervisedDataset

import wandb

os.environ["WANDB_PROJECT"]="dialogue-summarization"

# fmt: off
parser = argparse.ArgumentParser(prog="train", description="Training about Dialogue Summarization.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--model_id", type=str, required=True, help="model file path")
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer path")
g.add_argument("--save_dir", type=str, default="resource/results", help="model save path")
g.add_argument("--batch_size", type=int, default=1, help="batch size (both train and eval)")
g.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps")
g.add_argument("--warmup_steps", type=int, help="scheduler warmup steps")
g.add_argument("--lr", type=float, default=2e-5, help="learning rate")
g.add_argument("--epoch", type=int, default=5, help="training epoch")
g.add_argument("--metric", type=str, default='rouge', help="metric for evaluation (eval_loss or rouge)")
# fmt: on

def print_trainable_parameters(model):
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
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def main(args):

    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        # torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # lora config
    config = LoraConfig(
    r=8,
    lora_alpha=32,
    # target_modules=["q_proj","up_proj","o_proj","k_proj","down_proj","gate_proj","v_proj"], # ["query_key_value"]
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
    )

    # peft model
    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    if args.tokenizer == None:
        args.tokenizer = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = CustomDataset("resource/data/일상대화요약_train.json", tokenizer)
    valid_dataset = CustomDataset("resource/data/일상대화요약_dev.json", tokenizer)

    train_dataset = Dataset.from_dict({
        'input_ids': train_dataset.inp,
        "labels": train_dataset.label,
        })
    valid_dataset = Dataset.from_dict({
        'input_ids': valid_dataset.inp,
        "labels": valid_dataset.label,
        })
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)


    ## Postprocess ##
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

    ## Metric ##
    def compute_rouge_f1(predictions, labels):
        rouge = Rouge()

        # remove padding tokens (-100)
        predictions = np.array(predictions)
        labels = np.array(labels)

        labels = labels[labels != -100]
        predictions = predictions[labels != -100]

        # Some simple post-processing
        predictions, labels = postprocess_text(predictions, labels)

        # decode predictions and labels
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        rouge_scores = rouge.get_scores(predictions, labels, avg=True)
        return round(rouge_scores['rouge-1']['f'], 4)


    def compute_metrics(eval_pred: EvalPrediction):
        # compute Rouge-1 F1 score
        predictions = eval_pred.predictions # (batch_size, seq_len)
        labels = eval_pred.label_ids # (batch_size, seq_len)

        return {'rouge-1': compute_rouge_f1(predictions, labels)}

    training_args = SFTConfig(
        output_dir=args.save_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,

        eval_strategy="steps", # for early stopping
        eval_steps=5, # for early stopping
        metric_for_best_model = args.metric, # for early stopping
        compute_metrics=compute_metrics,
        load_best_model_at_end = True, # for early stopping
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)], # for early stopping
        greater_is_better = True, # for early stopping. If true, the best model is the one with the highest value of the metric (used for rouge)
        
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=0.1,

        num_train_epochs=args.epoch,
        max_steps=-1,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,

        log_level="info",
        logging_steps=1,
        save_strategy="steps",
        save_total_limit=3,

        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},

        max_seq_length=1024,
        packing=True,

        seed=42,
        report_to="wandb",
        run_name=args.model_id,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        args=training_args,
    )

    trainer.train()


if __name__ == "__main__":
    exit(main(parser.parse_args()))