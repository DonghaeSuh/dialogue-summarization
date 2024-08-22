import argparse
from peft import LoraConfig, TaskType
from peft import get_peft_model, prepare_model_for_kbit_training

import evaluate

import numpy as np
import torch
from datasets import Dataset
<<<<<<< HEAD
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback, EvalPrediction
=======
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          BitsAndBytesConfig, 
                          EvalPrediction,
                          EarlyStoppingCallback)
import pandas as pd
import numpy as np
import gc
import evaluate

from rouge import Rouge

from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import os

>>>>>>> DonghaeSuh
from src.data import CustomDataset, DataCollatorForSupervisedDataset
from trl import SFTTrainer, SFTConfig

# Argument parsing
parser = argparse.ArgumentParser(prog="train", description="Training for Conversational Context Inference.")
g = parser.add_argument_group("Common Parameter")
<<<<<<< HEAD
g.add_argument("--model_id", type=str, required=True, help="Model file path")
g.add_argument("--tokenizer", type=str, help="Huggingface tokenizer path")
g.add_argument("--save_dir", type=str, default="resource/results", help="Model save path")
g.add_argument("--batch_size", type=int, default=1, help="Batch size (both train and eval)")
g.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
g.add_argument("--warmup_steps", type=int, help="Scheduler warmup steps")
g.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
g.add_argument("--epoch", type=int, default=5, help="Training epoch")
g.add_argument("--metric", type=str, default='total', help="metric for evaluation (eval_loss or rouge)")
=======
g.add_argument("--model_id", type=str, required=True, help="model file path")
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer path")
g.add_argument("--save_dir", type=str, default="resource/results", help="model save path")
g.add_argument("--batch_size", type=int, default=1, help="batch size (both train and eval)")
g.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps")
g.add_argument("--warmup_steps", type=int, help="scheduler warmup steps")
g.add_argument("--lr", type=float, default=2e-5, help="learning rate")
g.add_argument("--epoch", type=int, default=10, help="training epoch")
g.add_argument("--metric", type=str, default='rouge1', help="metric for evaluation (eval_loss or rouge)")
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
>>>>>>> DonghaeSuh

def main(args):
    # Quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # PEFT configuration
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, 
                            inference_mode=False, 
                            r=8, 
                            lora_alpha=32, 
                            lora_dropout=0.05,
                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

    # Load and prepare model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model = model.to('cuda')

    if args.tokenizer == None:
        args.tokenizer = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = CustomDataset("resource/data/일상대화요약_train.json", tokenizer)
    valid_dataset = CustomDataset("resource/data/일상대화요약_dev.json", tokenizer)

    train_dataset = Dataset.from_dict({
        'input_ids': train_dataset.inp,
        "labels": train_dataset.label,
        })
    valid_dataset = Dataset.from_dict({
        'input_ids': valid_dataset.inp,
        "labels": valid_dataset.label,
        })
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
        # Setup evaluation
    rouge = evaluate.load("rouge")
    bert_score = evaluate.load("bertscore")
    bleurt = evaluate.load('bleurt', 'bleurt-large-512', module_type="metric")

<<<<<<< HEAD
=======
    rouge = evaluate.load("rouge")
    # rouge = Rouge()

>>>>>>> DonghaeSuh
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        return preds, labels

<<<<<<< HEAD
    def compute_metrics(eval_preds: EvalPrediction):
        labels = eval_preds.label_ids # (batch_size, seq_len)
        predictions = eval_preds.predictions[0].reshape(labels.shape[0],-1) # (batch_size, seq_len)
=======
    def compute_metrics(eval_pred: EvalPrediction):
        # compute Rouge-1 F1 score
        labels = eval_pred.label_ids # (batch_size, seq_len)
        predictions = eval_pred.predictions[0].reshape(labels.shape[0],-1) # (batch_size, seq_len)
>>>>>>> DonghaeSuh

        # Replace -100 with pad_token_id
        mask  = np.where(labels == -100)
        labels[mask] = tokenizer.pad_token_id
        predictions[mask] = tokenizer.pad_token_id

        # Decoding
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Simple postprocessing
        predictions, labels = postprocess_text(predictions, labels)

        rouge_scores = rouge.compute(predictions=predictions, references=labels, rouge_types=["rouge1"])
        # rouge_scores = rouge.get_scores(predictions, labels, avg=True)
<<<<<<< HEAD
        bert_scores = bert_score.compute(predictions=predictions, references=labels, lang="ko")
        bleurt_scores = bleurt.compute(predictions=predictions, references=labels)
    
        bertScore = sum(bert_scores['f1']) / 102
        bleurtScore = sum(bleurt_scores['scores']) / 102
        rouge1 = rouge_scores['rouge1']
        # print(rouge_scores)
        total = (bertScore + bleurtScore + rouge1) / 3
        return {"total": total,"rouge1": rouge1, "BERTScore": bertScore, "BLEURT": bleurtScore}
    
    def preprocess_logits_for_metrics(logits, labels):
          """
          Original Trainer may have a memory leak. 
          This is a workaround to avoid storing too many tensors that are not needed.
          """
          pred_ids = torch.argmax(logits, dim=-1)
          return pred_ids, labels
          
    # Training arguments
=======

        return {k: round(v, 4) for k, v in rouge_scores.items()}

    def preprocess_logits_for_metrics(logits, labels):
      """
      Original Trainer may have a memory leak. 
      This is a workaround to avoid storing too many tensors that are not needed.
      """
      pred_ids = torch.argmax(logits, dim=-1)
      return pred_ids, labels

>>>>>>> DonghaeSuh
    training_args = SFTConfig(
        output_dir=args.save_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,

<<<<<<< HEAD
        eval_strategy="steps",
        eval_steps=1,
=======
        eval_strategy="steps", # for early stopping
        eval_steps=5, # for early stopping
>>>>>>> DonghaeSuh
        metric_for_best_model = args.metric, # for early stopping
        load_best_model_at_end = True, # for early stopping
        greater_is_better = True, # for early stopping. If true, the best model is the one with the highest value of the metric (used for rouge)
        
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
<<<<<<< HEAD
        # eval_accumulation_steps=4,
        learning_rate=args.lr,
        weight_decay=0.1,
        
=======
        eval_accumulation_steps=4,
        learning_rate=args.lr,
        weight_decay=0.1,

>>>>>>> DonghaeSuh
        num_train_epochs=args.epoch,
        max_steps=-1,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
<<<<<<< HEAD
        
=======

>>>>>>> DonghaeSuh
        log_level="info",
        logging_steps=1,
        save_strategy="steps",
        save_total_limit=4,
<<<<<<< HEAD
        save_steps=1,
        
=======

>>>>>>> DonghaeSuh
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},

<<<<<<< HEAD
=======
        max_seq_length=1024,
        packing=True,

>>>>>>> DonghaeSuh
        seed=42,
        # packing=True,

        report_to="wandb",
<<<<<<< HEAD
        run_name=args.model_id,
        max_seq_length=2048,
        )
=======
        run_name=args.model_id.split('/')[-1],
    )
>>>>>>> DonghaeSuh

    # Trainer setup
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
<<<<<<< HEAD
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=15)], # for early stopping
=======
        args=training_args,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)], # for early stopping
>>>>>>> DonghaeSuh
        preprocess_logits_for_metrics = preprocess_logits_for_metrics
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    exit(main(parser.parse_args()))
