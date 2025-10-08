import json
import logging
import os
import argparse
import random
import math
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import Trainer, TrainingArguments, set_seed
import importlib
import time
from tqdm import tqdm
# from utils import *
 
 
def _encode(exs, tokenizer, max_len, max_samples=None, discard_cutoff_samples=False, apply_gen_token=None):
    all_input_ids = []
    all_label_ids = []
   
    random.shuffle(exs)
    if max_samples is not None:
        exs = exs[:max_samples]
        
    if apply_gen_token is not None:
        gen_token_id = tokenizer.convert_tokens_to_ids([apply_gen_token])[0]
 
    for i, ex in enumerate(tqdm(exs, desc="Processing data...")):
        if apply_gen_token is not None:
            input_ids = tokenizer.apply_chat_template([
                {"role": "system",      "content": ex["instruction"]},
                {"role": "user",        "content": ex["input"]},
                {"role": "assistant",   "content": ex["output"]}
            ])
 
            index = input_ids.index(gen_token_id)
            label_ids = [-100] * index + input_ids[index + 1:]
            input_ids = input_ids[:index] + input_ids[index + 1:]
        
        else:
            input_ids = tokenizer.apply_chat_template([
                {"role": "system",      "content": ex["instruction"]},
                {"role": "user",        "content": ex["input"]},
                {"role": "assistant",   "content": ex["output"]}
            ])
            prefix_ids = tokenizer.apply_chat_template([
                {"role": "system",      "content": ex["instruction"]},
                {"role": "user",        "content": ex["input"]}
            ], add_generation_prompt=True)
 
            label_ids = [-100] * len(prefix_ids) + input_ids[len(prefix_ids):]
 
        if i < 2:
            print("Input text:\n{}".format(tokenizer.decode(input_ids)))
            print("Input ids:\n{}".format(input_ids))
            print("Label ids:\n{}".format(label_ids))
 
        assert len(input_ids) == len(label_ids)
        if len(input_ids) > max_len:
            if discard_cutoff_samples:
                continue
            input_ids = input_ids[:max_len]
            label_ids = label_ids[:max_len]
        else:
            input_ids += [tokenizer.pad_token_id] * (max_len - len(input_ids))
            label_ids += [-100] * (max_len - len(label_ids))
 
        all_input_ids += [input_ids]
        all_label_ids += [label_ids]
 
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_label_ids = torch.tensor(all_label_ids, dtype=torch.long)
 
    return dict(
        input_ids=all_input_ids,
        labels=all_label_ids,
        attention_mask=all_input_ids.ne(tokenizer.pad_token_id),
    )
 
 
class SFTDataset(Dataset):
    def __init__(self, source, tokenizer, max_len, max_samples=None, apply_gen_token=None):
        super().__init__()
 
        if apply_gen_token is not None:
            tokenizer.add_special_tokens({"additional_special_tokens": tokenizer.additional_special_tokens + [apply_gen_token]})
 
        encoded_inputs = _encode(source, tokenizer, max_len, max_samples, apply_gen_token=apply_gen_token)
        self.input_ids = encoded_inputs["input_ids"]
        self.labels = encoded_inputs["labels"]
        self.attention_mask = encoded_inputs["attention_mask"]
 
    def __len__(self):
        return self.input_ids.size(0)
 
    def __getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i]
        )
 
 
def get_fct(fct_path):
    module_name, fct_name = fct_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), fct_name)
 
 
def parse_args():
    parser = argparse.ArgumentParser()
 
    parser.add_argument("--model_path", type=str, default=None, required=True)
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--eval_data_path", type=str, default=None)
    parser.add_argument("--max_train_samples", type=int,default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--input_perturbation", type=float, default=None)
    parser.add_argument("--cutoff_len", type=int, default=2048)
    parser.add_argument("--discard_cutoff_samples", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_strategy", type=str, choices=["no", "epoch", "steps"], default="steps")
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_strategy", type=str, choices=["no", "epoch", "steps"], default="steps")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--continue_from_checkpoint", type=str, default=None)
    parser.add_argument("--report_to", type=str, choices=["none", "wandb", "tensorboard"], default="tensorboard")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--compute_metrics", type=str, default="fct.compute_metrics_accuracy")
    parser.add_argument("--grid_search", type=str, default=None)
    parser.add_argument("--apply_gen_token", type=str, default=None)
 
    args = parser.parse_args()
 
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
 
    return args
 
 
def main():
    args = parse_args()
 
    set_seed(args.seed)
 
    # For multi-machine, TODO
    # local_rank = args.local_rank
    # world_size = int(os.environ.get("WORLD_SIZE", 1))
    # ddp = world_size != 1
    # n_gpu = torch.cuda.device_count()
    
    if args.task_name is None:
        args.task_name = time.strftime("%Y%m%d%H%M%S", time.localtime())
 
    deepspeed_config = None
    if args.deepspeed is not None:
        if args.deepspeed == "z0":
            deepspeed_config = json.load(open("../ds_config/ds_z0_config.json"))
        elif args.deepspeed == "z2":
            deepspeed_config = json.load(open("../ds_config/ds_z2_config.json"))
        elif args.deepspeed == "z3":
            deepspeed_config = json.load(open("../ds_config/ds_z3_config.json"))
 
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        cache_dir=args.cache_dir,
        trust_remote_code=True
    )
 
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
 
    train_source = json.load(open(args.train_data_path, "r"))
    train_dataset = SFTDataset(train_source, tokenizer=tokenizer, max_len=args.cutoff_len, max_samples=args.max_train_samples, apply_gen_token=args.apply_gen_token)
 
    eval_dataset = None
    if args.eval_data_path:
        eval_source = json.load(open(args.eval_data_path, "r"))
        eval_dataset = SFTDataset(eval_source, tokenizer=tokenizer, max_len=args.cutoff_len, max_samples=args.max_eval_samples, apply_gen_token=args.apply_gen_token)
 
    n_steps_per_epoch = math.ceil(len(train_dataset) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = int(args.num_train_epochs * n_steps_per_epoch)
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / n_steps_per_epoch)
 
    def run_one_train():
 
        def format_lr(lr):
            base, exp = format(lr, ".0e").split("e")
            exp = str(int(exp))
            return "{}e{}".format(base, exp)
 
        lr = args.learning_rate
        bsz = args.train_batch_size * args.gradient_accumulation_steps * torch.cuda.device_count()
        ep = args.num_train_epochs
        save_dir = os.path.join(args.output_dir, "{}_lr{}_bsz{}_ep{}".format(args.task_name, format_lr(lr), bsz, ep))
    
        training_args = TrainingArguments(
            output_dir=save_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_steps=int(args.max_train_steps * args.warmup_ratio),
            eval_strategy=args.eval_strategy if eval_dataset else "no",
            eval_steps=args.eval_steps,
            save_strategy=args.save_strategy,
            save_steps=args.save_steps,
            logging_strategy="steps",
            logging_dir="logs",
            logging_steps=args.logging_steps,
            local_rank=args.local_rank,
            fp16=args.fp16,
            bf16=args.bf16,
            report_to=args.report_to,
            neftune_noise_alpha=args.input_perturbation,
            deepspeed=deepspeed_config,
        )
 
        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                logits = logits[0]
                label = label
            return torch.argmax(logits, dim=-1)
 
        compute_metrics = get_fct(args.compute_metrics)
 
        trainer = Trainer(
            model=model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics if eval_dataset else None,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics if eval_dataset else None
        )
 
        if args.continue_from_checkpoint is not None:
            trainer.train(resume_from_checkpoint=args.continue_from_checkpoint)
        else:
            trainer.train()
 
        trainer.save_model()
        trainer.save_state()
    
    if args.grid_search is not None:
        for args in get_fct(args.grid_search)(args):
            run_one_train()
    
    else:
        run_one_train()
 
 
if __name__ == "__main__":
    main()
