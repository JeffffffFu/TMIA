#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Continuous update fine-tuning script for Llama 3.2 3B using Unsloth on SQuAD.
"""

import os
import time
import math
import json
import random
import argparse
import numpy as np
from datetime import datetime
from datasets import load_dataset, Dataset, concatenate_datasets
import torch
from torch.utils.data import Subset
from transformers import DataCollatorForLanguageModeling, TrainerCallback

from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments, is_bfloat16_supported
from data.load_data import get_data
from data.prepare_data import split_dataset4


# ------------------------ Evaluation helpers ------------------------ #

def normalize_answer(s: str) -> str:
    """SQuAD-style normalization."""
    import re
    import string

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s: str):
    if not s:
        return []
    return normalize_answer(s).split()


def exact_match_score(prediction: str, ground_truth: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    from collections import Counter

    pred_tokens = get_tokens(prediction)
    truth_tokens = get_tokens(ground_truth)

    if len(pred_tokens) == 0 and len(truth_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0.0

    common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common_tokens.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    if not ground_truths:
        return 0.0
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def _build_eval_prompt(question: str, context: str) -> str:
    return (
        "You are a question answering assistant. Extract exact phrases from the given context "
        "to answer questions. Only provide the exact text/phrases from the context that answer "
        "the question. Do not add explanations or commentary. If the information is not found, "
        "respond with 'Not found'.\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def generate_answer_for_eval(model, tokenizer, question, context, max_seq_length, device, max_new_tokens: int = 32):
    # 简单截断 context，避免过长
    max_context_chars = max_seq_length * 4  # 大致估计（英文字母平均 4 chars/token）
    if len(context) > max_context_chars:
        context = context[:max_context_chars]

    prompt = _build_eval_prompt(question, context)

    tokenized = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length - max_new_tokens,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **tokenized,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][tokenized["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    if "\n" in generated:
        generated = generated.split("\n")[0].strip()

    return generated or "Not found"


def compute_ppl_on_dataset(model, tokenizer, dataset, max_seq_length: int, max_samples: int = 500, device=None):

    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    if max_samples is not None and max_samples > 0:
        eval_dataset = dataset.select(range(min(max_samples, len(dataset))))
    else:
        eval_dataset = dataset
    
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i, example in enumerate(eval_dataset):
            text = example["text"]
            
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_length,
            ).to(device)
            
            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # 累计loss和token数
            num_tokens = inputs["input_ids"].numel()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    model.train()  # 恢复训练模式
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    
    max_ppl = 1e10
    if ppl > max_ppl:
        ppl = max_ppl
    
    return ppl


def compute_per_sample_ppl(model, tokenizer, dataset, max_seq_length: int, max_samples: int = None, device=None):

    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # 如果指定了max_samples，则采样
    if max_samples is not None and max_samples > 0:
        eval_dataset = dataset.select(range(min(max_samples, len(dataset))))
    else:
        eval_dataset = dataset
    
    per_sample_results = []
    
    with torch.no_grad():
        for i, example in enumerate(eval_dataset):
            text = example["text"]
            
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_length,
            ).to(device)
            
            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # 计算该样本的PPL
            num_tokens = inputs["input_ids"].numel()
            if num_tokens > 0:
                sample_loss = loss.item()
                sample_ppl = math.exp(sample_loss)
            else:
                sample_loss = float('inf')
                sample_ppl = float('inf')
            
            per_sample_results.append({
                "index": i,
                "ppl": sample_ppl,
                "loss": sample_loss,
                "num_tokens": num_tokens
            })
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(eval_dataset)} samples...")
    
    model.train()
    
    return per_sample_results


class PPLEvaluationCallback(TrainerCallback):

    def __init__(self, train_dataset, val_dataset, tokenizer, max_seq_length, eval_samples=500):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.eval_samples = eval_samples
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):

        if logs is None:
            return
        
        if "loss" in logs:
            device = next(model.parameters()).device
            
            train_ppl = compute_ppl_on_dataset(
                model, self.tokenizer, self.train_dataset,
                self.max_seq_length, self.eval_samples, device
            )
            
            val_ppl = compute_ppl_on_dataset(
                model, self.tokenizer, self.val_dataset,
                self.max_seq_length, self.eval_samples, device
            )
            
            print(f"\n[Step {state.global_step}] Train PPL: {train_ppl:.4f} | Val PPL: {val_ppl:.4f}")
            
            logs["train_ppl"] = train_ppl
            logs["val_ppl"] = val_ppl


def evaluate_on_squad_split(model, tokenizer, split: str, max_seq_length: int, max_samples: int | None) -> dict:

    squad = load_dataset("squad")
    dataset = squad[split]

    if max_samples is not None and max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    device = next(model.parameters()).device

    total_em = 0.0
    total_f1 = 0.0
    n = len(dataset)

    print(f"\n[Eval] Evaluating on SQuAD {split} split ({n} samples)...")

    for i, ex in enumerate(dataset):
        question = ex["question"]
        context = ex["context"]
        answers = ex["answers"]["text"]

        if answers and len(answers) > 0:
            ground_truths = [a.strip() for a in answers if a.strip()] or ["Not found"]
        else:
            ground_truths = ["Not found"]

        pred = generate_answer_for_eval(model, tokenizer, question, context, max_seq_length, device)

        em = metric_max_over_ground_truths(exact_match_score, pred, ground_truths)
        f1 = metric_max_over_ground_truths(f1_score, pred, ground_truths)

        total_em += em
        total_f1 += f1

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n} examples...")

    avg_em = (total_em / n) * 100
    avg_f1 = (total_f1 / n) * 100
    print(f"[Eval] Split={split} | EM={avg_em:.2f}% | F1={avg_f1:.2f}%")
    return {"exact_match": avg_em, "f1": avg_f1, "num_samples": n}


def unlearn_on_remove_set_LLM(model, tokenizer, remove_dataset, max_seq_length, device, num_epochs=3, learning_rate=1e-4):

    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    
    model.train()
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
    
    tokenized_dataset = remove_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"] if "text" in remove_dataset.column_names else []
    )
    
    # 使用 DataCollatorForLanguageModeling
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    

    unlearn_learning_rate = learning_rate * 0.2
    optimizer = AdamW(model.parameters(), lr=unlearn_learning_rate, weight_decay=0.01)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=16,
            shuffle=True,
            collate_fn=data_collator
        )
        
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass

            outputs = model(**inputs)
            loss = outputs.loss
            
            optimizer.zero_grad()
            (-loss).backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        

        model.eval()
        remove_ppl = compute_ppl_on_dataset(
            model, tokenizer, remove_dataset, max_seq_length,
            max_samples=min(1000, len(remove_dataset)), device=device
        )
        model.train()
        print(f"      Unlearn epoch {epoch + 1}/{num_epochs}, Remove PPL: {remove_ppl:.2f}")


def finetune_on_retain_set_LLM(model, tokenizer, retain_dataset, max_seq_length, device, num_epochs=1, learning_rate=1e-4):

    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    
    model.train()
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
    
    tokenized_dataset = retain_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"] if "text" in retain_dataset.column_names else []
    )
    
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=16,
            shuffle=True,
            collate_fn=data_collator
        )
        
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass

            outputs = model(**inputs)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        retain_ppl = compute_ppl_on_dataset(
            model, tokenizer, retain_dataset, max_seq_length,
            max_samples=min(1000, len(retain_dataset)), device=device
        )
        print(f"      Finetune epoch {epoch + 1}/{num_epochs}, Retain PPL: {retain_ppl:.2f}")


def train_single_model_LLM(args, train_dataset, val_dataset, trial, timestamps, is_shadow=False):

    model_prefix = "[SHADOW]" if is_shadow else ""
    model_type = "shadow" if is_shadow else "target"
    
    model_name_arg = args.get('net_name', 'llama3b').lower()
    model_name_map = {
        'llama3b': "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        "llama8b": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    }
    if model_name_arg in model_name_map:
        model_name = model_name_map[model_name_arg]
    else:
        model_name = args.get('model_name', "unsloth/Llama-3.2-3B-Instruct-bnb-4bit")
    print(f"  {model_prefix} -> Selected model: {model_name} (from args['model_name']='{model_name_arg}')")
    max_seq_length = args.get('max_seq_length', int(os.getenv("MAX_SEQ_LENGTH", "256")))
    per_device_train_batch_size = args.get('per_device_train_batch_size', int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", "16")))
    gradient_accumulation_steps = args.get('gradient_accumulation_steps', int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "4")))
    num_train_epochs = args.get('num_train_epochs', int(os.getenv("NUM_TRAIN_EPOCHS", "5")))
    update_proportion = args.get('proportion_of_group_unlearn')
    cooldown_period = args.get('cooldown_period', 20)
    

    if 'num_alternations' in args:
        num_alternations = args.get('num_alternations')
    else:
        if 'llama3b' in model_name_arg:
            num_alternations = 4
            remove_epoch = args.get('remove_epoch', 3)
            retain_epoch = args.get('retain_epoch', 1)
        elif 'llama8b' in model_name_arg:
            num_alternations = 2
            remove_epoch = args.get('remove_epoch', 2)
            retain_epoch = args.get('retain_epoch', 1)
        else:
            num_alternations = 4
            remove_epoch = args.get('remove_epoch', 2)
            retain_epoch = args.get('retain_epoch', 1)

    random_seed = args.get('random_seed', args.get('random', 3407))
    learning_rate = 1.5e-4

    print(f'\n{model_prefix} ========== {model_type.upper()} MODEL - The {trial}-th trial ==========')
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    print(f"  {model_prefix} [1] Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    tokenizer.model_max_length = max_seq_length
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"  {model_prefix} [2] Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=random_seed,
        use_rslora=True,
        loftq_config=None,
    )

    if isinstance(train_dataset, Subset):
        train_data_list = [train_dataset[i] for i in range(len(train_dataset))]
        train_dataset = Dataset.from_list(train_data_list)
    if isinstance(val_dataset, Subset):
        val_data_list = [val_dataset[i] for i in range(len(val_dataset))]
        val_dataset = Dataset.from_list(val_data_list)
    
    print(f"  {model_prefix} -> Train size: {len(train_dataset)} samples")
    print(f"  {model_prefix} -> Validation size: {len(val_dataset)} samples")

    print(f"  {model_prefix} [3] Initializing training update system...")
    

    all_samples_list = []
    for i in range(len(train_dataset)):
        all_samples_list.append(train_dataset[i])
    for i in range(len(val_dataset)):
        all_samples_list.append(val_dataset[i])
    
    all_samples = Dataset.from_list(all_samples_list)
    total_samples = len(all_samples)
    
    sample_status = np.zeros(total_samples, dtype=int)
    sample_status[:len(train_dataset)] = 1
    
    print(f"  {model_prefix} -> Total samples: {total_samples}")
    print(f"  {model_prefix} -> Initial training set size: {len(train_dataset)} (status=1)")
    print(f"  {model_prefix} -> Initial validation set size: {len(val_dataset)} (status=0)")
    
    current_train_indices = list(range(len(train_dataset)))
    
    sample_status_history = []
    for i in range(total_samples):
        sample_status_history.append([sample_status[i]])
    
    fixed_unseen_indices = set()
    if args['dataset_name'] == 'simpleqa':
        num_fixed_unseen = 10
        initial_unseen_indices = np.where(sample_status == 0)[0].tolist()
        all_indices = list(range(total_samples))
        
        if len(initial_unseen_indices) >= num_fixed_unseen:
            fixed_unseen_indices = set(random.sample(initial_unseen_indices, num_fixed_unseen))
        else:
            fixed_unseen_indices = set(random.sample(initial_unseen_indices, len(initial_unseen_indices)))
            remaining_needed = num_fixed_unseen - len(fixed_unseen_indices)
            other_indices = [idx for idx in all_indices if idx not in fixed_unseen_indices]
            additional_fixed = random.sample(other_indices, min(remaining_needed, len(other_indices)))
            fixed_unseen_indices.update(additional_fixed)
            for idx in additional_fixed:
                sample_status[idx] = 0
                if idx in current_train_indices:
                    current_train_indices.remove(idx)
        print(f"  {model_prefix} -> Fixed {len(fixed_unseen_indices)} samples to remain unseen permanently (SimpleQA dataset)")
        
        for i in range(total_samples):
            sample_status_history[i][0] = sample_status[i]  # 更新初始状态
        
        current_train_indices = [idx for idx in current_train_indices if idx not in fixed_unseen_indices]
    
    last_update_timestamp = {i: None for i in range(total_samples)}
    
    net_name = args.get('net_name', 'llama')
    dataset_name = args.get('dataset_name', 'squad')
    proportion = args.get('proportion_of_group_unlearn')
    U_method = args.get('U_method')
    save_path = os.getcwd() + f"/save/{U_method}/{net_name}/{dataset_name}/{proportion}/{model_type}/{trial}/"
    os.makedirs(save_path, exist_ok=True)
    
    timestamp_logs = []
    device = next(model.parameters()).device
    
    print(f"  {model_prefix} [3.5] Training initial model on train set...")
    initial_train_data = [all_samples[i] for i in current_train_indices]
    initial_train_dataset = Dataset.from_list(initial_train_data)
    
    print(f"  {model_prefix} -> Initial training set size: {len(initial_train_dataset)} samples")
    
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    initial_training_args = UnslothTrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        embedding_learning_rate=1e-5,
        warmup_ratio=0.03,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        save_steps=500,
        save_total_limit=1,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=True,
        seed=random_seed,
    )

    initial_trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=initial_train_dataset,
        eval_dataset=None,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        data_collator=data_collator,
        args=initial_training_args,
    )
    
    print(f"  {model_prefix} -> Training initial model...")
    initial_trainer.train()
    print(f"  {model_prefix} -> Initial model training completed")
    
    print(f"  {model_prefix} [4] Starting training update learning...")
    for k in range(timestamps):
        print(f'\n  {model_prefix} -> Timestamp {k}')
        
        remove_indices = []
        insert_indices = []
        
        num_to_remove = max(1, int(len(current_train_indices) * update_proportion))
        if num_to_remove > 0 and len(current_train_indices) > 0:
            eligible_for_remove = [
                idx for idx in current_train_indices
                if last_update_timestamp[idx] is None or (k - last_update_timestamp[idx] >= cooldown_period)
            ]
            
            if len(eligible_for_remove) >= num_to_remove:
                remove_indices = random.sample(eligible_for_remove, min(num_to_remove, len(eligible_for_remove)))
                print(f"    {model_prefix} -> Remove: {len(remove_indices)} samples from training set")
            else:
                print(f"    {model_prefix} -> Skip remove: only {len(eligible_for_remove)} eligible samples (need {num_to_remove}, {len(current_train_indices) - len(eligible_for_remove)} in cooldown)")
        
        available_test_indices = [
            idx for idx in range(total_samples)
            if sample_status[idx] == 0
               and idx not in remove_indices
               and idx not in fixed_unseen_indices
               and (last_update_timestamp[idx] is None or (k - last_update_timestamp[idx] >= cooldown_period))
        ]
        num_to_insert = max(1, int(len(current_train_indices) * update_proportion))
        
        if num_to_insert > 0 and len(available_test_indices) >= num_to_insert:
            insert_indices = random.sample(
                available_test_indices,
                min(num_to_insert, len(available_test_indices))
            )
            print(f"    {model_prefix} -> Insert: {len(insert_indices)} samples from validation set to training set")
        else:
            if num_to_insert > 0 and len(available_test_indices) < num_to_insert:
                total_available = len([idx for idx in range(total_samples) if sample_status[idx] == 0 and idx not in remove_indices])
                in_cooldown = total_available - len(available_test_indices)
                print(f"    {model_prefix} -> Skip insert: only {len(available_test_indices)} available samples (need {num_to_insert}, {in_cooldown} in cooldown)")
        
        for idx in remove_indices:
            sample_status[idx] = 0
            last_update_timestamp[idx] = k
        current_train_indices = [idx for idx in current_train_indices if idx not in remove_indices]
        
        for idx in insert_indices:
            sample_status[idx] = 1
            last_update_timestamp[idx] = k
        current_train_indices.extend(insert_indices)
        
        has_any_operation = len(remove_indices) > 0 or len(insert_indices) > 0
        
        if not has_any_operation:
            print(f"    {model_prefix} -> Skip model update: all samples in cooldown (no remove/insert operations)")
        else:

            remove_dataset = None
            if len(remove_indices) > 0:
                remove_data = [all_samples[i] for i in remove_indices]
                remove_dataset = Dataset.from_list(remove_data)
                print(f"    {model_prefix} -> Prepared remove set: {len(remove_indices)} samples")
            
            retain_indices = np.where(sample_status == 1)[0].tolist()
            retain_dataset = None
            if len(retain_indices) > 0:
                retain_data = [all_samples[i] for i in retain_indices]
                retain_dataset = Dataset.from_list(retain_data)
                print(f"    {model_prefix} -> Prepared retain set: {len(retain_indices)} samples (including {len(insert_indices)} newly inserted)")
            
            for alt_iter in range(num_alternations):
                is_last_alternation = (alt_iter == num_alternations - 1)
                print(f"    {model_prefix} -> Alternation {alt_iter + 1}/{num_alternations}:")
                
                if remove_dataset is not None and len(remove_indices) > 0:
                    unlearn_on_remove_set_LLM(
                        model, tokenizer, remove_dataset, max_seq_length, device,
                        num_epochs=remove_epoch, learning_rate=learning_rate
                    )
                    print(f"      {model_prefix} -> Remove unlearn (gradient ascent): {len(remove_indices)} samples, {remove_epoch} epochs")
                    
                    if len(insert_indices) > 0:
                        insert_data_tmp = [all_samples[i] for i in insert_indices]
                        insert_dataset_tmp = Dataset.from_list(insert_data_tmp)
                        insert_ppl_after_remove = compute_ppl_on_dataset(
                            model, tokenizer, insert_dataset_tmp, max_seq_length,
                            max_samples=min(200, len(insert_dataset_tmp)), device=device
                        )
                    else:
                        insert_ppl_after_remove = float('inf')
                    
                    if len(remove_indices) > 0:
                        remove_ppl_after_remove = compute_ppl_on_dataset(
                            model, tokenizer, remove_dataset, max_seq_length,
                            max_samples=min(200, len(remove_dataset)), device=device
                        )
                    else:
                        remove_ppl_after_remove = float('inf')
                    
                    insert_str = f"{insert_ppl_after_remove:.4f}" if insert_ppl_after_remove != float('inf') else "N/A"
                    remove_str = f"{remove_ppl_after_remove:.4f}" if remove_ppl_after_remove != float('inf') else "N/A"
                    print(f"      {model_prefix} -> After remove unlearn - Insert PPL: {insert_str}, Remove PPL: {remove_str}")
                
                if not is_last_alternation:
                    if retain_dataset is not None and len(retain_indices) > 0:
                        finetune_on_retain_set_LLM(
                            model, tokenizer, retain_dataset, max_seq_length, device,
                            num_epochs=retain_epoch, learning_rate=learning_rate
                        )
                        print(f"      {model_prefix} -> Retain finetune (correct labels): {len(retain_indices)} samples, {retain_epoch} epochs")
                        
                        if len(insert_indices) > 0:
                            insert_data_tmp = [all_samples[i] for i in insert_indices]
                            insert_dataset_tmp = Dataset.from_list(insert_data_tmp)
                            insert_ppl_after_retain = compute_ppl_on_dataset(
                                model, tokenizer, insert_dataset_tmp, max_seq_length,
                                max_samples=min(200, len(insert_dataset_tmp)), device=device
                            )
                        else:
                            insert_ppl_after_retain = float('inf')
                        
                        if len(remove_indices) > 0:
                            remove_ppl_after_retain = compute_ppl_on_dataset(
                                model, tokenizer, remove_dataset, max_seq_length,
                                max_samples=min(200, len(remove_dataset)), device=device
                            )
                        else:
                            remove_ppl_after_retain = float('inf')
                        
                        insert_str = f"{insert_ppl_after_retain:.4f}" if insert_ppl_after_retain != float('inf') else "N/A"
                        remove_str = f"{remove_ppl_after_retain:.4f}" if remove_ppl_after_retain != float('inf') else "N/A"
                        print(f"      {model_prefix} -> After retain finetune - Insert PPL: {insert_str}, Remove PPL: {remove_str}")
                else:
                    print(f"      {model_prefix} -> Skipping retain finetune (last alternation)")
                    if len(insert_indices) > 0:
                        insert_data_tmp = [all_samples[i] for i in insert_indices]
                        insert_dataset_tmp = Dataset.from_list(insert_data_tmp)
                        insert_ppl_after_retain = compute_ppl_on_dataset(
                            model, tokenizer, insert_dataset_tmp, max_seq_length,
                            max_samples=min(200, len(insert_dataset_tmp)), device=device
                        )
                    else:
                        insert_ppl_after_retain = float('inf')
                    
                    if len(remove_indices) > 0:
                        remove_ppl_after_retain = compute_ppl_on_dataset(
                            model, tokenizer, remove_dataset, max_seq_length,
                            max_samples=min(200, len(remove_dataset)), device=device
                        )
                    else:
                        remove_ppl_after_retain = float('inf')
            
            print(f"    {model_prefix} -> Alternating training completed ({num_alternations} alternations)")
            print(f"    {model_prefix} -> Evaluating PPL on insert and remove data...")
            
            if len(insert_indices) > 0:
                insert_data = [all_samples[i] for i in insert_indices]
                insert_dataset = Dataset.from_list(insert_data)
                insert_ppl = compute_ppl_on_dataset(
                    model, tokenizer, insert_dataset, max_seq_length,
                    max_samples=min(200, len(insert_dataset)), device=device
                )
                print(f"    {model_prefix} -> Insert PPL: {insert_ppl:.2f} ({len(insert_indices)} samples)")
            else:
                insert_ppl = float('inf')
                print(f"    {model_prefix} -> Insert PPL: N/A (no insert samples)")
            
            if len(remove_indices) > 0:
                remove_data = [all_samples[i] for i in remove_indices]
                remove_dataset_final = Dataset.from_list(remove_data)
                remove_ppl_final = compute_ppl_on_dataset(
                    model, tokenizer, remove_dataset_final, max_seq_length,
                    max_samples=min(200, len(remove_dataset_final)), device=device
                )
                print(f"    {model_prefix} -> Remove PPL: {remove_ppl_final:.2f} ({len(remove_indices)} samples)")
            else:
                remove_ppl_final = float('inf')
                print(f"    {model_prefix} -> Remove PPL: N/A (no remove samples)")
        
        for idx in fixed_unseen_indices:
            sample_status[idx] = 0
        
        for i in range(total_samples):
            sample_status_history[i].append(sample_status[i])
        
        print(f"    {model_prefix} -> Computing PPL...")
        
        train_indices = np.where(sample_status == 1)[0].tolist()
        if len(train_indices) > 0:
            train_data = [all_samples[i] for i in train_indices]
            train_dataset_current = Dataset.from_list(train_data)
            train_ppl = compute_ppl_on_dataset(
                model, tokenizer, train_dataset_current, max_seq_length,
                max_samples=min(200, len(train_dataset_current)), device=device
            )
        else:
            train_ppl = float('inf')
        val_indices = np.where(sample_status == 0)[0].tolist()
        if len(val_indices) > 0:
            val_data = [all_samples[i] for i in val_indices]
            val_dataset_current = Dataset.from_list(val_data)
            val_ppl = compute_ppl_on_dataset(
                model, tokenizer, val_dataset_current, max_seq_length,
                max_samples=min(200, len(val_dataset_current)), device=device
            )
        else:
            val_ppl = float('inf')
        print(f"    {model_prefix} -> Train PPL: {train_ppl:.2f} | Val PPL: {val_ppl:.2f}")
        
        print(f"    {model_prefix} -> Computing PPL for all samples...")
        model.eval()
        outputs_current = []
        indices_list = []
        
        with torch.no_grad():
            for idx in range(total_samples):
                sample = all_samples[idx]
                text = sample["text"]
                
                # Tokenize
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_seq_length,
                ).to(device)
                
                # Forward pass
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
                
                sample_ppl = math.exp(loss)
                
                max_ppl = 1e10
                if sample_ppl > max_ppl:
                    sample_ppl = max_ppl
                
                outputs_current.append([sample_ppl])
                indices_list.append(idx)
                
                if (idx + 1) % 1000 == 0:
                    print(f"      {model_prefix} -> Processed {idx+1}/{total_samples} samples...")
        
        model.train()
        
        outputs_current = np.array(outputs_current)  # shape: (total_samples, 1)
        indices_array = np.array(indices_list)
        
        timestamp_save_path = f"{save_path}/timestamp_{k}/"
        os.makedirs(timestamp_save_path, exist_ok=True)
        
        np.save(f"{timestamp_save_path}/outputs_current.npy", outputs_current)
        np.save(f"{timestamp_save_path}/sample_indices.npy", indices_array)

        log_entry = f"Timestamp {k}:\n"
        log_entry += f"  -> Remove: {len(remove_indices)} samples\n"
        log_entry += f"  -> Insert: {len(insert_indices)} samples\n"
        log_entry += f"  -> Training set size: {len(current_train_indices)}\n"
        log_entry += f"  -> Train PPL: {train_ppl:.2f} | Val PPL: {val_ppl:.2f}\n"
        timestamp_logs.append(log_entry)
        
        print(f"    {model_prefix} -> Timestamp {k} completed")
    
    print(f"\n  {model_prefix} [6] Saving final results...")
    
    sample_status_array = np.array(sample_status_history)
    np.save(f"{save_path}/sample_status_history.npy", sample_status_array)
    print(f"  {model_prefix} -> Saved sample status history to {save_path}/sample_status_history.npy")
    print(f"      {model_prefix} Shape: {sample_status_array.shape} (samples x timestamps)")
    
    num_timestamps_with_initial = sample_status_array.shape[1]
    num_timestamps = num_timestamps_with_initial - 1
    
    converted_status_array = np.zeros((total_samples, num_timestamps), dtype=int)
    
    for i in range(total_samples):
        for k in range(1, num_timestamps_with_initial):
            prev_status = sample_status_array[i, k - 1]
            curr_status = sample_status_array[i, k]
            
            if prev_status == 0 and curr_status == 0:
                converted_status_array[i, k - 1] = 0
            elif prev_status == 1 and curr_status == 1:
                converted_status_array[i, k - 1] = 1
            elif prev_status == 0 and curr_status == 1:
                converted_status_array[i, k - 1] = 2
            elif prev_status == 1 and curr_status == 0:
                converted_status_array[i, k - 1] = 3
    
    np.save(f"{save_path}/sample_status_converted.npy", converted_status_array)
    
    converted_stats = {
        0: np.sum(converted_status_array == 0),
        1: np.sum(converted_status_array == 1),
        2: np.sum(converted_status_array == 2),
        3: np.sum(converted_status_array == 3)
    }
    
    num_unseen_per_sample = np.sum(converted_status_array == 0, axis=1)
    num_retain_per_sample = np.sum(converted_status_array == 1, axis=1)
    num_insert_per_sample = np.sum(converted_status_array == 2, axis=1)
    num_remove_per_sample = np.sum(converted_status_array == 3, axis=1)
    num_breakpoints_per_sample = num_insert_per_sample + num_remove_per_sample
    
    all_unseen_samples = np.where((num_breakpoints_per_sample == 0) & (num_unseen_per_sample == num_timestamps))[0]
    all_retain_samples = np.where((num_breakpoints_per_sample == 0) & (num_retain_per_sample == num_timestamps))[0]
    
    pattern_statistics = {
        'all_unseen': len(all_unseen_samples),
        'all_retain': len(all_retain_samples),
    }
    
    for num_insert in range(num_timestamps + 1):
        for num_remove in range(num_timestamps + 1):
            if num_insert + num_remove == 0:
                continue
            if num_insert + num_remove > num_timestamps:
                continue
            mask = (num_insert_per_sample == num_insert) & (num_remove_per_sample == num_remove)
            count = np.sum(mask)
            if count > 0:
                pattern_key = f"{num_insert}_insert_{num_remove}_remove"
                pattern_statistics[pattern_key] = count
    breakpoint_indices_dict = {}
    for i in range(total_samples):
        insert_positions = np.where(converted_status_array[i, :] == 2)[0].tolist()
        remove_positions = np.where(converted_status_array[i, :] == 3)[0].tolist()
        
        insert_timestamps = insert_positions
        remove_timestamps = remove_positions
        
        if len(insert_timestamps) > 0 or len(remove_timestamps) > 0:
            breakpoint_indices_dict[i] = {
                'insert': insert_timestamps,
                'remove': remove_timestamps
            }
    
    breakpoint_indices_file = f"{save_path}/breakpoint_indices.npy"
    np.save(breakpoint_indices_file, breakpoint_indices_dict, allow_pickle=True)
    print(f"  {model_prefix} -> Saved breakpoint indices to {breakpoint_indices_file}")
    
    all_unseen_count = pattern_statistics['all_unseen']
    all_retain_count = pattern_statistics['all_retain']
    all_unseen_pct = (all_unseen_count / total_samples) * 100 if total_samples > 0 else 0.0
    all_retain_pct = (all_retain_count / total_samples) * 100 if total_samples > 0 else 0.0
    print(f"      {model_prefix} {'All Unseen (no breakpoint)':<40} {all_unseen_count:<15} {all_unseen_pct:>13.2f}%")
    print(f"      {model_prefix} {'All Retain (no breakpoint)':<40} {all_retain_count:<15} {all_retain_pct:>13.2f}%")
    
    for pattern_key in sorted(pattern_statistics.keys()):
        if pattern_key in ['all_unseen', 'all_retain']:
            continue
        count = pattern_statistics[pattern_key]
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0.0
        parts = pattern_key.split('_')
        num_insert = int(parts[0])
        num_remove = int(parts[2])
        pattern_desc = f"{num_insert} insert, {num_remove} remove breakpoints"
        print(f"      {model_prefix} {pattern_desc:<40} {count:<15} {percentage:>13.2f}%")
    
    main_stats_file_path = f"{save_path}/status_statistics.txt"
    with open(main_stats_file_path, 'w', encoding='utf-8') as f:
        f.write(f"{model_type.upper()} Model - Trial {trial} - Converted Status Pattern Statistics\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Total timestamps: {num_timestamps}\n\n")
        f.write(f"Status encoding:\n")
        f.write(f"  0: Unseen (0->0, not breakpoint)\n")
        f.write(f"  1: Retain (1->1, not breakpoint)\n")
        f.write(f"  2: Insert (0->1, breakpoint)\n")
        f.write(f"  3: Remove (1->0, breakpoint)\n\n")
        f.write(f"Pattern Distribution:\n")
        f.write(f"{'Pattern':<50} {'Count':<15} {'Percentage':<15}\n")
        f.write("-" * 80 + "\n")
        
        f.write(f"{'All Unseen (no breakpoint)':<50} {all_unseen_count:<15} {all_unseen_pct:>13.2f}%\n")
        f.write(f"{'All Retain (no breakpoint)':<50} {all_retain_count:<15} {all_retain_pct:>13.2f}%\n")
        
        for pattern_key in sorted(pattern_statistics.keys()):
            if pattern_key in ['all_unseen', 'all_retain']:
                continue
            count = pattern_statistics[pattern_key]
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0.0
            parts = pattern_key.split('_')
            num_insert = int(parts[0])
            num_remove = int(parts[2])
            pattern_desc = f"{num_insert} insert, {num_remove} remove breakpoints"
            f.write(f"{pattern_desc:<50} {count:<15} {percentage:>13.2f}%\n")
        
        f.write(f"\n\nDetailed Status Count Statistics:\n")
        f.write(f"{'Status Type':<30} {'Total Count':<20} {'Average per Sample':<20}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Unseen (0->0)':<30} {converted_stats[0]:<20} {converted_stats[0] / total_samples:.2f}\n")
        f.write(f"{'Retain (1->1)':<30} {converted_stats[1]:<20} {converted_stats[1] / total_samples:.2f}\n")
        f.write(f"{'Insert (0->1)':<30} {converted_stats[2]:<20} {converted_stats[2] / total_samples:.2f}\n")
        f.write(f"{'Remove (1->0)':<30} {converted_stats[3]:<20} {converted_stats[3] / total_samples:.2f}\n")
    print(f"  {model_prefix} -> Saved main status statistics to {main_stats_file_path}")
    
    log_file_path = f"{save_path}/timestamp_logs.txt"
    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write(f"{model_type.upper()} Model - Trial {trial} - Timestamp Logs\n")
        f.write("=" * 50 + "\n\n")
        for log_entry in timestamp_logs:
            f.write(log_entry + "\n")
    print(f"  {model_prefix} -> Saved timestamp logs to {log_file_path}")


def continuous_update_finetune_LLM(args):

    print("dataset and net_name:", args.get('dataset_name', 'squad'), args.get('net_name', None))
    dataset_name = args.get('dataset_name', 'squad')
    if dataset_name.lower() == 'simpleqa':
        timestamps = args.get('timestamps', 40)
        print(f"  -> SimpleQA dataset detected: using {timestamps} timestamps")
    else:
        timestamps = args.get('timestamps', 60)
    update_proportion = args.get('proportion_of_group_unlearn')
    max_train_samples = args.get('max_train_samples', 10000)
    max_val_samples = args.get('max_val_samples', 10000)
    net_name = args.get('net_name', None)
    random_seed = args.get('random_seed', args.get('random', 3407))
    trials = args.get('trials', 1)
    print(f"  -> Total timestamps: {timestamps}")
    print(f"  -> Update proportion per step: {update_proportion * 100}%")
    print(f"  -> Max train samples: {max_train_samples}")
    print(f"  -> Max val samples: {max_val_samples}")
    
    print("\n[Loading data for target model...]")
    train_data_full, test_data_full = get_data(
        dataset_name, 
        net_name, 
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples
    )
    
    print(f"  -> Full train data size: {len(train_data_full)}")
    print(f"  -> Full test data size: {len(test_data_full)}")
    
    print("\n[Splitting data...]")
    
    train_size = len(train_data_full)
    train_indices = list(range(train_size))
    random.seed(random_seed)
    random.shuffle(train_indices)
    target_m_indices = train_indices[:train_size//2]
    shadow_m_indices = train_indices[train_size//2:]
    
    target_m = train_data_full.select(target_m_indices)
    shadow_m = train_data_full.select(shadow_m_indices)
    
    print(f"  -> Target train size: {len(target_m)}")
    print(f"  -> Shadow train size: {len(shadow_m)}")
    
    test_size = len(test_data_full)
    test_indices = list(range(test_size))
    random.shuffle(test_indices)
    target_um_indices = test_indices[:test_size//2]
    shadow_um_indices = test_indices[test_size//2:]
    
    target_um = test_data_full.select(target_um_indices)
    shadow_um = test_data_full.select(shadow_um_indices)
    
    print(f"  -> Target test size: {len(target_um)}")
    print(f"  -> Shadow test size: {len(shadow_um)}")
    

    for t in range(trials):
        print(f'\n========== The {t}-th trial ==========')
        
        train_single_model_LLM(
            args=args,
            train_dataset=target_m,
            val_dataset=target_um,
            trial=t,
            timestamps=timestamps,
            is_shadow=False
        )
        
        train_single_model_LLM(
            args=args,
            train_dataset=shadow_m,
            val_dataset=shadow_um,
            trial=t,
            timestamps=timestamps,
            is_shadow=True
        )
    
    print("\nDone. Continuous update fine-tuning completed.")

