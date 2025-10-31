from huggingface_hub import login

login("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")

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"     
os.environ["TRANSFORMERS_NO_FLAX"] = "1"  
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import numpy as np
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType)
from torch.utils.data import DataLoader, Dataset
import yaml
import wandb
from huggingface_hub import HfApi

import contextlib                                   
import torch.distributed as dist                    
from accelerate.utils import extract_model_from_parallel  

config_file = "config.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

dsn1 = config["text_QA_dataset"]
dsn2 = config["TTS_dataset"]

model_name = config["model_name"]
tokenizer_name = config["tokenizer_name"]

run_name = config["run_name"]
project_name = config["project_name"]
base_repo_id = config["save_folder"]

epochs = config["epochs"]
batch_size = config["batch_size"]
save_steps = config["save_steps"]
pad_token = config["pad_token"]
number_processes = config["number_processes"]
learning_rate = config["learning_rate"]
config_ratio = config["ratio"]



class SingleDataset(Dataset):
    def __init__(self, dataset, batch_total=None):
        self.dataset = dataset
        self.batch_total = batch_total or len(dataset)

    def __len__(self):
        print("accessing length", len(self.dataset))
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

class BatchedRatioDataset(Dataset):
    def __init__(self, dataset1, dataset2, batch_total, ratio=config_ratio):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.batch_total = batch_total
        self.ratio = ratio

        num_cycles_ds1 = len(dataset1) // (batch_total * ratio)
        num_cycles_ds2 = len(dataset2) // batch_total
        self.num_cycles = min(num_cycles_ds1, num_cycles_ds2)

        self.length = self.num_cycles * (ratio + 1) * batch_total

    def __len__(self):
        print("accessing length", self.length)
        return int(self.length)

    def __getitem__(self, index):
        # Compute the cycle length in terms of samples.
        cycle_length = (self.ratio + 1) * self.batch_total
        cycle = index // cycle_length
        pos_in_cycle = index % cycle_length

        if pos_in_cycle < self.ratio * self.batch_total:
            batch_in_cycle = pos_in_cycle // self.batch_total
            sample_in_batch = pos_in_cycle % self.batch_total
            ds1_index = cycle * self.ratio * self.batch_total + batch_in_cycle * self.batch_total + sample_in_batch
            return self.dataset1[ds1_index]
        else:
            sample_in_batch = pos_in_cycle - self.ratio * self.batch_total
            ds2_index = cycle * self.batch_total + sample_in_batch
            return self.dataset2[ds2_index]




class FSDPTrainer(Trainer):
    def __init__(self, *args, log_ratio=config_ratio, **kwargs):
        super().__init__(*args, **kwargs)
        self.repo_id = base_repo_id
        self.api = HfApi()

        self.log_ratio = log_ratio
        self.text_step  = 0
        self.audio_step = 0

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=0,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def log(self, logs, start_time=None):
        super().log(logs, start_time)
        if self.is_world_process_zero():
            global_step = self.state.global_step
            cycle_length = self.log_ratio + 1
            if (global_step % cycle_length) + self.log_ratio - 1 < self.log_ratio:
                wandb.log({"audio_loss": logs["loss"], "audio_step": self.audio_step})
                self.audio_step += 1
            else:
                wandb.log({"text_loss": logs["loss"], "text_step": self.text_step})
                self.text_step += 1

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        self.save_and_push_model(output_dir)

    def save_and_push_model(self, output_dir):
        is_main = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0  

        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

        if isinstance(self.model, FSDP):                                              
            ctx = FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy)
        else:
            ctx = contextlib.nullcontext()                                            

        with ctx:
            cpu_state_dict = self.model.state_dict()

        if is_main:                                                                     
            to_save = extract_model_from_parallel(self.model)                           
            to_save.save_pretrained(output_dir, state_dict=cpu_state_dict, safe_serialization=True)  


def data_collator(features):
    input_ids = [f["input_ids"] for f in features]

    if any("attention_mask" not in f for f in features):
        attention_mask = [[1]*len(ids) for ids in input_ids]
    else:
        attention_mask = [f["attention_mask"] for f in features]

    if any("labels" not in f for f in features):
        labels = input_ids
    else:
        labels = [f["labels"] for f in features]

    total = sum(len(l) for l in labels)
    valid = sum((t != -100) for l in labels for t in l)
    if valid == 0:
        print("[WARN] batch with 0 valid labels")

    input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(
        i, dtype=torch.long) for i in input_ids], batch_first=True, padding_value=pad_token)
    attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(
        m, dtype=torch.long) for m in attention_mask], batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(
        l, dtype=torch.long) for l in labels], batch_first=True, padding_value=-100)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


wandb.init(project=project_name, name=run_name)


tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, attn_implementation="flash_attention_2",dtype=torch.float16)
model.gradient_checkpointing_enable()

number_add_tokens = 7 * 4096 + 10
new_tokens = [f"<custom_token_{i}>" for i in range(0, number_add_tokens + 1)]
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))


# ds1 = load_dataset(dsn1, split="train",)
ds2 = load_dataset(dsn2, split="train")
ds2 = ds2.filter(
    lambda ex: len(ex["input_ids"]) <= 2048,
    num_proc=6
)
split_ds = ds2.train_test_split(test_size=0.01, shuffle=True, seed=42)

ds2 = SingleDataset(split_ds["train"])
eval_dataset = SingleDataset(split_ds["test"])


batch_total = batch_size * number_processes
# train_dataset = BatchedRatioDataset(ds1, ds2, batch_total, ratio=config_ratio)
train_dataset = SingleDataset(ds2, batch_total)

training_args = TrainingArguments(
    overwrite_output_dir=True,
    # eval_strategy="steps",
    # eval_steps=500,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    logging_steps=1,
    bf16=True,
    output_dir=f"./{base_repo_id}",
    # fsdp="auto_wrap",
    report_to="wandb",
    save_steps=save_steps,
    remove_unused_columns=True,
    learning_rate=learning_rate,
    lr_scheduler_type="cosine",
)


trainer = FSDPTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # eval_dataset=eval_dataset,
    data_collator=data_collator,
    log_ratio=config_ratio
)

trainer.train()
