"""
Dataset loading and preprocessing for EchoHeart.
Downloads EmpatheticDialogues and DailyDialog from HuggingFace,
tokenizes for DialoGPT fine-tuning, and saves to disk.
"""

import os
from dataclasses import dataclass
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


MAX_LENGTH = 512
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data_cache")


@dataclass
class ConversationSample:
    context: List[str]
    response: str


def load_empathetic_dialogues() -> List[ConversationSample]:
    dataset = load_dataset("empathetic_dialogues", trust_remote_code=True)
    samples = []
    for split in ["train", "validation"]:
        grouped: Dict[str, List] = {}
        for row in dataset[split]:
            conv_id = row["conv_id"]
            grouped.setdefault(conv_id, []).append(row)
        for utterances in grouped.values():
            utterances.sort(key=lambda x: x["utterance_idx"])
            turns = [u["utterance"].strip() for u in utterances]
            for i in range(1, len(turns)):
                samples.append(ConversationSample(
                    context=turns[max(0, i - 3):i],
                    response=turns[i],
                ))
    return samples


def load_daily_dialog() -> List[ConversationSample]:
    dataset = load_dataset("daily_dialog", trust_remote_code=True)
    samples = []
    for split in ["train", "validation"]:
        for dialog in dataset[split]["dialog"]:
            turns = [t.strip() for t in dialog]
            for i in range(1, len(turns)):
                samples.append(ConversationSample(
                    context=turns[max(0, i - 3):i],
                    response=turns[i],
                ))
    return samples


class ChatDataset(Dataset):
    def __init__(self, samples: List[ConversationSample], tokenizer, max_length: int = MAX_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = self._encode(samples)

    def _encode(self, samples: List[ConversationSample]):
        encodings = []
        eos = self.tokenizer.eos_token
        for s in tqdm(samples, desc="Tokenizing"):
            text = eos.join(s.context + [s.response]) + eos
            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            encodings.append({
                "input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
            })
        return encodings

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        item = self.encodings[idx]
        input_ids = item["input_ids"]
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": item["attention_mask"],
            "labels": labels,
        }


def build_dataset(model_name: str = "microsoft/DialoGPT-medium"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading EmpatheticDialogues...")
    emp_samples = load_empathetic_dialogues()
    print(f"  {len(emp_samples)} samples")

    print("Loading DailyDialog...")
    dd_samples = load_daily_dialog()
    print(f"  {len(dd_samples)} samples")

    all_samples = emp_samples + dd_samples
    print(f"Total: {len(all_samples)} samples")

    split = int(0.9 * len(all_samples))
    train_dataset = ChatDataset(all_samples[:split], tokenizer)
    val_dataset = ChatDataset(all_samples[split:], tokenizer)

    os.makedirs(DATA_DIR, exist_ok=True)
    torch.save(train_dataset, os.path.join(DATA_DIR, "train.pt"))
    torch.save(val_dataset, os.path.join(DATA_DIR, "val.pt"))
    print(f"Saved datasets to {DATA_DIR}")

    return train_dataset, val_dataset


if __name__ == "__main__":
    build_dataset()
