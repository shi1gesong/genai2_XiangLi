"""
Dataset loading and preprocessing for EchoHeart.
Uses datasets compatible with datasets>=5.0 (Parquet-based, no legacy scripts):
  - blended_skill_talk  (~76k empathetic multi-turn conversations, Facebook)
  - AlekseyKorshuk/persona-chat (~8k persona-based chit-chat conversations)
"""

import os
from dataclasses import dataclass
from typing import List

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


def load_blended_skill_talk() -> List[ConversationSample]:
    dataset = load_dataset("blended_skill_talk")
    samples = []
    for split in ["train", "validation"]:
        for row in dataset[split]:
            turns = [t.strip() for t in row.get("previous_utterance", []) if t.strip()]
            if len(turns) < 2:
                continue
            for i in range(1, len(turns)):
                samples.append(ConversationSample(
                    context=turns[max(0, i - 3):i],
                    response=turns[i],
                ))
    return samples


def load_persona_chat() -> List[ConversationSample]:
    dataset = load_dataset("AlekseyKorshuk/persona-chat")
    samples = []
    for split in ["train", "validation"]:
        for row in dataset[split]:
            utterances = row.get("utterances", [])
            for utt in utterances:
                history = utt.get("history", [])
                candidates = utt.get("candidates", [])
                if not history or not candidates:
                    continue
                # last candidate is the gold response
                response = candidates[-1].strip()
                context = [h.strip() for h in history[-3:] if h.strip()]
                if context and response:
                    samples.append(ConversationSample(context=context, response=response))
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

    print("Loading BlendedSkillTalk...")
    bst_samples = load_blended_skill_talk()
    print(f"  {len(bst_samples)} samples")

    print("Loading PersonaChat...")
    pc_samples = load_persona_chat()
    print(f"  {len(pc_samples)} samples")

    all_samples = bst_samples + pc_samples
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
