"""
Evaluation utilities for EchoHeart.
Computes perplexity on the validation set and runs qualitative sample generation.
"""

import argparse
import math
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.model import EchoHeart
from data.preprocess import DATA_DIR


def compute_perplexity(model: EchoHeart, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            n_tokens = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


QUALITATIVE_PROMPTS = [
    ["I've been feeling really lonely lately and don't know what to do."],
    ["My dog passed away yesterday. I'm devastated."],
    ["I failed my exam even though I studied so hard. I feel like giving up."],
    ["I just got a promotion at work! I'm so happy!"],
    ["I'm scared about moving to a new city where I don't know anyone."],
]


def run_qualitative(model: EchoHeart, device: torch.device):
    model.eval()
    model.to(device)
    print("\n--- Qualitative Samples ---\n")
    for context in QUALITATIVE_PROMPTS:
        response = model.generate_response(context)
        print(f"User:  {context[-1]}")
        print(f"EchoHeart: {response}")
        print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--model_name", default="microsoft/DialoGPT-medium")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--qualitative", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EchoHeart(dialogpt_name=args.model_name)
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint found, evaluating base model weights.")
    model.to(device)

    val_path = os.path.join(DATA_DIR, "val.pt")
    if os.path.exists(val_path):
        val_ds = torch.load(val_path)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)
        ppl = compute_perplexity(model, val_loader, device)
        print(f"Validation Perplexity: {ppl:.2f}")
    else:
        print("No cached validation set found. Run data/preprocess.py first.")

    if args.qualitative:
        run_qualitative(model, device)


if __name__ == "__main__":
    main()
