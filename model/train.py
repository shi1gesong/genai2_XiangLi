"""
Fine-tuning script for EchoHeart.
Usage:
    python model/train.py --epochs 3 --use_wandb
"""

import argparse
import os
import sys

# Ensure project root is in path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from model.model import EchoHeart
from data.preprocess import build_dataset, DATA_DIR, ChatDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="microsoft/DialoGPT-medium")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--output_dir", default="checkpoints")
    return p.parse_args()


def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.use_wandb:
        import wandb
        wandb.init(project="echoheart", config=vars(args))

    # Load or build datasets
    train_path = os.path.join(DATA_DIR, "train.pt")
    val_path = os.path.join(DATA_DIR, "val.pt")
    if os.path.exists(train_path) and os.path.exists(val_path):
        print("Loading cached datasets...")
        train_ds = torch.load(train_path, weights_only=False)
        val_ds = torch.load(val_path, weights_only=False)
    else:
        train_ds, val_ds = build_dataset(args.model_name)

    num_workers = 2 if os.name != "nt" else 0  # Windows doesn't support num_workers>0 well
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            num_workers=num_workers, pin_memory=pin_memory)

    model = EchoHeart(dialogpt_name=args.model_name).to(device).to(torch.bfloat16)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
        eps=1e-8, weight_decay=0.01
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            if torch.isnan(loss):
                print(f"NaN loss at step {step}!")
                print("input_ids max:", input_ids.max(), "min:", input_ids.min())
                print("logits max:", outputs.logits.max(), "min:", outputs.logits.min())
                break

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if step % 100 == 0:
                avg = total_loss / (step + 1)
                print(f"Epoch {epoch} | Step {step}/{len(train_loader)} | Loss {avg:.4f}")
                if args.use_wandb:
                    import wandb
                    wandb.log({"train/loss": avg, "train/step": step + (epoch - 1) * len(train_loader)})

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch} | Val Loss: {val_loss:.4f}")

        if args.use_wandb:
            import wandb
            wandb.log({"val/loss": val_loss, "val/perplexity": torch.exp(torch.tensor(val_loss)).item(), "epoch": epoch})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            # Also save tokenizer for inference
            model.tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))
            print(f"  Saved best model to {ckpt_path}")

    print("Training complete.")
    if args.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    train()
