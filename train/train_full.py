"""
train_full.py — Trains the complete IndicBERT → LiquidBrain → BiasClassifier
pipeline on the existing labeled dataset.

Two-stage training:
  Stage 1 (epochs 1-3): Train only LiquidBrain + classifier head.
                         IndicBERT is fully frozen. Fast, stable.
  Stage 2 (epochs 4-5): Unfreeze IndicBERT's last layer group + pooler.
                         Fine-tune end-to-end at a lower LR.

Usage:
    cd train
    python train_full.py

Output:
    train/bias_classifier.pth   — LiquidBrain + classifier head weights
    train/fine_tuned_bias_model.pth — updated IndicBERT weights
"""

import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer, AutoModel

# ── Path setup ─────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from model.bias_classifier import FullBiasPipeline, LABELS

# ── Hyperparameters ────────────────────────────────────────────────────────
STAGE1_EPOCHS  = 3       # LiquidBrain + head only
STAGE2_EPOCHS  = 2       # end-to-end fine-tune
BATCH_SIZE     = 4       # small batch — LTC is memory-heavy
LR_HEAD        = 1e-3    # higher LR for randomly-initialised head
LR_FINETUNE    = 2e-5    # lower LR for IndicBERT fine-tune
MAX_LEN        = 128     # token length (LTC processes all 128 steps)
LOG_EVERY      = 20
MODEL_NAME     = "ai4bharat/indic-bert"
CSV_PATH       = os.path.join(_ROOT, "data", "training_triplets_large.csv")
BERT_SAVE      = os.path.join(_ROOT, "train", "fine_tuned_bias_model.pth")
HEAD_SAVE      = os.path.join(_ROOT, "train", "bias_classifier.pth")

# ── Device ─────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Training on: {device}")

# ── Dataset ────────────────────────────────────────────────────────────────

class LabeledArticleDataset(Dataset):
    """
    Simple labeled dataset — each item is (text, label_index).
    No triplet construction needed since we're doing classification.
    """
    LABEL_MAP = {"Left": 0, "Center": 1, "Right": 2}

    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["text", "bias_label"])
        df["text"] = df["text"].str.strip()
        df = df[df["text"].str.len() > 20]

        # Normalise any legacy label variants
        df["bias_label"] = df["bias_label"].replace({
            "Neutral":  "Center",
            "Centrist": "Center",
        })
        df = df[df["bias_label"].isin(self.LABEL_MAP)]
        self.data = df.reset_index(drop=True)

        counts = df["bias_label"].value_counts().to_dict()
        print(f"[Dataset] {len(self.data)} articles — {counts}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row   = self.data.iloc[idx]
        text  = row["text"]
        label = self.LABEL_MAP[row["bias_label"]]
        return text, label


# ── Model setup ────────────────────────────────────────────────────────────
print(f"\nLoading {MODEL_NAME}…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert      = AutoModel.from_pretrained(MODEL_NAME).to(device)

# Freeze all IndicBERT params for Stage 1
for param in bert.parameters():
    param.requires_grad = False

pipeline = FullBiasPipeline().to(device)
# Try loading existing classifier weights to resume training
pipeline.load(HEAD_SAVE)

# ── Loss ───────────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()

# ── Dataset & DataLoader ───────────────────────────────────────────────────
dataset    = LabeledArticleDataset(CSV_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=True)


def encode_batch(texts: list) -> torch.Tensor:
    """
    Tokenise a list of strings and return IndicBERT last_hidden_state.
    Shape: (batch, MAX_LEN, 768)
    """
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = bert(**inputs)
    return outputs.last_hidden_state   # (batch, MAX_LEN, 768)


def run_epoch(optimizer, epoch_num, total_epochs):
    pipeline.train()
    bert.train()
    epoch_loss = 0.0
    correct    = 0
    total      = 0

    for step, (texts, labels) in enumerate(dataloader, 1):
        labels = labels.to(device)
        optimizer.zero_grad()

        hidden = encode_batch(list(texts))   # (batch, 128, 768)
        logits = pipeline(hidden)            # (batch, 3)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds   = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        if step % LOG_EVERY == 0 or step == len(dataloader):
            acc = correct / total * 100
            avg = epoch_loss / step
            print(f"  Epoch {epoch_num}/{total_epochs} | "
                  f"Step {step}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f} | Avg: {avg:.4f} | Acc: {acc:.1f}%")

    return epoch_loss / len(dataloader), correct / total * 100


# ── Stage 1: Train head + LiquidBrain only ─────────────────────────────────
print(f"\n{'='*60}")
print(f"STAGE 1 — Training LiquidBrain + classifier head ({STAGE1_EPOCHS} epochs)")
print(f"IndicBERT is frozen.")
print(f"{'='*60}\n")

optimizer_s1 = torch.optim.AdamW(pipeline.parameters(), lr=LR_HEAD)

for epoch in range(1, STAGE1_EPOCHS + 1):
    avg_loss, acc = run_epoch(optimizer_s1, epoch, STAGE1_EPOCHS)
    print(f"── Stage 1 Epoch {epoch} complete | Loss: {avg_loss:.4f} | Acc: {acc:.1f}%\n")

# ── Stage 2: Unfreeze IndicBERT last layer + pooler ────────────────────────
print(f"\n{'='*60}")
print(f"STAGE 2 — End-to-end fine-tune ({STAGE2_EPOCHS} epochs)")
print(f"Unfreezing IndicBERT last layer group + pooler.")
print(f"{'='*60}\n")

for param in bert.encoder.albert_layer_groups[-1].parameters():
    param.requires_grad = True
for param in bert.pooler.parameters():
    param.requires_grad = True

trainable = sum(p.numel() for p in bert.parameters() if p.requires_grad)
total_p   = sum(p.numel() for p in bert.parameters())
print(f"IndicBERT trainable: {trainable:,} / {total_p:,} params\n")

optimizer_s2 = torch.optim.AdamW([
    {"params": pipeline.parameters(),                          "lr": LR_HEAD},
    {"params": filter(lambda p: p.requires_grad,
                      bert.parameters()),                      "lr": LR_FINETUNE},
])

for epoch in range(1, STAGE2_EPOCHS + 1):
    avg_loss, acc = run_epoch(optimizer_s2, epoch, STAGE2_EPOCHS)
    print(f"── Stage 2 Epoch {epoch} complete | Loss: {avg_loss:.4f} | Acc: {acc:.1f}%\n")

# ── Save ───────────────────────────────────────────────────────────────────
pipeline.save(HEAD_SAVE)
torch.save(bert.state_dict(), BERT_SAVE)
print(f"\nIndicBERT weights saved to {BERT_SAVE}")
print("Training complete.")
