"""
train.py — Fine-tunes IndicBERT (ai4bharat/indic-bert) on Indian news bias
using Triplet Margin Loss.

Usage:
    cd train
    python train.py

Output:
    fine_tuned_bias_model.pth  — saved in the train/ directory
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

# ── Path setup ─────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from triplet_loader import BiasTripletDataset

# ── Hyperparameters ────────────────────────────────────────────────────────
EPOCHS      = 5
BATCH_SIZE  = 8
LR          = 2e-5
MARGIN      = 0.5
MAX_LEN     = 128
LOG_EVERY   = 10
MODEL_NAME  = "ai4bharat/indic-bert"
CSV_PATH    = os.path.join(_ROOT, "data", "training_triplets_large.csv")
SAVE_PATH   = os.path.join(os.path.dirname(__file__), "fine_tuned_bias_model.pth")

# ── Device ─────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Training on: {device}")

# ── Model & Tokenizer ──────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME}…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.train()

# ── Layer Freezing ─────────────────────────────────────────────────────────
# Freeze embeddings and all encoder groups
for param in model.embeddings.parameters():
    param.requires_grad = False

for param in model.encoder.parameters():
    param.requires_grad = False

# Unfreeze only the last transformer group and the pooler
for param in model.encoder.albert_layer_groups[-1].parameters():
    param.requires_grad = True

for param in model.pooler.parameters():
    param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Layer freezing: {trainable:,} / {total:,} parameters trainable")

# ── Loss & Optimizer ───────────────────────────────────────────────────────
criterion = nn.TripletMarginLoss(margin=MARGIN, p=2)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR
)

# ── Dataset & DataLoader ───────────────────────────────────────────────────
dataset    = BiasTripletDataset(CSV_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def encode(texts: list) -> torch.Tensor:
    """
    Tokenise a list of strings and return the [CLS] token embedding.
    Shape: (batch_size, 768)
    Gradients are enabled so the model can learn.
    """
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    # [CLS] token is the first token of last_hidden_state
    cls_emb = outputs.last_hidden_state[:, 0, :]   # (batch, 768)
    return cls_emb


# ── Training Loop ──────────────────────────────────────────────────────────
print(f"\nStarting training — {EPOCHS} epochs, {len(dataloader)} steps/epoch\n")

for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0

    for step, (anchors, positives, negatives) in enumerate(dataloader, 1):
        optimizer.zero_grad()

        anchor_emb   = encode(list(anchors))
        positive_emb = encode(list(positives))
        negative_emb = encode(list(negatives))

        loss = criterion(anchor_emb, positive_emb, negative_emb)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if step % LOG_EVERY == 0 or step == len(dataloader):
            avg = epoch_loss / step
            print(f"Epoch {epoch}/{EPOCHS} | Step {step}/{len(dataloader)} "
                  f"| Loss: {loss.item():.4f} | Avg: {avg:.4f}")

    print(f"── Epoch {epoch} complete. Avg loss: {epoch_loss / len(dataloader):.4f}\n")

# ── Save ───────────────────────────────────────────────────────────────────
torch.save(model.state_dict(), SAVE_PATH)
print(f"Model saved to {SAVE_PATH}")
