"""
triplet_loader.py — PyTorch Dataset that builds (anchor, positive, negative)
triplets on-the-fly from a CSV with columns: text, bias_label.

For each anchor article, a positive is sampled from the same bias class
and a negative from a different bias class.
"""

import random
import pandas as pd
from torch.utils.data import Dataset


class BiasTripletDataset(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["text", "bias_label"])
        df["text"] = df["text"].str.strip()
        df = df[df["text"].str.len() > 20]

        self.data = df.reset_index(drop=True)

        # Group indices by bias label for fast sampling
        self.by_label: dict = {}
        for idx, row in self.data.iterrows():
            label = row["bias_label"]
            self.by_label.setdefault(label, []).append(idx)

        self.labels = list(self.by_label.keys())
        print(f"[BiasTripletDataset] Loaded {len(self.data)} articles "
              f"across labels: {self.labels}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor_row  = self.data.iloc[idx]
        anchor_text = anchor_row["text"]
        anchor_label = anchor_row["bias_label"]

        # Positive: same label, different index
        pos_pool = [i for i in self.by_label[anchor_label] if i != idx]
        pos_idx  = random.choice(pos_pool) if pos_pool else idx
        positive_text = self.data.iloc[pos_idx]["text"]

        # Negative: different label
        neg_label = random.choice([l for l in self.labels if l != anchor_label])
        neg_idx   = random.choice(self.by_label[neg_label])
        negative_text = self.data.iloc[neg_idx]["text"]

        return anchor_text, positive_text, negative_text
