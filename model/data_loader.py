import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class IndianNewsDataset(Dataset):
    """
    A PyTorch Dataset class that reads Ritvik's CSV and serves it up
    one Triplet (Anchor, Positive, Negative) at a time.
    """
    def __init__(self, csv_file_path):
        print(f"Loading dataset from {csv_file_path}...")
        # Read the CSV using Pandas
        self.data = pd.read_csv(csv_file_path)
        
        # Ensure the CSV has the columns we expect
        expected_columns = ['anchor_text', 'positive_text', 'negative_text']
        for col in expected_columns:
            if col not in self.data.columns:
                raise ValueError(f"CSV is missing the required column: {col}")
                
        print(f"Successfully loaded {len(self.data)} article triplets.")

    def __len__(self):
        # Tells PyTorch how many total examples we have
        return len(self.data)

    def __getitem__(self, idx):
        # Grabs a single row of data based on the index
        row = self.data.iloc[idx]
        
        return {
            "anchor": row['anchor_text'],
            "positive": row['positive_text'],
            "negative": row['negative_text']
        }

def get_news_dataloader(csv_file_path, batch_size=8):
    """
    Wraps the Dataset in a PyTorch DataLoader to handle automatic batching.
    Batch size 8 is a safe starting point for an M4 Mac.
    """
    dataset = IndianNewsDataset(csv_file_path)
    
    # shuffle=True ensures the model doesn't memorize the order of the CSV
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
