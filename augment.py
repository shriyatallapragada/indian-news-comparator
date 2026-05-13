import pandas as pd

def augment_dataset():
    print("Loading original 137 articles...")
    df = pd.read_csv("data/training_triplets.csv")
    
    new_rows = []
    
    for _, row in df.iterrows():
        text = str(row['text'])
        bias = row['bias_label']
        
        # Split the article by paragraphs (double line breaks)
        # We only keep paragraphs that are longer than 100 characters so we don't train on junk
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 100]
        
        for p in paragraphs:
            new_rows.append({'text': p, 'bias_label': bias})
            
    out_df = pd.DataFrame(new_rows)
    out_df.to_csv("data/training_triplets_large.csv", index=False)
    print(f"Boom! Expanded 137 articles into {len(out_df)} clean training paragraphs!")

if __name__ == "__main__":
    augment_dataset()