import torch
import torch.nn as nn
import torch.optim as optim
from embedder import IndicNewsEmbedder
from liquid_brain import NewsComparatorBrain

def run_dummy_training():
    print("--- INITIALIZING TRAINING SEQUENCE ---")
    
    # 1. Load our two Bits
    embedder = IndicNewsEmbedder()
    brain = NewsComparatorBrain()
    
    # 2. The Optimizer (The Teacher)
    # AdamW is the industry standard for updating neural networks.
    # lr=0.01 is the "Learning Rate" (how big of a step it takes when correcting a mistake)
    optimizer = optim.AdamW(brain.parameters(), lr=0.01)
    
    # 3. The Loss Function (The Grader)
    # Triplet margin loss actively pushes the 'Negative' away from the 'Anchor'
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    
    # 4. Our Dummy "Indian News" Dataset
    anchor_text = "The Election Commission announced the voting dates for the state assembly today."
    positive_text = "State assembly polling dates were officially released by the Election Commission this morning."
    negative_text = "In a desperate bid to delay their inevitable defeat, the corrupt establishment finally announced the election dates."

    print("\n--- EXTRACTING VECTORS (NO GRADIANT) ---")
    # We don't want to train IndicBERT, so we just extract the math once to save M4 Memory
    with torch.no_grad():
        anchor_vec = embedder.get_embeddings(anchor_text)
        pos_vec = embedder.get_embeddings(positive_text)
        neg_vec = embedder.get_embeddings(negative_text)

    print("\n--- STARTING TRAINING LOOP (10 EPOCHS) ---")
    # An "Epoch" is one full read-through of the data
    for epoch in range(1, 11):
        # Step A: Clear the whiteboard (reset the math from the last loop)
        optimizer.zero_grad()
        
        # Step B: The model makes its current guess
        anchor_state = brain(anchor_vec)
        pos_state = brain(pos_vec)
        neg_state = brain(neg_vec)
        
        # Step C: The Grader checks how wrong the model is
        loss = criterion(anchor_state, pos_state, neg_state)
        
        # Step D: Backpropagation (The actual 'Learning')
        # This calculates exactly which of the 64 neurons caused the mistake
        loss.backward()
        
        # Step E: Update the weights
        optimizer.step()
        
        # Print the progress
        print(f"Epoch {epoch}/10 | Loss: {loss.item():.4f}")

    print("\n--- TRAINING COMPLETE ---")
    print("Let's test the Divergence Score now that it has learned!")
    
    # Test how far apart the Anchor and Negative are now
    final_divergence = brain.calculate_divergence(anchor_state, neg_state)
    print(f"Post-Training Divergence Score: {final_divergence:.4f} (Should be higher than 0!)")

if __name__ == "__main__":
    run_dummy_training()