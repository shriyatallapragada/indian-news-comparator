import torch
import torch.nn as nn
from ncps.torch import LTC
from ncps.wirings import AutoNCP

class NewsComparatorBrain(nn.Module):
    """
    The Liquid Neural Network that takes IndicBERT embeddings and 
    analyzes the temporal flow/framing of the news article.
    """
    # Changed to 64 total neurons, 32 output neurons
    def __init__(self, input_dim=768, total_neurons=64, output_dim=32):
        super(NewsComparatorBrain, self).__init__()
        
        # 1. Device Management
        self.device = self._get_device()
        
        # 2. The Biological Wiring
        # Out of 64 total cells, it will automatically figure out how many 
        # sensory and interneurons it needs to produce a 32-neuron output.
        wiring = AutoNCP(total_neurons, output_size=output_dim)
        
        # 3. The Liquid Time-Constant (LTC) Cell
        self.liquid_layer = LTC(input_dim, wiring).to(self.device)
        
        print(f"Liquid Brain successfully wired with {total_neurons} total neurons on: {self.device}")

    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def forward(self, bert_embeddings):
        """
        Processes the sequence of words to output a final 'Narrative State'.
        """
        bert_embeddings = bert_embeddings.to(self.device)
        
        # Pass the vector through the flowing liquid network
        out, hidden_state = self.liquid_layer(bert_embeddings)
        
        # We only care about the very last hidden state (the model's "final thought")
        final_thought = out[:, -1, :] 
        return final_thought

    def calculate_divergence(self, state_a, state_b):
        """
        Calculates how mathematically different two articles are.
        Returns a score from 0.0 (Identical) to 1.0 (Completely Divergent).
        """
        similarity = torch.nn.functional.cosine_similarity(state_a, state_b, dim=1)
        divergence = (1 - similarity) / 2
        return divergence.item()
    def save_brain(self, file_path="news_brain.pth"):
        """Saves the trained Liquid Neurons to a file."""
        torch.save(self.state_dict(), file_path)
        print(f"Brain successfully saved to {file_path}")

    def load_brain(self, file_path="news_brain.pth"):
        """Loads pre-trained weights into the Liquid Neurons."""
        # map_location ensures it loads correctly whether on CPU or MPS
        self.load_state_dict(torch.load(file_path, map_location=self.device, weights_only=True))
        print(f"Brain successfully loaded from {file_path}")