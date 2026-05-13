import torch
import os
from transformers import AutoTokenizer, AutoModel
import warnings

warnings.filterwarnings("ignore")

# Path to fine-tuned weights — sits in train/ relative to repo root
_FINE_TUNED_PATH = os.path.join(
    os.path.dirname(__file__), "..", "train", "fine_tuned_bias_model.pth"
)

class IndicNewsEmbedder:
    def __init__(self, model_name: str = "ai4bharat/indic-bert"):
        print(f"Loading {model_name} into memory...")
        self.device = self._get_device()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        # Load fine-tuned weights if available
        fine_tuned = os.path.abspath(_FINE_TUNED_PATH)
        if os.path.exists(fine_tuned):
            self.model.load_state_dict(
                torch.load(fine_tuned, map_location=self.device, weights_only=True)
            )
            print(f"Fine-tuned weights loaded from {fine_tuned}")
        else:
            print("No fine-tuned weights found — using base IndicBERT")

        self.model.eval()
        print(f"Embedder successfully loaded on: {self.device}")

    def _get_device(self):
        """Detects if Apple Silicon (MPS) is available, else falls back to CPU."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def get_embeddings(self, text: str):
        """
        Takes a string of text and returns a sequence of vectors representing the words.
        """
        # Step A: Tokenize the text (break it into chunks and turn to ID numbers)
        # padding=True and truncation=True ensure all articles are handled safely
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512, 
            padding="max_length"
        )
        
        # Move the inputs to your M4 GPU
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        # Step B: Pass through IndicBERT
        with torch.no_grad(): # Saves memory since we aren't training BERT
            outputs = self.model(**inputs)
            
        # We want the "last_hidden_state". This is the final mathematical 
        # representation of the sentence context.
        return outputs.last_hidden_state