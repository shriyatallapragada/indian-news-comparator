"""
bias_classifier.py — Full inference pipeline combining IndicBERT,
LiquidBrain (LTC), and a classification head.

Architecture:
    text
      → IndicBERT (last_hidden_state, shape: batch × seq × 768)
      → LiquidBrain LTC (processes full token sequence, shape: batch × 32)
      → Linear(32 → 3)  [Left, Center, Right]
      → softmax → probabilities
      → continuous bias score on -5..+5 axis

The score is computed as:
    score = (P_right - P_left) * 5
so it is grounded in what the model actually learned, not hand-crafted anchors.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.liquid_brain import NewsComparatorBrain

# Saved weights path
_CLASSIFIER_PATH = os.path.join(
    os.path.dirname(__file__), "..", "train", "bias_classifier.pth"
)

# Label order — must match training
LABELS = ["Left", "Center", "Right"]   # index 0, 1, 2


class BiasClassifier(nn.Module):
    """
    Thin classification head that sits on top of the LiquidBrain output.
    Input:  (batch, 32)  — LiquidBrain's final thought vector
    Output: (batch, 3)   — raw logits for [Left, Center, Right]
    """

    def __init__(self, input_dim: int = 32, num_classes: int = 3):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.to(next(self.parameters()).device))


class FullBiasPipeline(nn.Module):
    """
    End-to-end module: LiquidBrain → BiasClassifier.

    The IndicBERT embedder is kept separate (it lives in IndicNewsEmbedder)
    so we don't double-load the transformer weights.  This module receives
    the already-computed last_hidden_state tensor from IndicBERT.

    Forward input:  (batch, seq, 768)  — IndicBERT last_hidden_state
    Forward output: (batch, 3)         — logits
    """

    def __init__(self):
        super().__init__()
        self.brain      = NewsComparatorBrain(input_dim=768,
                                              total_neurons=64,
                                              output_dim=32)
        self.classifier = BiasClassifier(input_dim=32, num_classes=3)
        # Move classifier to same device as brain
        device = self.brain.device
        self.classifier = self.classifier.to(device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (batch, seq, 768) from IndicBERT
        returns logits: (batch, 3)
        """
        narrative = self.brain(hidden_states)   # (batch, 32)
        logits    = self.classifier(narrative)  # (batch, 3)
        return logits

    def predict_score(self, hidden_states: torch.Tensor) -> float:
        """
        Convenience method for inference.
        Returns a single float on the -5..+5 axis.
        """
        with torch.no_grad():
            logits = self.forward(hidden_states)          # (1, 3)
            probs  = F.softmax(logits, dim=-1).squeeze(0) # (3,)

        p_left   = probs[0].item()   # index 0 = Left
        p_right  = probs[2].item()   # index 2 = Right
        score    = (p_right - p_left) * 5
        return round(score, 2)

    def predict_label(self, hidden_states: torch.Tensor) -> str:
        """Returns the predicted bias label string."""
        with torch.no_grad():
            logits = self.forward(hidden_states)
            idx    = logits.argmax(dim=-1).item()
        return LABELS[idx]

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str = _CLASSIFIER_PATH) -> None:
        torch.save(self.state_dict(), path)
        print(f"[BiasClassifier] Saved to {path}")

    def load(self, path: str = _CLASSIFIER_PATH) -> bool:
        """Loads weights. Returns True if successful, False if file missing."""
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            print(f"[BiasClassifier] No saved weights at {abs_path} — using random init")
            return False
        device = next(self.parameters()).device
        self.load_state_dict(
            torch.load(abs_path, map_location=device, weights_only=True)
        )
        print(f"[BiasClassifier] Loaded weights from {abs_path}")
        return True
