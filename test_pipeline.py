import sys
sys.path.insert(0, '.')

from model.bias_classifier import FullBiasPipeline
from model.embedder import IndicNewsEmbedder

print("Loading pipeline...")
p = FullBiasPipeline()
loaded = p.load('train/bias_classifier.pth')
print(f"Weights loaded: {loaded}")

print("\nLoading embedder...")
e = IndicNewsEmbedder()

print("\nRunning test score...")
hidden = e.get_embeddings("The government announced new economic reforms today.")
score = p.predict_score(hidden)
label = p.predict_label(hidden)
print(f"Score: {score:+.2f}, Label: {label}")
print("\nPipeline OK")
