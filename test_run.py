from model.embedder import IndicNewsEmbedder
from model.liquid_brain import NewsComparatorBrain

# 1. Boot up the engines
print("--- BOOTING AI ENGINES ---")
embedder = IndicNewsEmbedder()
brain = NewsComparatorBrain()

# 2. Simulate Ritvik fetching two articles about the same event
article_neutral = "The Supreme Court delivered a verdict today regarding the electoral bonds issue, stating the scheme is unconstitutional."
article_biased = "In a massive blow to the ruling establishment, the Supreme Court struck down the highly controversial and opaque electoral bonds scheme."

# 3. Convert text to Math (Bit 1)
print("\n--- EXTRACTING EMBEDDINGS ---")
vec_neutral = embedder.get_embeddings(article_neutral)
vec_biased = embedder.get_embeddings(article_biased)

# 4. Pass through the Liquid Network (Bit 2)
print("\n--- ANALYZING NARRATIVE FLOW ---")
state_neutral = brain(vec_neutral)
state_biased = brain(vec_biased)

# 5. Calculate the result
divergence_score = brain.calculate_divergence(state_neutral, state_biased)

print("\n=== FINAL RESULT ===")
print(f"Narrative Divergence Score: {divergence_score:.4f} (Scale: 0.0 to 1.0)")
print("If the score is close to 0, they are written similarly. If higher, the framing differs.")