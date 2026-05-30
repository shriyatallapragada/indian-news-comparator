import pandas as pd
import torch
from model.embedder import IndicNewsEmbedder
from model.liquid_brain import NewsComparatorBrain

def simulate_user_experience():
    print("--- LOADING ML ARCHITECTURE ---")
    embedder = IndicNewsEmbedder()
    brain = NewsComparatorBrain()
    
    print("\n--- LOADING RITVIK'S DATABASE ---")
    # Simulate Ritvik's backend database connection
    db = pd.read_csv("data/mock_database.csv")
    print(f"Connected to database. {len(db)} articles available for recommendation.")

    # ==========================================
    # THE SIMULATION: Jahnavi's Extension triggers
    # ==========================================
    print("\n[USER ACTION] User is currently reading a highly biased article...")
    user_reading_topic = "Elections"
    user_reading_text = "In a massive blow to the ruling party, the EC was forced to announce the election dates today, signaling the end of their regime."
    
    print("\n--- ML INFERENCE (YOUR JOB) ---")
    # 1. Convert the user's article to a narrative state
    user_vec = embedder.get_embeddings(user_reading_text)
    user_state = brain(user_vec)

    print("\n--- CONNECTION SEARCH (RITVIK'S JOB) ---")
    # 2. Filter Ritvik's database for the exact same topic
    relevant_articles = db[db['topic'] == user_reading_topic]
    
    recommendation = None
    highest_divergence = 0.0

    # 3. Scan the database to find the best "Alternative Perspective"
    for index, row in relevant_articles.iterrows():
        # We only want to recommend Neutral or Centrist articles to keep it professional
        if row['bias_label'] in ['Neutral', 'Centrist']:
            
            # Extract the vector for the database article
            db_vec = embedder.get_embeddings(row['text'])
            db_state = brain(db_vec)
            
            # Calculate how mathematically different it is from what the user is reading
            divergence = brain.calculate_divergence(user_state, db_state)
            
            # We want to recommend the article that provides the MOST different (neutral) framing
            if divergence > highest_divergence:
                highest_divergence = divergence
                recommendation = row

    # ==========================================
    # THE OUTPUT: Sending data back to Jahnavi
    # ==========================================
    print("\n=== FINAL CHROME EXTENSION PAYLOAD ===")
    if recommendation is not None:
        print(f"User is reading: A highly subjective take on {user_reading_topic}.")
        print(f"Divergence Score: {highest_divergence:.4f}")
        print(f"Action: Recommending a Cross-Perspective Alternative.")
        print(f"--> RECOMMENDED SOURCE: {recommendation['source']} ({recommendation['bias_label']})")
        print(f"--> RECOMMENDED URL TO DISPLAY: {recommendation['url']}")
    else:
        print("No neutral alternatives found in database for this topic.")

if __name__ == "__main__":
    simulate_user_experience()