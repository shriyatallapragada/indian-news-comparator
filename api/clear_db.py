"""
clear_db.py — clears the news_articles collection via the ChromaDB HTTP server.
Make sure `chroma run --path chroma_db --port 8002` is running first.

Usage:
    python clear_db.py
"""
import chromadb

client = chromadb.HttpClient(host="127.0.0.1", port=8002)
try:
    client.delete_collection("news_articles")
    print("Deleted news_articles collection.")
except Exception as e:
    print(f"Could not delete (may not exist): {e}")

client.get_or_create_collection("news_articles", metadata={"hnsw:space": "cosine"})
print("Fresh news_articles collection created.")
