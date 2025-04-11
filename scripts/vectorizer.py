import json
import os
import chromadb
from sentence_transformers import SentenceTransformer

# Set up absolute paths relative to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")
RAW_ARTICLES_PATH = os.path.join(DATA_FOLDER, "raw_articles.json")
CHROMADB_PATH = os.path.join(DATA_FOLDER, "chroma_db")

# Ensure data folder exists
os.makedirs(DATA_FOLDER, exist_ok=True)

# Initialize ChromaDB client and collection
client = chromadb.PersistentClient(path=CHROMADB_PATH)
collection = client.get_or_create_collection(name="news_articles")

# Load advanced NLP model
model = SentenceTransformer("all-MiniLM-L6-v2")

def vectorize_articles():
    """
    Reads scraped articles from raw_articles.json, generates embeddings using
    SentenceTransformer, and stores them in ChromaDB with metadata.
    """
    if not os.path.exists(RAW_ARTICLES_PATH):
        print("No raw_articles.json found. Run scraper.py first.")
        return False

    with open(RAW_ARTICLES_PATH, "r", encoding="utf-8") as file:
        articles = json.load(file)

    if not articles:
        print("No articles to process.")
        return False

    # (Optional) Clear previous embeddings for a fresh run
    existing_ids = collection.get()["ids"]
    if existing_ids and any(existing_ids):
        collection.delete(ids=existing_ids)
        print(f"Deleted {len(existing_ids)} old embeddings.")
    else:
        print("No existing embeddings to delete. Skipping deletion step.")

    processed = 0
    for article in articles:
        full_text = article.get("full_text", "").strip()
        url = article.get("url", "").strip()
        if not full_text or full_text == "Content not found" or not url:
            print(f"Skipping article with insufficient content or URL: {url}")
            continue

        # Compute embedding using SentenceTransformer
        embedding = model.encode(full_text).tolist()
        # Use URL as a unique article ID (or if available, use article.get("article_id"))
        metadata = {"article_id": url, "url": url, "title": article.get("title", "")}
        # Add article to ChromaDB
        collection.add(ids=[url], embeddings=[embedding], documents=[full_text], metadatas=[metadata])
        processed += 1

    print(f"Stored {processed} articles in ChromaDB.")
    return True

if __name__ == "__main__":
    vectorize_articles()
