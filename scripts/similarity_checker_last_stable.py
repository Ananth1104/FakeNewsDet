import json
import os
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import subprocess

# Set up absolute paths relative to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")
USER_INPUT_PATH = os.path.join(DATA_FOLDER, "user_input.json")
LLM_INPUT_PATH = os.path.join(DATA_FOLDER, "llm_input.json")
BEST_ARTICLE_PATH = os.path.join(DATA_FOLDER, "best_article.json")
CHROMADB_PATH = os.path.join(DATA_FOLDER, "chroma_db")
LLM_SCRIPT = os.path.join(os.path.dirname(__file__), "llm_summary.py")  # LLM summarizer in scripts/

# Load the SentenceTransformer model for encoding queries
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_user_query():
    """Loads the user query (the fake news input) from user_input.json."""
    if not os.path.exists(USER_INPUT_PATH):
        print("User input file not found.")
        return None
    with open(USER_INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    query = data.get("user_query", "").strip()
    if not query:
        print("User query is empty.")
        return None
    return query

def load_raw_articles():
    """Loads the raw articles (scraped with full_text) from raw_articles.json."""
    raw_path = os.path.join(DATA_FOLDER, "raw_articles.json")
    if os.path.exists(raw_path):
        with open(raw_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def check_similarity():
    print("\n🔎 Checking similarity...")

    # Load user query
    query = load_user_query()
    if not query:
        print("No valid user query found. Exiting.")
        return None

    # Initialize ChromaDB and get collection
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    collection = client.get_or_create_collection("news_articles")

    # Encode the user query
    query_embedding = model.encode(query).reshape(1, -1)

    # Query ChromaDB to get the top 1 matching article.
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=1,
        include=["documents", "metadatas", "distances"]
    )

    # Debug print
    print(f"Raw ChromaDB results: {results}")

    # Validate results
    if (not results["documents"] or not results["metadatas"] or 
        len(results["metadatas"][0]) == 0):
        print("No similar articles found in the database.")
        return None

    # Extract best match details
    best_metadata = results["metadatas"][0][0]
    best_distance = results["distances"][0][0]
    similarity = round(1 - best_distance, 4)

    # Expect metadata to contain "article_id"; fallback to "url" if missing.
    best_article_id = best_metadata.get("article_id") or best_metadata.get("url")
    if not best_article_id:
        print("No valid article identifier found in metadata.")
        return None

    print(f"Found similar article with ID: {best_article_id} and similarity: {similarity}")

    # Load raw articles and find the matching article by article_id (or URL)
    raw_articles = load_raw_articles()
    best_article_text = None
    for article in raw_articles:
        if article.get("article_id") == best_article_id or article.get("url") == best_article_id:
            best_article_text = article.get("full_text", "")
            break

    if not best_article_text:
        print("No matching article full_text found in raw_articles.json.")
        return None

    print(f"Most similar article found with similarity: {similarity:.4f}")

    # Save best article to best_article.json
    with open(BEST_ARTICLE_PATH, "w", encoding="utf-8") as f:
        json.dump({"real_text": best_article_text, "similarity_score": similarity}, f, indent=4)
    print(f"Saved best article to {BEST_ARTICLE_PATH}")

    # Prepare LLM input data
    llm_input = {
        "fake_text": query,
        "real_text": best_article_text,
        "similarity_score": similarity
    }
    with open(LLM_INPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(llm_input, f, indent=4)
    print(f"Saved LLM input data to {LLM_INPUT_PATH}")

    # Run the LLM summarizer script, passing the llm_input.json file path as argument.
    try:
        subprocess.run(["python", LLM_SCRIPT, LLM_INPUT_PATH], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running LLM script: {e}")

    return llm_input

if __name__ == "__main__":
    check_similarity()

