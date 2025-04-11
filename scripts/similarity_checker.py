import json
import os
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Define absolute path for the data folder
DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
LLM_INPUT_PATH = os.path.join(DATA_FOLDER, "llm_input.json")
CHROMADB_PATH = os.path.join(DATA_FOLDER, "chroma_db")

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_raw_articles():
    raw_file = os.path.join(DATA_FOLDER, "raw_articles.json")
    if os.path.exists(raw_file):
        with open(raw_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def check_similarity(user_query):
    """
    Compares the provided user_query with stored article embeddings in ChromaDB
    and returns a dict with keys: fake_text, real_text, similarity_score.
    """
    # Initialize ChromaDB client and collection
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    collection = client.get_or_create_collection("news_articles")

    # Query ChromaDB with the user query
    results = collection.query(
        query_texts=[user_query],
        n_results=1,
        include=["documents", "metadatas", "distances"]
    )

    # Debug: print raw results
    print(f"ChromaDB raw results: {results}")

    if not results["documents"] or not results["metadatas"] or len(results["metadatas"][0]) == 0:
        print("No similar articles found in the database.")
        return None

    # Extract best match details
    best_metadata = results["metadatas"][0][0]
    best_distance = results["distances"][0][0]
    similarity = round(1 - best_distance, 4)

    # Use metadata to get article identifier (fallback to URL)
    best_article_id = best_metadata.get("article_id") or best_metadata.get("url")
    if not best_article_id:
        print("No valid article identifier found in metadata.")
        return None

    # Load raw articles and find the matching article using article_id or URL
    raw_articles = load_raw_articles()
    best_article_text = ""
    for article in raw_articles:
        if article.get("article_id") == best_article_id or article.get("url") == best_article_id:
            best_article_text = article.get("full_text", "")
            break

    if not best_article_text:
        print("No matching article found in raw_articles.json.")
        return None

    # Prepare the pipeline result
    pipeline_result = {
        "fake_text": user_query,
        "real_text": best_article_text,
        "similarity_score": similarity
    }

    # Write the result to llm_input.json for further processing
    with open(LLM_INPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pipeline_result, f, indent=4)

    return pipeline_result

if __name__ == "__main__":
    test_query = "Trump is dead"
    result = check_similarity(test_query)
    print(result)
