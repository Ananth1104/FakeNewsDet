import os
import json
import hashlib
from flask import Flask, render_template, request, redirect, url_for, flash, session
from pymongo import MongoClient
from scripts.news_fetcher import get_news_articles
from scripts.scraper import scrape_articles
from scripts.vectorizer import vectorize_articles
from scripts.similarity_checker import check_similarity
from scripts.llm_summary import generate_summary
from scripts.crypto_utils import encrypt_query, decrypt_query  # AES Encryption

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Replace with a strong secret key

# MongoDB setup
mongo_uri = "mongodb+srv://nishanthsaravanamurali2004:I5Ft3s3XkUf18CdD@cluster1.uocyk.mongodb.net/"
client = MongoClient(mongo_uri)
db = client["Fake_news_detection"]
user_collection = db["Users"]
history_collection = db["Search_history"]

# Define absolute paths for the data folder
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")
USER_INPUT_PATH = os.path.join(DATA_FOLDER, "user_input.json")
LLM_INPUT_PATH = os.path.join(DATA_FOLDER, "llm_input.json")


def hash_password(password):
    """ Hash password using SHA-256 """
    return hashlib.sha256(password.encode()).hexdigest()


@app.route("/")
def index():
    """ Home page (Redirects to login if not logged in) """
    if "user_id" not in session:
        #print(session["user_id"])
        return redirect(url_for("login"))
    return redirect(url_for("dashboard"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    """ User Signup """
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        hashed_password = hash_password(password)

        # Check if user already exists
        if user_collection.find_one({"mail_id": email}):
            flash("Email already registered. Please log in.", "warning")
            return redirect(url_for("login"))

        # Insert new user
        user_data = {
            "name": name,
            "mail_id": email,
            "password": hashed_password,
            "search_history_id": []
        }
        user_collection.insert_one(user_data)

        flash("Signup successful! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """ User Login """
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        hashed_password = hash_password(password)

        # Validate credentials
        user = user_collection.find_one({"mail_id": email, "password": hashed_password})
        if user:
            session["user_id"] = str(user["_id"])  # Store user ID in session
            session["user_name"] = user["name"]
            session["user_email"] = user["mail_id"]
            flash("Login successful!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid email or password", "danger")

    return render_template("login.html")


@app.route("/dashboard")
def dashboard():
    """ User Dashboard """
    if "user_id" not in session:
        flash("Please log in to continue.", "warning")
        return redirect(url_for("login"))

    user_id = session["user_id"]
    user_name = session["user_name"]
    user_email = session["user_email"]

    # Fetch user's search history
    search_history_records = history_collection.find({"user_id": user_id})
    search_history = []
    for record in search_history_records:
        decrypted_query = decrypt_query(record["encrypted_query"])  # Decrypt search queries
        search_history.append(decrypted_query)

    return render_template("dashboard.html", user_name=user_name, user_email=user_email, search_history=search_history)


@app.route("/logout")
def logout():
    """ Logout user """
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("index"))


@app.route("/submit", methods=["GET", "POST"])
def submit():
    """ News Verification (Restricted to Logged-in Users) """
    if "user_id" not in session:
        flash("Please log in to verify news.", "warning")
        return redirect(url_for("login"))

    user_id = session["user_id"]

    if request.method == "POST":
        news_claim = request.form.get("news_claim", "").strip()
        if not news_claim:
            flash("Please enter a news claim.", "warning")
            return redirect(url_for("submit"))

        # Encrypt and store search query
        encrypted_query = encrypt_query(news_claim)
        history_collection.insert_one({"user_id": user_id, "encrypted_query": encrypted_query})

        # Save user query for pipeline processing
        os.makedirs(DATA_FOLDER, exist_ok=True)
        with open(USER_INPUT_PATH, "w", encoding="utf-8") as f:
            json.dump({"user_query": news_claim}, f, indent=4)
        flash("News claim submitted. Processing...", "info")

        # Pipeline steps
        fetched_articles = get_news_articles(news_claim)
        if not fetched_articles:
            flash("Failed to fetch articles from NewsAPI.", "danger")
            return redirect(url_for("submit"))

        if not scrape_articles():
            flash("Failed to scrape articles.", "danger")
            return redirect(url_for("submit"))

        if not vectorize_articles():
            flash("Failed to vectorize articles.", "danger")
            return redirect(url_for("submit"))

        pipeline_result = check_similarity(news_claim)
        if not pipeline_result:
            flash("No similar article found. Try a different claim.", "danger")
            return redirect(url_for("submit"))

        summary = generate_summary(LLM_INPUT_PATH)
        pipeline_result["summary"] = summary

        # Save final result
        with open(LLM_INPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(pipeline_result, f, indent=4)

        flash("Analysis complete.", "success")
        return redirect(url_for("results"))

    return render_template("submit.html")


@app.route("/results")
def results():
    """ Display Results """
    if os.path.exists(LLM_INPUT_PATH):
        with open(LLM_INPUT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    return render_template("results.html",
                           summary=data.get("summary", "No summary available."),
                           fake_text=data.get("fake_text", ""),
                           real_text=data.get("real_text", ""),
                           similarity_score=data.get("similarity_score", ""))


if __name__ == "__main__":
    app.run(debug=True)
