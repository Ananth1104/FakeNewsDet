import os
import json
import openai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")
openai.api_key = API_KEY
openai.api_base = os.getenv("MISTRAL_API_BASE", "https://api.mistral.ai/v1")

def generate_summary(input_path):
    """
    Reads llm_input.json and generates a summary using Mistral's LLM.
    The prompt instructs the LLM to compare the user's claim with the verified news article,
    determine whether the claim is supported by credible evidence, and produce a natural language analysis.
    It should start with either:
      "Your claim that [user claim] is True..." or "Your claim that [user claim] is False..."
    and finally provide a very brief TL;DR summary.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    fake_text = data.get("fake_text", "")
    real_text = data.get("real_text", "")
    similarity = data.get("similarity_score", 0)

    prompt = (
        f"You are a fact-checking assistant. A user submitted the following claim:\n\n"
        f"\"{fake_text}\"\n\n"
        f"A verified news article related to this topic is:\n\n"
        f"\"{real_text}\"\n\n"
        f"The similarity score between the texts is {similarity} (scale 0 to 1).\n\n"
        "Analyze the two texts carefully. If the verified news article supports the user's claim, "
        "start your response with 'Your claim that [user claim] is True' and explain how the verified article confirms it. "
        "If the verified news article does not support the claim, start your response with 'Your claim that [user claim] is False' "
        "and explain the discrepancies or missing evidence. Finally, provide a concise TL;DR summary in one sentence."
    )

    response = openai.ChatCompletion.create(
        model="mistral-tiny",  # Adjust the model if needed
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    summary = response.choices[0].message.content
    return summary

if __name__ == "__main__":
    input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "llm_input.json"))
    summary = generate_summary(input_path)
    print("LLM Summary:")
    print(summary)
