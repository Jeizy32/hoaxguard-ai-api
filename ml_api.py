import os
import requests
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model & vectorizer
model = joblib.load("hoax_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Supabase Edge Function config
SUPABASE_EDGE_URL = "https://<your-project>.functions.supabase.co"  # Ganti dengan URL Supabase
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")               # Isi env di Railway


@app.route("/predict", methods=["POST"])
def predict():
    """Prediksi hoax/non-hoax pakai model ML lokal"""
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]

    return jsonify({"prediction": str(prediction)})


@app.route("/analyze", methods=["POST"])
def analyze():
    """Gabungan prediksi ML + analisis Perplexity"""
    data = request.get_json()
    text = data.get("text", "")
    url = data.get("url", "")

    if not text and not url:
        return jsonify({"error": "No text or URL provided"}), 400

    # --- Step 1: Prediksi ML Lokal ---
    prediction = None
    if text:
        X = vectorizer.transform([text])
        prediction = model.predict(X)[0]

    # --- Step 2: Call Supabase Edge Functions ---
    headers = {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }

    payload = {"content": text or url, "url": url, "title": text[:100] if text else ""}

    # Call analyze-news
    try:
        analyze_resp = requests.post(
            f"{SUPABASE_EDGE_URL}/analyze-news",
            headers=headers,
            json=payload,
            timeout=30
        )
        analyze_data = analyze_resp.json() if analyze_resp.ok else {"error": "Perplexity analyze failed"}
    except Exception as e:
        analyze_data = {"error": f"Analyze API error: {str(e)}"}

    # Call search-similar-news
    try:
        search_resp = requests.post(
            f"{SUPABASE_EDGE_URL}/search-similar-news",
            headers=headers,
            json=payload,
            timeout=30
        )
        search_data = search_resp.json() if search_resp.ok else {"error": "Perplexity search failed"}
    except Exception as e:
        search_data = {"error": f"Search API error: {str(e)}"}

    # --- Step 3: Merge hasil ---
    result = {
        "ml_prediction": str(prediction) if prediction is not None else "N/A",
        "ai_analysis": analyze_data,
        "similar_news": search_data.get("similarNews", []),
        "input_url": url
    }

    return jsonify(result)


if __name__ == "__main__":
    # Railway pake PORT dari environment
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
