from flask import Flask, request, jsonify
import joblib

# Load model & vectorizer
model = joblib.load("hoax_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if not data or "content" not in data:
        return jsonify({"error": "No content provided"}), 400
    
    text = data["content"]
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    confidence = max(model.predict_proba(X)[0])

    return jsonify({
        "prediction": prediction,
        "confidence": float(confidence)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
