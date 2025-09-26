from flask import Flask, request, jsonify
import joblib

# Load model & vectorizer
model = joblib.load("hoax_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    text = data.get("text")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Transform text
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]

    return jsonify({"prediction": str(prediction)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
