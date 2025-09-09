from flask import Flask, request, jsonify
import joblib
import re
import pandas as pd

app = Flask(__name__)

# Load models and preprocessors
vectorizer = joblib.load("models/countVectorizer.pkl")
scaler = joblib.load("models/scaler.pkl")
model_dt = joblib.load("models/model_dt.pkl")
model_rf = joblib.load("models/model_rf.pkl")
model_xgb = joblib.load("models/model_xgb.pkl")

label_map = {0: "Negative", 1: "Positive"}

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text).strip()
    return text

def predict_text(text, model_name):
    text = preprocess(text)
    print(f"[Debug] Preprocessed text: {text}")

    X = vectorizer.transform([text])
    X_scaled = scaler.transform(X.toarray())

    if model_name == "dt":
        pred = model_dt.predict(X_scaled)[0]
    elif model_name == "rf":
        pred = model_rf.predict(X_scaled)[0]
    elif model_name == "xgb":
        pred = model_xgb.predict(X_scaled)[0]
    else:
        raise ValueError("Invalid model choice.")

    print(f"[Debug] Prediction: {pred}")
    return label_map[pred]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    model_name = data.get("model", "dt")
    try:
        prediction = predict_text(model_name, text)
        return jsonify({"text": text, "prediction": prediction})
    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/bulk_predict", methods=["POST"])
def bulk_predict():
    try:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        chunk_size = 10  # process 10 rows at a time
        results = []
        for chunk in pd.read_csv(file, chunksize=chunk_size):
            if 'text' not in chunk.columns:
                return jsonify({"error": "'text' column missing in CSV"}), 400
            for t in chunk['text']:
                pred = predict_text(t, request.form.get('model', 'dt'))
                results.append({"text": t, "prediction": pred})
        return jsonify(results)
    except Exception as e:
        print(f"Error in bulk_predict: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)
