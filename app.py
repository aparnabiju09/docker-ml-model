from flask import Flask, request, jsonify
import joblib, numpy as np

app = Flask(__name__)
artifact = joblib.load("model.pkl")
model = artifact["model"]
target_names = artifact["target_names"]

@app.route("/")
def home():
    return "Dockerized ML Model API is running."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    pred_idx = int(model.predict(features)[0])
    return jsonify({"prediction": target_names[pred_idx], "index": pred_idx})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
