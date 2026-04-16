from flask import Flask, render_template, request, jsonify
import joblib
import yaml
import numpy as np
import os

from src.predict import build_engineered_features

app = Flask(__name__, template_folder="templates")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(BASE_DIR, "config", "config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

model_path     = os.path.join(BASE_DIR, config.get("model_path", "models/model.pkl"))
threshold_path = os.path.join(BASE_DIR, "models", "threshold.pkl")

# ── Load pipeline and threshold ───────────────────────────────────────────────
pipeline  = joblib.load(model_path)

if os.path.exists(threshold_path):
    RISK_THRESHOLD = float(joblib.load(threshold_path))
else:
    RISK_THRESHOLD = 0.40   # sensible fallback

print(f"[app] Model loaded from  : {model_path}")
print(f"[app] Risk threshold     : {RISK_THRESHOLD:.2f}")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/app")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "threshold": RISK_THRESHOLD})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Build full 29-feature vector (21 raw + 8 engineered) using shared helper
        features = build_engineered_features(data)

        input_array   = np.array(features).reshape(1, -1)
        probabilities = pipeline.predict_proba(input_array)[0]

        prediction = int(probabilities[1] >= RISK_THRESHOLD)

        return jsonify({
            "prediction":             prediction,
            "probability_no_diabetes": round(float(probabilities[0]) * 100, 1),
            "probability_diabetes":    round(float(probabilities[1]) * 100, 1),
            "threshold":               round(RISK_THRESHOLD, 2),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)