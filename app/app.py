from flask import Flask, render_template, request, jsonify
import joblib
import yaml
import numpy as np
import os

app = Flask(__name__)

# Custom threshold for better diabetes recall (catches 53% vs 18% of diabetic cases)
RISK_THRESHOLD = 0.35

# Load config
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Load model and scaler
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config["model_path"])
scaler_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Extract features in correct order
        features = [
            float(data.get("HighBP", 0)),
            float(data.get("HighChol", 0)),
            float(data.get("CholCheck", 0)),
            float(data.get("BMI", 25)),
            float(data.get("Smoker", 0)),
            float(data.get("Stroke", 0)),
            float(data.get("HeartDiseaseorAttack", 0)),
            float(data.get("PhysActivity", 0)),
            float(data.get("Fruits", 0)),
            float(data.get("Veggies", 0)),
            float(data.get("HvyAlcoholConsump", 0)),
            float(data.get("AnyHealthcare", 0)),
            float(data.get("NoDocbcCost", 0)),
            float(data.get("GenHlth", 3)),
            float(data.get("MentHlth", 0)),
            float(data.get("PhysHlth", 0)),
            float(data.get("DiffWalk", 0)),
            float(data.get("Sex", 0)),
            float(data.get("Age", 5)),
            float(data.get("Education", 4)),
            float(data.get("Income", 5)),
        ]
        
        # Add engineered features (must match training)
        bmi = features[3]
        age = features[18]
        high_bp = features[0]
        high_chol = features[1]
        gen_hlth = features[13]
        phys_activity = features[7]
        heart_disease = features[6]
        stroke = features[5]
        
        bmi_age = bmi * age
        bp_chol = high_bp * high_chol
        health_activity = gen_hlth * (1 - phys_activity)
        cardio_risk = high_bp + high_chol + heart_disease + stroke
        
        features.extend([bmi_age, bp_chol, health_activity, cardio_risk])
        
        # Scale and predict
        input_array = np.array(features).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        
        probabilities = model.predict_proba(input_scaled)[0]
        # Use custom threshold for better recall on diabetes cases
        prediction = 1 if probabilities[1] >= RISK_THRESHOLD else 0
        
        return jsonify({
            "prediction": prediction,
            "probability_no_diabetes": round(float(probabilities[0]) * 100, 1),
            "probability_diabetes": round(float(probabilities[1]) * 100, 1),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)