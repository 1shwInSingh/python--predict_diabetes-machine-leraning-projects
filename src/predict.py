import joblib
import numpy as np

# Custom threshold for better diabetes recall
RISK_THRESHOLD = 0.35

def load_model(model_path, scaler_path="models/scaler.pkl"):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict(model, scaler, input_data):
    """Predict diabetes risk with probability score using custom threshold."""
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    
    probability = model.predict_proba(input_scaled)[0]
    # Use custom threshold for better recall on diabetes cases
    prediction = 1 if probability[1] >= RISK_THRESHOLD else 0
    
    return {
        "prediction": int(prediction),
        "probability_no_diabetes": round(float(probability[0]) * 100, 1),
        "probability_diabetes": round(float(probability[1]) * 100, 1)
    }