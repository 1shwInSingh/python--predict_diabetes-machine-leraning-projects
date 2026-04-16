import joblib
import numpy as np

def load_model(model_path, scaler_path="models/scaler.pkl"):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict(model, scaler, input_data):
    """Predict diabetes risk with probability score."""
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    return {
        "prediction": int(prediction),
        "probability_no_diabetes": round(float(probability[0]) * 100, 1),
        "probability_diabetes": round(float(probability[1]) * 100, 1)
    }