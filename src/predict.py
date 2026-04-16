import joblib
import numpy as np
import pandas as pd
import os

from src.data_preprocessing import FEATURE_COLUMNS

# Fallback threshold if threshold.pkl is not found
DEFAULT_THRESHOLD = 0.40

# Feature order that must match training (21 raw BRFSS + 8 engineered = 29)
RAW_FEATURE_ORDER = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]


def load_model(model_path, scaler_path=None):
    """
    Load the trained pipeline.
    scaler_path is accepted for backward compatibility but ignored —
    the scaler is baked into the pipeline.
    Returns (pipeline, threshold).
    """
    pipeline = joblib.load(model_path)

    # Try to load the auto-derived threshold saved alongside the model
    base_dir = os.path.dirname(model_path)
    threshold_path = os.path.join(base_dir, "threshold.pkl")
    if os.path.exists(threshold_path):
        threshold = joblib.load(threshold_path)
    else:
        threshold = DEFAULT_THRESHOLD

    return pipeline, threshold


def build_engineered_features(raw: dict) -> list:
    """
    Given a dict of the 21 raw BRFSS fields, compute engineered features
    and return the full ordered feature vector (29 values).
    """
    bmi           = float(raw.get("BMI", 25))
    age           = float(raw.get("Age", 5))
    high_bp       = float(raw.get("HighBP", 0))
    high_chol     = float(raw.get("HighChol", 0))
    gen_hlth      = float(raw.get("GenHlth", 3))
    phys_activity = float(raw.get("PhysActivity", 0))
    heart_disease = float(raw.get("HeartDiseaseorAttack", 0))
    stroke        = float(raw.get("Stroke", 0))
    smoker        = float(raw.get("Smoker", 0))
    diff_walk     = float(raw.get("DiffWalk", 0))

    # Original 4
    bmi_age          = bmi * age
    bp_chol          = high_bp * high_chol
    health_activity  = gen_hlth * (1 - phys_activity)
    cardio_risk      = high_bp + high_chol + heart_disease + stroke

    # New 4
    if   bmi < 18.5: bmi_category = 0.0
    elif bmi < 25.0: bmi_category = 1.0
    elif bmi < 30.0: bmi_category = 2.0
    else:            bmi_category = 3.0

    age_band      = min(int(age / 3), 4)
    multimorbidity = high_bp + high_chol + heart_disease + stroke + smoker + diff_walk
    framingham    = (
        high_bp * 1.5 + high_chol * 1.2 + smoker * 1.3
        + heart_disease * 2.0 + max(bmi - 25, 0) * 0.1 + age * 0.2
    )

    raw_values = [float(raw.get(k, 0)) for k in RAW_FEATURE_ORDER]
    engineered  = [bmi_age, bp_chol, health_activity, cardio_risk,
                   bmi_category, float(age_band), multimorbidity, framingham]
    return raw_values + engineered


def predict(model_or_pipeline, scaler_or_threshold, input_data):
    """
    Unified predict function.
    - model_or_pipeline: the sklearn Pipeline (or legacy model)
    - scaler_or_threshold: threshold float (new) or scaler (legacy)
    - input_data: list of feature values (already in full feature order)
    Returns dict with prediction, probabilities.
    """
    # Detect new vs legacy call signature
    if isinstance(scaler_or_threshold, float):
        pipeline  = model_or_pipeline
        threshold = scaler_or_threshold
    else:
        # Legacy path — scaler passed separately
        pipeline  = model_or_pipeline
        threshold = DEFAULT_THRESHOLD

    # Wrap in DataFrame so pipeline receives named columns (no sklearn warning)
    input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
    probabilities = pipeline.predict_proba(input_df)[0]
    prediction = 1 if probabilities[1] >= threshold else 0

    return {
        "prediction": int(prediction),
        "probability_no_diabetes": round(float(probabilities[0]) * 100, 1),
        "probability_diabetes":    round(float(probabilities[1]) * 100, 1),
        "threshold_used":          round(threshold, 2),
    }