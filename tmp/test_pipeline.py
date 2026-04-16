# -*- coding: utf-8 -*-
"""End-to-end validation — uses the same code paths as the Flask app."""
import sys
sys.path.insert(0, '.')

import joblib
import pandas as pd
from src.predict import build_engineered_features
from src.data_preprocessing import FEATURE_COLUMNS

pipeline  = joblib.load('models/pipeline.pkl')
threshold = joblib.load('models/threshold.pkl')
print(f'OK  Pipeline type  : {type(pipeline).__name__}')
print(f'OK  Threshold      : {threshold:.2f}')

# --- High-risk patient ---
high_risk = {
    'HighBP': 1, 'HighChol': 1, 'CholCheck': 1, 'BMI': 35,
    'Smoker': 1, 'Stroke': 0, 'HeartDiseaseorAttack': 1,
    'PhysActivity': 0, 'Fruits': 0, 'Veggies': 0,
    'HvyAlcoholConsump': 0, 'AnyHealthcare': 1, 'NoDocbcCost': 0,
    'GenHlth': 5, 'MentHlth': 10, 'PhysHlth': 15,
    'DiffWalk': 1, 'Sex': 1, 'Age': 11, 'Education': 3, 'Income': 2
}
features  = build_engineered_features(high_risk)
assert len(features) == 29, f"Expected 29 features, got {len(features)}"
input_df  = pd.DataFrame([features], columns=FEATURE_COLUMNS)
proba     = pipeline.predict_proba(input_df)[0]
pred      = int(proba[1] >= threshold)
print(f'\nHigh-risk patient:')
print(f'  Prediction  : {"DIABETIC" if pred==1 else "NO DIABETES"}')
print(f'  P(diabetes) : {proba[1]*100:.1f}%')
assert pred == 1, "FAIL: High-risk patient should be DIABETIC"
print(f'  PASS')

# --- Low-risk patient ---
low_risk = {
    'HighBP': 0, 'HighChol': 0, 'CholCheck': 1, 'BMI': 22,
    'Smoker': 0, 'Stroke': 0, 'HeartDiseaseorAttack': 0,
    'PhysActivity': 1, 'Fruits': 1, 'Veggies': 1,
    'HvyAlcoholConsump': 0, 'AnyHealthcare': 1, 'NoDocbcCost': 0,
    'GenHlth': 1, 'MentHlth': 0, 'PhysHlth': 0,
    'DiffWalk': 0, 'Sex': 0, 'Age': 4, 'Education': 6, 'Income': 8
}
features2 = build_engineered_features(low_risk)
input_df2 = pd.DataFrame([features2], columns=FEATURE_COLUMNS)
proba2    = pipeline.predict_proba(input_df2)[0]
pred2     = int(proba2[1] >= threshold)
print(f'\nLow-risk patient:')
print(f'  Prediction  : {"DIABETIC" if pred2==1 else "NO DIABETES"}')
print(f'  P(diabetes) : {proba2[1]*100:.1f}%')
assert pred2 == 0, "FAIL: Low-risk patient should be NO DIABETES"
print(f'  PASS')

# --- Borderline patient ---
border = {
    'HighBP': 1, 'HighChol': 0, 'CholCheck': 1, 'BMI': 28,
    'Smoker': 0, 'Stroke': 0, 'HeartDiseaseorAttack': 0,
    'PhysActivity': 1, 'Fruits': 1, 'Veggies': 1,
    'HvyAlcoholConsump': 0, 'AnyHealthcare': 1, 'NoDocbcCost': 0,
    'GenHlth': 3, 'MentHlth': 2, 'PhysHlth': 3,
    'DiffWalk': 0, 'Sex': 1, 'Age': 7, 'Education': 5, 'Income': 5
}
features3 = build_engineered_features(border)
input_df3 = pd.DataFrame([features3], columns=FEATURE_COLUMNS)
proba3    = pipeline.predict_proba(input_df3)[0]
pred3     = int(proba3[1] >= threshold)
print(f'\nBorderline patient:')
print(f'  Prediction  : {"DIABETIC" if pred3==1 else "NO DIABETES"}')
print(f'  P(diabetes) : {proba3[1]*100:.1f}%')
print(f'  (no assertion - borderline can go either way)')

print('\n=== All critical tests PASSED ===')
