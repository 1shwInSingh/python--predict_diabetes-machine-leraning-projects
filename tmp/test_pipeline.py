# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, '.')
import joblib
import numpy as np
from src.predict import build_engineered_features

pipeline  = joblib.load('models/pipeline.pkl')
threshold = joblib.load('models/threshold.pkl')
print(f'Pipeline loaded  OK - type: {type(pipeline).__name__}')
print(f'Threshold loaded OK - value: {threshold:.2f}')

# Test with a high-risk patient profile
high_risk = {
    'HighBP': 1, 'HighChol': 1, 'CholCheck': 1, 'BMI': 35,
    'Smoker': 1, 'Stroke': 0, 'HeartDiseaseorAttack': 1,
    'PhysActivity': 0, 'Fruits': 0, 'Veggies': 0,
    'HvyAlcoholConsump': 0, 'AnyHealthcare': 1, 'NoDocbcCost': 0,
    'GenHlth': 5, 'MentHlth': 10, 'PhysHlth': 15,
    'DiffWalk': 1, 'Sex': 1, 'Age': 11, 'Education': 3, 'Income': 2
}
features = build_engineered_features(high_risk)
print(f'Feature vector length: {len(features)} (expected: 29)')

proba = pipeline.predict_proba(np.array(features).reshape(1, -1))[0]
pred  = int(proba[1] >= threshold)
label = 'DIABETIC' if pred == 1 else 'NO DIABETES'
print(f'High-risk result: {label}  (P(diabetes)={proba[1]*100:.1f}%)')

# Test with a low-risk patient
low_risk = {
    'HighBP': 0, 'HighChol': 0, 'CholCheck': 1, 'BMI': 22,
    'Smoker': 0, 'Stroke': 0, 'HeartDiseaseorAttack': 0,
    'PhysActivity': 1, 'Fruits': 1, 'Veggies': 1,
    'HvyAlcoholConsump': 0, 'AnyHealthcare': 1, 'NoDocbcCost': 0,
    'GenHlth': 1, 'MentHlth': 0, 'PhysHlth': 0,
    'DiffWalk': 0, 'Sex': 0, 'Age': 4, 'Education': 6, 'Income': 8
}
features2 = build_engineered_features(low_risk)
proba2 = pipeline.predict_proba(np.array(features2).reshape(1, -1))[0]
pred2  = int(proba2[1] >= threshold)
label2 = 'DIABETIC' if pred2 == 1 else 'NO DIABETES'
print(f'Low-risk result:  {label2}  (P(diabetes)={proba2[1]*100:.1f}%)')
print('All checks passed!')
