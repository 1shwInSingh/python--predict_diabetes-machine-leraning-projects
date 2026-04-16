# -*- coding: utf-8 -*-
"""Show recall/precision/F1 across thresholds to pick a good sensitivity level."""
import sys
sys.path.insert(0, '.')

import joblib, numpy as np, pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
from src.data_preprocessing import load_data, clean_data, add_features, FEATURE_COLUMNS

print("Loading data and pipeline...")
pipeline  = joblib.load('models/pipeline.pkl')
threshold = float(joblib.load('models/threshold.pkl'))

df = load_data('data/raw/diabetes.csv')
df = clean_data(df)
df = add_features(df)
df['Diabetes_binary'] = (df['Diabetes_012'] > 0).astype(int)
y = df['Diabetes_binary']
X = df.drop(['Diabetes_012', 'Diabetes_binary'], axis=1)

from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Getting probabilities...")
probas = pipeline.predict_proba(X_test)[:, 1]

print(f"\n{'Thresh':>6}  {'Recall':>7}  {'Prec':>7}  {'F1':>7}  {'FN (missed)':>12}  {'FP (false alarm)':>16}")
print("-" * 65)
for t in [0.10, 0.13, 0.15, 0.17, 0.18, 0.20, 0.23, 0.25, 0.30]:
    preds  = (probas >= t).astype(int)
    rec    = recall_score(y_test, preds)
    prec   = precision_score(y_test, preds, zero_division=0)
    f1     = f1_score(y_test, preds, zero_division=0)
    cm     = confusion_matrix(y_test, preds)
    fn, fp = cm[1][0], cm[0][1]
    marker = "  <-- current" if abs(t - threshold) < 0.01 else ""
    print(f"  {t:.2f}   {rec:7.3f}   {prec:7.3f}   {f1:7.3f}   {fn:>8}        {fp:>10} {marker}")
