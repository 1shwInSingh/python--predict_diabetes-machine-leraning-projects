import pandas as pd
import numpy as np

# Continuous columns that use 0 as a "missing/unknown" placeholder in BRFSS data
ZERO_AS_MISSING = ["BMI", "MentHlth", "PhysHlth"]

# Column order expected by the pipeline (21 raw + 8 engineered = 29 total features)
FEATURE_COLUMNS = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
    # Engineered features
    "BMI_Age", "BP_Chol", "Health_Activity", "CardioRisk",
    "BMI_Category", "AgeBand", "Multimorbidity", "FraminghamRisk",
]


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, fix BRFSS zero-as-missing, cap BMI outliers."""
    df = df.drop_duplicates()
    df = df.dropna()

    # --- Fix zero-as-missing in continuous fields ---
    for col in ZERO_AS_MISSING:
        if col in df.columns:
            median_val = df[col][df[col] > 0].median()
            df[col] = df[col].replace(0, median_val)

    # Cap BMI outliers at 1st and 99th percentile
    bmi_lower = df["BMI"].quantile(0.01)
    bmi_upper = df["BMI"].quantile(0.99)
    df["BMI"] = df["BMI"].clip(bmi_lower, bmi_upper)

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add clinically-motivated interaction and derived features."""
    df = df.copy()

    bmi = df["BMI"]
    age = df["Age"]
    high_bp = df["HighBP"]
    high_chol = df["HighChol"]
    gen_hlth = df["GenHlth"]
    phys_activity = df["PhysActivity"]
    heart_disease = df["HeartDiseaseorAttack"]
    stroke = df["Stroke"]
    smoker = df["Smoker"]
    diff_walk = df["DiffWalk"]

    # --- Original 4 features (kept for backward compat) ---
    df["BMI_Age"] = bmi * age
    df["BP_Chol"] = high_bp * high_chol
    df["Health_Activity"] = gen_hlth * (1 - phys_activity)
    df["CardioRisk"] = high_bp + high_chol + heart_disease + stroke

    # --- New features ---
    # WHO BMI risk category: 0=underweight, 1=normal, 2=overweight, 3=obese
    df["BMI_Category"] = pd.cut(
        bmi,
        bins=[0, 18.5, 25, 30, 100],
        labels=[0, 1, 2, 3],
    ).astype(float)

    # Age decade band (Age in BRFSS is 1–13 scale; divide gives coarser grouping)
    df["AgeBand"] = (age / 3).astype(int).clip(0, 4).astype(float)

    # Multi-morbidity count: number of co-existing conditions
    df["Multimorbidity"] = (
        high_bp + high_chol + heart_disease + stroke + smoker + diff_walk
    ).astype(float)

    # Simplified Framingham-style risk proxy
    df["FraminghamRisk"] = (
        high_bp * 1.5 + high_chol * 1.2 + smoker * 1.3
        + heart_disease * 2.0 + (bmi - 25).clip(0) * 0.1
        + age * 0.2
    )

    return df