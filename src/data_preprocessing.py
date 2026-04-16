import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Drop rows with any nulls
    df = df.dropna()
    
    # Cap BMI outliers at 1st and 99th percentile
    bmi_lower = df['BMI'].quantile(0.01)
    bmi_upper = df['BMI'].quantile(0.99)
    df['BMI'] = df['BMI'].clip(bmi_lower, bmi_upper)
    
    return df

def add_features(df):
    """Add interaction features to improve model performance."""
    df = df.copy()
    
    # BMI × Age interaction — older + higher BMI = higher risk
    df['BMI_Age'] = df['BMI'] * df['Age']
    
    # HighBP × HighChol — both together is a strong risk signal
    df['BP_Chol'] = df['HighBP'] * df['HighChol']
    
    # General health × physical activity interaction
    df['Health_Activity'] = df['GenHlth'] * (1 - df['PhysActivity'])
    
    # Combined cardiovascular risk
    df['CardioRisk'] = df['HighBP'] + df['HighChol'] + df['HeartDiseaseorAttack'] + df['Stroke']
    
    return df