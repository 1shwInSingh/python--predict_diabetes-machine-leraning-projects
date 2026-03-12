# predict_diabetes.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm

# Load and train once
df = pd.read_csv('diabetes.csv')

if 'Outcome' not in df.columns:
    df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                  'Insulin', 'BMI', 'DPF', 'Age', 'Outcome']

X = df.drop(columns='Outcome')
Y = df['Outcome']

scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(
    X_standardized, Y, test_size=0.2, stratify=Y, random_state=2
)

model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)


def predict_diabetes(data):
    input_array = np.asarray(data).reshape(1, -1)
    std_data = scaler.transform(input_array)
    prediction = model.predict(std_data)
    return int(prediction[0])