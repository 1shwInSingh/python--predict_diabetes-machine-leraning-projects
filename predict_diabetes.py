import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm

def setup_model():
    # 1. Load Data (assuming diabetes.csv is in the folder)
    try:
        df = pd.read_csv('diabetes.csv')
        # Handle cases where CSV might not have headers
        if 'Outcome' not in df.columns:
            df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                          'Insulin', 'BMI', 'DPF', 'Age', 'Outcome']
    except FileNotFoundError:
        print("Error: 'diabetes.csv' not found in this directory.")
        return None, None

    # 2. Prepare Data
    X = df.drop(columns='Outcome')
    Y = df['Outcome']

    # 3. Standardize Data
    scaler = StandardScaler()
    scaler.fit(X)
    X_standardized = scaler.transform(X)

    # 4. Train Model
    X_train, X_test, Y_train, Y_test = train_test_split(X_standardized, Y, test_size=0.2, stratify=Y, random_state=2)
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, Y_train)
    
    return classifier, scaler

def get_user_input():
    print("\n--- Enter Patient Details ---")
    try:
        preg = float(input("Number of Pregnancies: "))
        gluc = float(input("Glucose Level: "))
        bp   = float(input("Blood Pressure: "))
        skin = float(input("Skin Thickness: "))
        ins  = float(input("Insulin Level: "))
        bmi  = float(input("BMI (e.g., 25.5): "))
        dpf  = float(input("Diabetes Pedigree Function (e.g., 0.47): "))
        age  = float(input("Age: "))
        return (preg, gluc, bp, skin, ins, bmi, dpf, age)
    except ValueError:
        print("Invalid input. Please enter numbers only.")
        return None

def main():
    model, scaler = setup_model()
    if model is None: return

    while True:
        data = get_user_input()
        if data:
            # Reshape and Scale
            input_array = np.asarray(data).reshape(1, -1)
            std_data = scaler.transform(input_array)
            
            # Predict
            prediction = model.predict(std_data)
            
            print("\n" + "="*25)
            if prediction[0] == 0:
                print("RESULT: NOT DIABETIC")
            else:
                print("RESULT: DIABETIC")
            print("="*25)
        
        cont = input("\nCheck another patient? (y/n): ").lower()
        if cont != 'y':
            break

if __name__ == "__main__":
    main()
