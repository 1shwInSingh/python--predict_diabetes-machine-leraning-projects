import streamlit as st
import joblib
import yaml

st.set_page_config(page_title="Diabetes Predictor", layout="centered")

with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

model = joblib.load(config["model_path"])

st.title("🩺 Diabetes Prediction App")
st.info("This app predicts the likelihood of diabetes based on health indicators.")
st.write("### Enter health details:")

HighBP = 1 if st.checkbox("High Blood Pressure") else 0
HighChol = 1 if st.checkbox("High Cholesterol") else 0
CholCheck = 1 if st.checkbox("Had Cholesterol Check") else 0
Smoker = 1 if st.checkbox("Smoker") else 0
Stroke = 1 if st.checkbox("Had Stroke") else 0
HeartDiseaseorAttack = 1 if st.checkbox("Heart Disease / Attack") else 0
PhysActivity = 1 if st.checkbox("Physically Active") else 0
Fruits = 1 if st.checkbox("Eats Fruits Regularly") else 0
Veggies = 1 if st.checkbox("Eats Vegetables Regularly") else 0
HvyAlcoholConsump = 1 if st.checkbox("Heavy Alcohol Consumption") else 0
AnyHealthcare = 1 if st.checkbox("Has Healthcare Access") else 0
NoDocbcCost = 1 if st.checkbox("Couldn't See Doctor Due to Cost") else 0
DiffWalk = 1 if st.checkbox("Difficulty Walking") else 0

gender = st.selectbox("Gender", ["Female", "Male"])
Sex = 1 if gender == "Male" else 0

BMI = st.number_input("BMI", min_value=0.0)
GenHlth = st.slider("General Health (1 = Best, 5 = Worst)", 1, 5)

MentHlth = 0
PhysHlth = 0
Education = 3
Income = 5

age_options = {
    "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4,
    "40-44": 5, "45-49": 6, "50-54": 7, "55-59": 8,
    "60-64": 9, "65-69": 10, "70-74": 11, "75-79": 12, "80+": 13
}

age_label = st.selectbox("Age Group", list(age_options.keys()))
Age = age_options[age_label]

if st.button("Predict"):
    input_data = [
        HighBP, HighChol, CholCheck, BMI, Smoker, Stroke,
        HeartDiseaseorAttack, PhysActivity, Fruits, Veggies,
        HvyAlcoholConsump, AnyHealthcare, NoDocbcCost,
        GenHlth, MentHlth, PhysHlth, DiffWalk,
        Sex, Age, Education, Income
    ]

    prediction = model.predict([input_data])[0]

    st.write("### Result:")

    if prediction == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")