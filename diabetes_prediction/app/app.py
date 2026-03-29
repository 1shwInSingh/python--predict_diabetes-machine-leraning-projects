import streamlit as st
import joblib
import yaml

# page config
st.set_page_config(page_title="Diabetes Predictor", layout="centered")

# load config
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# load model
model = joblib.load(config["model_path"])

# title
st.title("🩺 Diabetes Prediction App")

st.info("This app predicts the likelihood of diabetes based on health indicators.")

st.write("### Enter health details:")

# ---- INPUTS ---- #

# binary inputs (0 or 1)
HighBP = st.selectbox("High Blood Pressure", [0, 1])
HighChol = st.selectbox("High Cholesterol", [0, 1])
CholCheck = st.selectbox("Cholesterol Check", [0, 1])
Smoker = st.selectbox("Smoker", [0, 1])
Stroke = st.selectbox("Stroke", [0, 1])
HeartDiseaseorAttack = st.selectbox("Heart Disease", [0, 1])
PhysActivity = st.selectbox("Physical Activity", [0, 1])
Fruits = st.selectbox("Fruits Intake", [0, 1])
Veggies = st.selectbox("Veggies Intake", [0, 1])
HvyAlcoholConsump = st.selectbox("Heavy Alcohol Consumption", [0, 1])
AnyHealthcare = st.selectbox("Healthcare Access", [0, 1])
NoDocbcCost = st.selectbox("No Doctor (Cost Issue)", [0, 1])
DiffWalk = st.selectbox("Difficulty Walking", [0, 1])
Sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])

# numeric inputs
BMI = st.number_input("BMI", min_value=0.0)
GenHlth = st.slider("General Health (1 = Best, 5 = Worst)", 1, 5)
MentHlth = st.number_input("Mental Health Days (0–30)", min_value=0)
PhysHlth = st.number_input("Physical Health Days (0–30)", min_value=0)
age_options = {
    "18-24": 1,
    "25-29": 2,
    "30-34": 3,
    "35-39": 4,
    "40-44": 5,
    "45-49": 6,
    "50-54": 7,
    "55-59": 8,
    "60-64": 9,
    "65-69": 10,
    "70-74": 11,
    "75-79": 12,
    "80+": 13
}

age_label = st.selectbox("Age Group", list(age_options.keys()))
Age = age_options[age_label]
Education = st.slider("Education Level (1–6)", 1, 6)
Income = st.slider("Income Level (1–8)", 1, 8)

# ---- PREDICTION ---- #

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