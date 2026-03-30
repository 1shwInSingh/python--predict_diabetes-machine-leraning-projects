# 🩺 Diabetes Prediction Web App

A Machine Learning-based web application that predicts the risk of diabetes using health-related indicators.

---

## 🚀 Overview

This project is an end-to-end Machine Learning application that:

* Processes real-world health data
* Trains a predictive model
* Provides an interactive web interface

Users can input their health details and instantly get a diabetes risk prediction.

---

## 🧠 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Streamlit

---

## 📂 Project Structure

```
diabetes_prediction/
│
├── app/
│   └── app.py
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── model.pkl
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── predict.py
│   └── utils.py
├── main.py
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

* Diabetes Health Indicators Dataset
* Includes features like:

  * Blood Pressure
  * Cholesterol
  * BMI
  * Lifestyle habits
  * Age group

---

## ⚙️ Setup Instructions

### 1. Clone Repository

```
git clone https://github.com/Kanishk-32/python--predict_diabetes-machine-leraning-projects.git
cd diabetes_prediction
```

---

### 2. Create Virtual Environment

```
python -m venv venv
venv\Scripts\activate
```

---

### 3. Install Dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Run Project

### Run ML Pipeline

```
python main.py
```

### Run Web App

```
streamlit run app/app.py
```

---

## 📸 App Preview

![App Screenshot 1](https://github.com/user-attachments/assets/497d1527-4061-4f0c-852b-bca4031b9f7b)

![App Screenshot 2](https://github.com/user-attachments/assets/368931d2-c23a-473c-b7a1-4b1ed434206e)

---

## 🎯 Model Details

* Algorithm: Logistic Regression
* Type: Binary Classification

**Output:**

* `0` → Low Risk of Diabetes
* `1` → High Risk of Diabetes

---

## 🌐 Features

* Clean and user-friendly UI
* Real-time predictions
* Checkbox-based inputs (Yes/No)
* Dropdowns and sliders for better UX

---

## 📈 Inputs Used

* Health conditions
* BMI
* General health
* Age group
* Gender

---

## 🔮 Future Improvements

* Improve model accuracy (Random Forest, XGBoost)
* Add data visualization
* Deploy online
* Enhance UI design

---

## 🤝 Contributing

Contributions are welcome. Feel free to fork and improve the project.

---

## 👨‍💻 Author

**Kanishk**

---

## ⭐ Support

If you found this project useful, give it a ⭐ on GitHub.
