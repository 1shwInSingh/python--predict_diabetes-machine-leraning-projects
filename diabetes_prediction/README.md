# 🩺 Diabetes Prediction App

A Machine Learning web application that predicts the likelihood of diabetes based on health indicators.

---

## 🚀 Project Overview

This project uses a **Logistic Regression model** trained on a real-world health dataset to predict whether a person is at risk of diabetes.

The app is built using:

* 🧠 Machine Learning (Scikit-learn)
* 📊 Data Processing (Pandas, NumPy)
* 🌐 Web App (Streamlit)

---

## 📂 Project Structure

```
diabetes_prediction/
│
├── app/
│   └── app.py              # Streamlit web app
│
├── config/
│   └── config.yaml        # Configuration file
│
├── data/
│   ├── raw/               # Original dataset
│   └── processed/         # Cleaned data
│
├── models/
│   └── model.pkl          # Trained ML model
│
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── predict.py
│   └── utils.py
│
├── main.py                # Main pipeline script
├── requirements.txt       # Dependencies
└── README.md
```

---

## 📊 Dataset

* Source: Diabetes Health Indicators Dataset (Kaggle)
* Contains health-related features like:

  * BMI
  * Age
  * Blood Pressure
  * Cholesterol
  * Lifestyle habits

---

## ⚙️ Installation

### 1. Clone the repository

```
git clone https://github.com/your-username/your-repo-name.git
cd diabetes_prediction
```

### 2. Create virtual environment

```
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Run the Project

### Run ML pipeline

```
python main.py
```

### Run Web App

```
streamlit run app/app.py
```

---

## 🧠 Model Details

* Algorithm: Logistic Regression
* Problem Type: Binary Classification
* Target:

  * 0 → No Diabetes
  * 1 → Diabetes

---

## 📈 Features Used

* High Blood Pressure
* High Cholesterol
* BMI
* Smoking Status
* Physical Activity
* General Health
* Age Group
* Education & Income

---

## 🎯 Results

* Model Accuracy: ~80% (depends on dataset split)
* Fast and lightweight model
<img width="1907" height="871" alt="image" src="https://github.com/user-attachments/assets/f3b50bea-b0f5-4562-bc58-9e71aba795a9" />
<img width="1919" height="860" alt="image" src="https://github.com/user-attachments/assets/65739a3b-f80c-45f9-99df-d0f1c80da173" />

---

## 🌐 App Preview

Users can:

* Enter health details
* Click **Predict**
* Get instant result

---

## 🔥 Future Improvements

* Improve model accuracy (Random Forest, XGBoost)
* Add charts & visualizations
* Deploy online (Streamlit Cloud / Render)
* Add user authentication

---

## 🤝 Contributing

Feel free to fork this repository and improve it!

---

## 📜 License

This project is open-source and free to use.

---

## 👨‍💻 Author

**Kanishk**

---

⭐ If you like this project, don’t forget to star the repo!
