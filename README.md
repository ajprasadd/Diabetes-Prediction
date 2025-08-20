# 🩺 Diabetes Prediction using Clinical Features (KNN & Random Forest)

This project is a **Machine Learning–based web application** built with **Streamlit** that predicts whether a person is likely to have diabetes based on various **clinical features**.  

The app uses two algorithms:
- **K-Nearest Neighbors (KNN)**
- **Random Forest Classifier**

It allows users to input their clinical details and receive a prediction instantly.

---

## ⚠️ Disclaimer
This project is created for **educational and research purposes only**.  
It should **NOT** be used for real medical diagnosis. Always consult a doctor for medical advice.

---

## 🚀 Features
- Input clinical features via Streamlit UI:
  - Pregnancies  
  - Glucose  
  - Blood Pressure  
  - Skin Thickness  
  - Insulin  
  - BMI  
  - Diabetes Pedigree Function  
  - Age  
- Data preprocessing with handling missing values and feature scaling.  
- Train and evaluate models (**KNN & Random Forest**).  
- Save models (`.pkl` files) for deployment.  
- Interactive web app for predictions.  

---

## 📂 Project Structure
/diabetes-prediction
│── diabetes.csv # Dataset (PIMA Indians Diabetes Dataset)
│── knn_model.pkl # Trained KNN model
│── rf_model.pkl # Trained Random Forest model
│── scaler.pkl # Scaler used for preprocessing
│── app.py (or knn.py) # Streamlit app
│── requirements.txt # Dependencies
│── README.md # Documentation



📦 Requirements
requirements.txt should include:

streamlit
scikit-learn
pandas
numpy
joblib


📊 Model Training (Google Colab / Jupyter)

To retrain models:
Preprocess diabetes.csv (replace invalid zeros, handle missing values).
Split dataset into train/test.
Scale features using StandardScaler.Train both KNN and Random Forest.
Save models with joblib.

🛠️ How the App Works
User enters clinical details in Streamlit UI.
Input is scaled using the saved scaler.pkl.
Model (KNN or Random Forest) predicts outcome.Result is displayed:
✅ "Person is NOT likely to have Diabetes"
⚠️ "Person is likely to have Diabetes"


🎯 Learning Outcomes
Practical use of clinical datasets for disease prediction.
Preprocessing (handling missing values, scaling).
Training and comparing ML models (KNN vs Random Forest).
Building a simple ML web app with Streamlit.

To run this project use this command after opening and installing all requirements 
python -m streamlit app.run
