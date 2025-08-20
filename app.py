# import streamlit as st 
# import pandas as pd 
# import joblib
# import numpy as np 


# #load the saved model and scaler 
# knn_model = joblib.load('knn_model.pkl')
# scaler = joblib.load('scaler.pkl')

# #title of app 
# st.title("Diabetes prediction App")

# #input fields for user data
# st.header("enter patient details")

# #define input fields
# Pregnencies = st.number_input("Pregnencies",min_value=0,max_value=20,value=0)

# Glucose = st.number_input("Glucose",min_value=0,max_value=200,value=200)

# Blood_Pressure = st.number_input("Blood Pressure",min_value=0,max_value=200,value=70)

# Skin_Thinckness = st.number_input("Skin Thickness",min_value=0,max_value=100,value=20)

# BMI = st.number_input("BMI",min_value=10,max_value=50,value=30)

# Insulin = st.number_input("Insulin",min_value=0,max_value=1000,value=100)

# Diabetes_Pedigree = st.number_input("Diabetes Pedigree function", min_value=0.0,max_value=2.5,value=0.50)

# Age = st.number_input ("Age",min_value=18,max_value=100,value=30)


# #create list of inputs

# user_inputs = np.array([[Pregnencies,Glucose,Blood_Pressure,Skin_Thinckness,Insulin,BMI,Diabetes_Pedigree,Age]])

# #scaling the user input using scaler 

# user_input_scaled = scaler.transform(user_inputs)

# #make prediction using trained knn model

# prediction = knn_model.predict(user_input_scaled)

# #output the result 

# if st.button("predict"):
#     if prediction == 1:
#         st.success ("The preson have diabetes");
#     else :
#         st.sucess("the person doesnot have diabetes");









import streamlit as st 
import pandas as pd 
import joblib
import numpy as np 

# Load the saved model and scaler 
knn_model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Title of app 
st.title("🩺 Diabetes Prediction App")

# Input fields for user data
st.header("Enter Patient Details")

# Define input fields with proper step values
Pregnencies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)

Glucose = st.number_input("Glucose", min_value=0, max_value=200, value=120, step=1)

Blood_Pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70, step=1)

Skin_Thinckness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20, step=1)

Insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=100, step=1)

BMI = st.number_input("BMI", min_value=10.0, max_value=50.0, value=30.0, step=0.1, format="%.1f")

Diabetes_Pedigree = st.number_input(
    "Diabetes Pedigree Function", 
    min_value=0.0, max_value=2.5, value=0.50, step=0.01, format="%.2f"
)

Age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)

# Create list of inputs
user_inputs = np.array([[Pregnencies, Glucose, Blood_Pressure, Skin_Thinckness, Insulin, BMI, Diabetes_Pedigree, Age]])

# Scale the user input using scaler 
user_input_scaled = scaler.transform(user_inputs)

# Make prediction using trained knn model
prediction = knn_model.predict(user_input_scaled)

# Output the result 
if st.button("Predict"):
    if prediction[0] == 1:
        st.success("✅ The person is likely to have Diabetes")
    else:
        st.success("✅ The person is NOT likely to have Diabetes")
