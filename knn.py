import streamlit as st 
import pandas as pd 
import joblib
import numpy as np 


#load the saved model and scaler 
knn_model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

#title of app 
st.title("Diabetes prediction App")

#input fields for user data
st.header("enter patient details")

#define input fields
Pregnencies = st.number_input("Pregnencies",min_value=0,max_value=20,value=0)

Glucose = st.number_input("Glucose",min_value=0,max_value=200,value=200)

Blood_Pressure = st.number_input("Blood Pressure",min_value=0,max_value=200,value=70)

Skin_Thinckness = st.number_input("Skin Thickness",min_value=0,max_value=100,value=20)

BMI = st.number_input("BMI",min_value=10,max_value=50,value=30)

Insulin = st.number_input("Insulin",min_value=0,max_value=1000,value=100)

Diabetes_Pedigree = st.number_input("Diabetes Pedigree function" min_value=0.0,max_value=2.5,value=0.50)

Age = st.number_input ("Age",min_value=18,max_value=100,value=30)


#create list of inputs

user_inputs = np.array([[Pregnencies,Glucose,Blood_Pressure,Skin_Thickness,Insulin,BMI,Diabetes_pedigree,Age]])

#scaling the user input using scaler 

user_input_scaled = scaler.transform(user_input)

#make prediction using trained knn model

prediction = knn_model.predict(user_input_scaled)

#output the result 

if st.button("predict"):
    if prediction == 1:
        st.success ("The preson have diabetes");
    else :
        st.sucess("the person doesnot have diabetes");