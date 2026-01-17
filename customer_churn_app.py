import streamlit as st
import numpy as  np
import pandas as pd
import joblib

model=joblib.load("churn_prediction_model.pkl")
scalar=joblib.load("scalar.pkl")

st.title("Customer Churn Preidction System")

st.write("pPlease enter the the details and press the predict button")

st.divider()

age=st.number_input("Enter the age",min_value=10,value=40)
tenure=st.number_input("Enter the tenure",min_value=1,value=10)
monthlycharges=st.number_input("Enter the monthly charges",min_value=1,value=10)
totalcharges = st.number_input("Enter total charges", min_value=0.0, value=1000.0)
gender=st.selectbox("Enter the gender",["Male","Female"])
contracttype=st.selectbox("Contract type",["Month-to-Month","One-Year","Two-Year"])
internetservice=st.selectbox("Internet Service Type",["Fiber Optic","DSL"])
techsupport=st.selectbox("Any Tech Support",["Yes","No"])

st.divider()

predictbutton=st.button("Predict")

if predictbutton:

    gender_encoded = 1 if gender == "Female" else 0
    contracttype_encoded = {"Month-to-Month": 0, "One-Year": 1, "Two-Year": 2}[contracttype]
    internetservice_encoded = {"DSL": 0, "Fiber Optic": 1}[internetservice]
    techsupport_encoded = 1 if techsupport == "Yes" else 0
    
    X=[[
        age,
        tenure,
        monthlycharges,
        totalcharges,
        gender_encoded,
        contracttype_encoded,
        internetservice_encoded,
        techsupport_encoded
    ]]
    X_array = scaler.transform(X)
    prob=model.predict_proba(X_array)[0][1]
    prediction="Yes" if prob>=0.5 else "No"
    if prob<0.40:
        risk="Low Risk"
    elif prob<0.65:
        risk="Medium Risk"
    else:
        risk = "High risk"
    st.write(f"Churn probability: {prob:.2f}")
    st.success(f"Prediction: {prediction}")
    st.write(f"Risk Level: {risk}")
else:
    st.write("Please enter the values")
