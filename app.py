# app.py
import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

st.title("Citi Bank Customer Churn Prediction Tool")
st.write("Predict whether a customer is likely to churn based on key attributes.")

# Input features
tenure = st.number_input("Tenure (months)", 0, 100, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 500.0, 50.0)
total_charges = st.number_input("Total Charges", 0.0, 5000.0, 1000.0)
contract_type = st.selectbox("Contract Type", ["Month-to-Month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# One-hot encoding (example mapping)
contract_map = {"Month-to-Month": [1,0], "One year": [0,1], "Two year": [0,0]}
internet_map = {"DSL": [1,0], "Fiber optic": [0,1], "No": [0,0]}

features = [tenure, monthly_charges, total_charges] + contract_map[contract_type] + internet_map[internet_service]
final_features = np.array(features).reshape(1, -1)

if st.button("Predict Churn"):
    prediction = model.predict(final_features)
    result = "Customer is likely to Churn" if prediction[0]==1 else "Customer is likely to Stay"
    st.success(result)
