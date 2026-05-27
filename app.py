import streamlit as st
import pandas as pd
import joblib

# Ladda modellen
model = joblib.load("model.pkl")

# Titel
st.title("Loan Approval App")

# Inputs
income = st.number_input("Income", value=30000.0)
age = st.number_input("Age", value=40.0)
credit_score = st.number_input("Credit Score", value=600.0)

# Prediktion
if st.button("Predict"):

    input_data = pd.DataFrame({
        "income": [income],
        "age": [age],
        "credit_score": [credit_score]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Denied")