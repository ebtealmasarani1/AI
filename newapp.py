import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Ladda modeller
model1 = joblib.load("model.pkl")
model2 = joblib.load("complement_model.pkl")

st.title("Model Comparison App")

# Inputs
income = st.number_input("Income", value=30000.0)
age = st.number_input("Age", value=40.0)
credit_score = st.number_input("Credit Score", value=600.0)

# Jämförelse
if st.button("Compare Models"):

    input_data = pd.DataFrame({
        "income": [income],
        "age": [age],
        "credit_score": [credit_score]
    })

    pred1 = model1.predict(input_data)[0]
    pred2 = model2.predict(input_data)[0]

    st.subheader("Results")

    st.write(
        "Decision Tree:",
        "Approved" if pred1 == 1 else "Denied"
    )

    st.write(
        "Logistic Regression:",
        "Approved" if pred2 == 1 else "Denied"
    )

    if pred1 == pred2:
        st.success("Models agree")
    else:
        st.warning("Models disagree")

# -------------------------
# Experimentläge
# -------------------------

if st.button("Run Experiment"):

    agree = 0
    disagree = 0

    disagreements = []

    for i in range(100):

        random_income = np.random.normal(30000, 8000)
        random_age = np.random.normal(40, 10)
        random_credit = np.random.normal(600, 100)

        test_data = pd.DataFrame({
            "income": [random_income],
            "age": [random_age],
            "credit_score": [random_credit]
        })

        p1 = model1.predict(test_data)[0]
        p2 = model2.predict(test_data)[0]

        if p1 == p2:
            agree += 1
        else:
            disagree += 1

            disagreements.append({
                "income": round(random_income, 2),
                "age": round(random_age, 2),
                "credit_score": round(random_credit, 2),
                "decision_tree": p1,
                "logistic_regression": p2
            })

    st.subheader("Experiment Results")

    st.write("Agree:", agree)
    st.write("Disagree:", disagree)

    if disagreements:
        st.subheader("Examples of Disagreement")

        st.write(pd.DataFrame(disagreements[:5]))