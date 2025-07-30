import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Cardiovascular Risk Predictor")

st.title("Cardiovascular Disease Risk Predictor")
st.markdown("This app uses machine learning to predict the risk of heart disease based on health parameters.")

# Load trained model
try:
    model = joblib.load("model.pkl")
except FileNotFoundError:
    st.error("model.pkl not found in the folder. Please add it and restart the app.")
    st.stop()

# Input fields
age = st.number_input("Age", 1, 120, 30)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
exang = st.selectbox("Exercise-Induced Angina", [0, 1])
oldpeak = st.number_input("ST depression", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope of ST (0–2)", [0, 1, 2])
ca = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (1=Normal, 2=Fixed, 3=Reversible)", [1, 2, 3])

# Convert sex to binary
sex = 1 if sex == "Male" else 0

# Predict on button click
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("High risk of cardiovascular disease")
    else:
        st.success("Low risk of cardiovascular disease")