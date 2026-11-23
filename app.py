# app.py
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Income Predictor", layout="wide")
st.title("💵 Adult Income Classification")
st.write("Predict whether an individual's income exceeds $50,000 per year.")

# --- SIDEBAR ---
st.sidebar.header("📊 Model Performance")
st.sidebar.metric(label="Test Accuracy", value="85.0%") # Update with your actual metric

st.sidebar.header("🏆 Top 5 Important Features")
st.sidebar.write("1. **capital.gain** (0.25)")
st.sidebar.write("2. **age** (0.18)")
st.sidebar.write("3. **hours.per.week** (0.12)")
st.sidebar.write("4. **education.num** (0.09)")
st.sidebar.write("5. **marital.status_Married** (0.07)")

# --- LOAD MODEL DIRECTLY ---
@st.cache_resource
def load_model():
    if os.path.exists("best_adult_income_model.joblib"):
        return joblib.load("best_adult_income_model.joblib")
    return None

model = load_model()

if model is None:
    st.error("⚠️ Model file 'best_adult_income_model.joblib' not found in this folder.")
    st.stop()

# --- MAIN INPUTS ---
st.subheader("Enter Individual's Details")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 17, 90, 30)
    workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Local-gov", "State-gov", "Self-emp-inc", "Federal-gov"])
    education = st.selectbox("Education", ["Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th", "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th", "Preschool", "12th"])
    education_num = st.number_input("Education Number (1-16)", 1, 16, 13)
    marital_status = st.selectbox("Marital Status", ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"])
    occupation = st.selectbox("Occupation", ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"])

with col2:
    relationship = st.selectbox("Relationship", ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"])
    race = st.selectbox("Race", ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"])
    sex = st.selectbox("Sex", ["Male", "Female"])
    capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
    capital_loss = st.number_input("Capital Loss", 0, 5000, 0)
    hours_per_week = st.slider("Hours per week", 1, 99, 40)
    native_country = st.selectbox("Native Country", ["United-States", "Mexico", "Philippines", "Germany", "Canada", "India", "England", "Other"])

# --- PREDICTION BUTTON ---
if st.button("Predict Income", type="primary"):
    # Notice the DOT (.) instead of hyphen (-) in the keys below!
    input_data = pd.DataFrame([{
        "age": age, 
        "workclass": workclass, 
        "fnlwgt": 100000, 
        "education": education, 
        "education.num": education_num,       # Changed to dot
        "marital.status": marital_status,     # Changed to dot
        "occupation": occupation,
        "relationship": relationship, 
        "race": race, 
        "sex": sex,
        "capital.gain": capital_gain,         # Changed to dot
        "capital.loss": capital_loss,         # Changed to dot
        "hours.per.week": hours_per_week,     # Changed to dot
        "native.country": native_country      # Changed to dot
    }])
    
    try:
        prediction = model.predict(input_data)[0]
        result = ">50K" if prediction == 1 else "<=50K"
        
        if result == ">50K":
            st.success(f"🎉 Predicted Income: **{result}**")
        else:
            st.info(f"Predicted Income: **{result}**")
    except Exception as e:
        st.error(f"Error: {e}")