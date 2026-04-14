import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# App Config
st.set_page_config(page_title="Scholarship Predictor", page_icon="🎓")

# Load Model
@st.cache_resource
def load_model():
    with open('scholarship_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()
CSV_FILE = 'scholarship_results.csv'

st.title("🎓 Scholarship Eligibility Portal")

# Sidebar for History
if os.path.exists(CSV_FILE):
    if st.sidebar.checkbox("Show Recent Applications"):
        history_df = pd.read_csv(CSV_FILE)
        st.sidebar.write(history_df.tail(10))

# Main Form
with st.form("prediction_form"):
    st.subheader("Student Details")
    name = st.text_input("Student Name")
    college = st.text_input("College Name")
    
    col1, col2 = st.columns(2)
    with col1:
        score = st.number_input("Academic Score", min_value=0.0, max_value=100.0, value=75.0)
        income = st.number_input("Household Income", min_value=0.0, step=500.0)
    
    with col2:
        semester = st.number_input("Semester", min_value=1, max_value=8, value=1)
        extra = st.selectbox("Extracurricular Activity", ["Yes", "No"])
    
    submit = st.form_submit_button("Predict & Save")

if submit:
    # Prepare data for prediction
    extra_val = 1 if extra == "Yes" else 0
    features = np.array([[score, income, extra_val]])
    
    # Predict
    prediction = model.predict(features)[0]
    result = "Eligible" if prediction == 1 else "not Eligible"
    
    # Display Result
    if prediction == 1:
        st.success(f"Result: {name} is ELIGIBLE!")
        st.balloons()
    else:
        st.error(f"Result: {name} is NOT ELIGIBLE.")

    # Save to CSV
    new_data = pd.DataFrame([{
        "Student Name": name,
        "college": college,
        "semester": semester,
        "score": score,
        "income": income,
        "Extracurricular": extra_val,
        "Status": result
    }])
    
    new_data.to_csv(CSV_FILE, mode='a', header=not os.path.exists(CSV_FILE), index=False)
    st.info("Record saved to database.")