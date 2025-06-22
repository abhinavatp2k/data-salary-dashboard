import streamlit as st
import pandas as pd
import joblib

model = joblib.load("salary_model.pkl")

st.title("💼 Data Job Salary Estimator")

job_title = st.selectbox("Job Title", ["Data Scientist", "Data Analyst", "ML Engineer", "Data Engineer"])
experience = st.selectbox("Experience Level", ["EN", "MI", "SE", "EX"])
employment = st.selectbox("Employment Type", ["FT", "PT", "CT", "FL"])
location = st.selectbox("Company Location", ["US", "IN", "DE", "CA", "GB"])
remote_ratio = st.slider("Remote Ratio (%)", 0, 100, 50)

input_df = pd.DataFrame([{
    'job_title': job_title,
    'experience_level': experience,
    'employment_type': employment,
    'company_location': location,
    'remote_ratio': remote_ratio
}])

st.subheader("🔍 Input Summary")
st.write(input_df)

prediction = model.predict(input_df)[0]
st.subheader(f"💰 Predicted Salary: **${prediction:,.0f}**")
