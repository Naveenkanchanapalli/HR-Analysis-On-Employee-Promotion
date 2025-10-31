import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("Model_xgb5.pkl")
encoders = joblib.load("label_encoders.pkl")

st.title("🚀 HR Employee Promotion Predictor")
st.markdown("Enter employee details to predict whether they will be promoted or not.")

# User Inputs
age = st.slider("Age", 20, 60, 30)
length_of_service = st.slider("Length of Service (in years)", 0, 35, 5)
no_of_trainings = st.selectbox("Number of Trainings", list(range(1, 10)))
previous_year_rating = st.selectbox("Previous Year Rating", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
avg_training_score = st.slider("Average Training Score", 40, 100, 60)
awards_won = st.selectbox("Awards Won?", [0, 1])
KPIs_met = st.selectbox("KPIs Met > 80%?", [0, 1])

department = st.selectbox("Department", encoders['department'].classes_)
region = st.selectbox("Region", encoders['region'].classes_)
education = st.selectbox("Education", encoders['education'].classes_)
gender = st.selectbox("Gender", encoders['gender'].classes_)
recruitment_channel = st.selectbox("Recruitment Channel", encoders['recruitment_channel'].classes_)

# Encode categorical inputs
input_data = pd.DataFrame({
    'department': [encoders['department'].transform([department])[0]],
    'region': [encoders['region'].transform([region])[0]],
    'education': [encoders['education'].transform([education])[0]],
    'gender': [encoders['gender'].transform([gender])[0]],
    'recruitment_channel': [encoders['recruitment_channel'].transform([recruitment_channel])[0]],
    'no_of_trainings': [no_of_trainings],
    'age': [age],
    'previous_year_rating': [previous_year_rating],
    'length_of_service': [length_of_service],
    'KPIs_met >80%': [KPIs_met],
    'awards_won?': [awards_won],
    'avg_training_score': [avg_training_score]
})

# Predict
if st.button("Predict Promotion"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("✅ The employee is **likely to be promoted**.")
    else:
        st.warning("⚠️ The employee is **not likely to be promoted**.")
