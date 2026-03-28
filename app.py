import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --- STEP 1: UI SETUP ---
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Predictor")
st.markdown("""
This app predicts a student's final exam score based on their study habits using **Linear Regression**.
""")

# --- STEP 2: DATA GENERATION (Simulating a Dataset) ---
# In a real BYOP, you would load a CSV here.
@st.cache_data
def load_data():
    np.random.seed(42)
    n_points = 100
    study_hours = np.random.randint(1, 10, n_points)
    attendance = np.random.randint(60, 100, n_points)
    # Formula: Score = (Study * 5) + (Attendance * 0.5) + noise
    scores = (study_hours * 5) + (attendance * 0.4) + np.random.normal(0, 2, n_points)
    
    df = pd.DataFrame({
        'Study_Hours': study_hours,
        'Attendance_Rate': attendance,
        'Final_Score': scores.clip(0, 100)
    })
    return df

df = load_data()

# --- STEP 3: MODEL TRAINING ---
X = df[['Study_Hours', 'Attendance_Rate']]
y = df['Final_Score']
model = LinearRegression()
model.fit(X, y)

# --- STEP 4: USER INPUT (Sidebar) ---
st.sidebar.header("Input Student Data")
user_study = st.sidebar.slider("Daily Study Hours", 0, 12, 5)
user_attendance = st.sidebar.slider("Attendance Percentage (%)", 0, 100, 85)

# --- STEP 5: PREDICTION ---
prediction = model.predict([[user_study, user_attendance]])[0]

# --- STEP 6: DISPLAY RESULTS ---
col1, col2 = st.columns(2)

with col1:
    st.metric(label="Predicted Final Score", value=f"{prediction:.2f} / 100")

with col2:
    if prediction >= 75:
        st.success("Status: Distinction Likely")
    elif prediction >= 40:
        st.warning("Status: Passing Grade")
    else:
        st.error("Status: Risk of Failure")

# Technical Insights for your Project Report
with st.expander("View Model Insights (Syllabus Concepts)"):
    st.write(f"**Linear Regression Weights:**")
    st.write(f"Study Hours Impact: {model.coef_[0]:.2f}")
    st.write(f"Attendance Impact: {model.coef_[1]:.2f}")
    st.info("This model uses supervised learning to map features to a continuous output.")