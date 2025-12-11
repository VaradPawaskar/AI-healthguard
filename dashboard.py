#%%
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the trained model
# Ensure 'rf_model.pkl' is in the same directory
try:
    model = joblib.load('rf_model.pkl')
except FileNotFoundError:
    st.error("Model file 'rf_model.pkl' not found. Please run train.py to generate it.")
    st.stop()

# 2. Page Configuration
st.set_page_config(
    page_title="AI HealthGuard | Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# 3. Header and Description
st.title("🛡️ AI HealthGuard Dashboard")
st.markdown("""
**Welcome, Doctor.** This system uses a Random Forest Machine Learning model (trained on the UCI Heart Disease dataset) 
to assess the likelihood of heart disease in patients based on their clinical parameters.
""")

st.sidebar.header("Patient Data Input")

# 4. Input Fields (Based on UCI Heart Disease Features)
def user_input_features():
    # Age
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=50)
    
    # Sex (0 = Female, 1 = Male)
    sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    
    # Chest Pain Type (cp)
    # Mapping based on typical UCI dataset standards
    cp = st.sidebar.selectbox("Chest Pain Type", options=[0, 1, 2, 3], 
                              format_func=lambda x: {
                                  0: "Typical Angina", 
                                  1: "Atypical Angina", 
                                  2: "Non-anginal Pain", 
                                  3: "Asymptomatic"
                              }[x])

    # Resting Blood Pressure (trestbps)
    trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)

    # Cholesterol (chol)
    chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)

    # Fasting Blood Sugar (fbs)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "True" if x == 1 else "False")

    # Resting ECG (restecg)
    restecg = st.sidebar.selectbox("Resting ECG Results", options=[0, 1, 2], 
                                   format_func=lambda x: {
                                       0: "Normal", 
                                       1: "ST-T Wave Abnormality", 
                                       2: "Left Ventricular Hypertrophy"
                                   }[x])

    # Max Heart Rate (thalach)
    thalach = st.sidebar.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)

    # Exercise Induced Angina (exang)
    exang = st.sidebar.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    # Oldpeak
    oldpeak = st.sidebar.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    # Slope
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2],
                                 format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x])

    # Number of Major Vessels (ca)
    ca = st.sidebar.slider("Number of Major Vessels (0-3)", 0, 3, 0)

    # Thalassemia (thal)
    thal = st.sidebar.selectbox("Thalassemia", options=[1, 2, 3], 
                                format_func=lambda x: {1: "Normal", 2: "Fixed Defect", 3: "Reversable Defect"}[x])

    # Create a DataFrame
    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# 5. Display User Input
st.subheader("Patient Vitals Overview")
st.write(input_df)

# 6. Prediction Logic
if st.button("Analyze Risk"):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    
    st.subheader("Assessment Result")
    
    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Heart Disease Detected")
        st.write(f"**Confidence:** {np.max(probability) * 100:.2f}%")
        st.warning("Recommendation: Immediate cardiology consultation recommended.")
    else:
        st.success(f"✅ Low Risk - Heart appears healthy")
        st.write(f"**Confidence:** {np.max(probability) * 100:.2f}%")
        st.info("Recommendation: Maintain healthy lifestyle and routine checkups.")

# 7. Sidebar Info
st.sidebar.markdown("---")
st.sidebar.info("AI HealthGuard v1.0\nModel: Random Forest Classifier")