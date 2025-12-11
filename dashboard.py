#%%
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# 1. Load the trained model
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

# 3. Cache the SHAP Explainer (Optimization)
# We use @st.cache_resource because the explainer object is large and static
@st.cache_resource
def get_explainer(_model):
    # TreeExplainer is optimized for Random Forest, XGBoost, etc.
    return shap.TreeExplainer(_model)

explainer = get_explainer(model)

# 4. Header and Description
st.title("🛡️ AI HealthGuard Dashboard")
st.markdown("""
**Welcome, Doctor.** This system uses a Random Forest Machine Learning model to assess heart disease risk.
It also provides **Explainable AI (XAI)** insights to justify predictions based on patient vitals.
""")

st.sidebar.header("Patient Data Input")

# 5. Input Fields (Based on UCI Heart Disease Features)
def user_input_features():
    # Age
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=50)
    
    # Sex
    sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    
    # Chest Pain Type (cp)
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

    # Create a DataFrame with correct column names matching training
    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# 6. Main Layout: Two Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Patient Vitals")
    st.write(input_df)

# 7. Prediction Logic
if st.button("Analyze Risk"):
    # Probability prediction (returns [prob_class_0, prob_class_1])
    # Note: Ensure your model returns standard sklearn probabilities
    probability = model.predict_proba(input_df)
    risk_score = probability[0][1] # Probability of Class 1 (Disease)
    
    st.divider()
    
    # Display Result
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:
        if risk_score > 0.5:
            st.error(f"⚠️ High Risk Detected")
            st.metric(label="Risk Probability", value=f"{risk_score * 100:.1f}%")
        else:
            st.success(f"✅ Low Risk Profile")
            st.metric(label="Risk Probability", value=f"{risk_score * 100:.1f}%")
            
    with col_res2:
        st.info("The chart below explains WHY the model made this prediction.")

    # 8. SHAP Explanation
    st.subheader("Model Decision Factors (SHAP Analysis)")
    with st.spinner("Calculating explainability scores..."):
        # Calculate SHAP values for this specific instance
        shap_values = explainer(input_df)
        
        # We need to access the values for class 1 (Disease)
        # Random Forests in sklearn return values for both classes [0, 1]
        # We handle this by checking the shape. 
        if len(shap_values.values.shape) == 3:
             # Shape: (1 sample, n_features, 2 classes) -> Take class 1
            shap_values_class1 = shap_values[:, :, 1]
        else:
            shap_values_class1 = shap_values
            
        # Create the plot using Matplotlib
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(shap_values_class1[0], show=False)
        st.pyplot(fig, bbox_inches='tight')
        
    st.markdown("""
    **How to read this chart:**
    * **Red bars** push the risk score **HIGHER**.
    * **Blue bars** push the risk score **LOWER**.
    * The values (E[f(x)]) represent the average base rate, and f(x) is the final prediction score.
    """)

# Sidebar Info
st.sidebar.markdown("---")
st.sidebar.info("AI HealthGuard v2.0\nModel: Random Forest + SHAP")
# %%
