import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import google.generativeai as genai  # CHANGED: Import Google Gemini Library

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

# 3. Cache the SHAP Explainer
@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = get_explainer(model)

# 4. Header
st.title("🛡️ AI HealthGuard Dashboard")
st.markdown("""
**Welcome, Doctor.** This system uses a Random Forest Machine Learning model to assess heart disease risk.
It also provides **Explainable AI (XAI)** insights to justify predictions based on patient vitals.
""")

# --- CONFIGURING GEMINI API VIA SECRETS ---
# We no longer ask for input. We check the secrets file.
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
except FileNotFoundError:
    st.error("⚠️ Secrets file not found. Please create a `.streamlit/secrets.toml` file.")
    st.stop()
except KeyError:
    st.error("⚠️ API Key not found. Please add `GEMINI_API_KEY` to your `secrets.toml` file.")
    st.stop()

st.sidebar.header("Patient Data Input")

# 5. Input Fields (Your original logic)
def user_input_features():
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    
    cp = st.sidebar.selectbox("Chest Pain Type", options=[1, 2, 3, 4], 
                              format_func=lambda x: {
                                  1: "Type 1 (Moderate Risk)", 
                                  2: "Type 2 (Low Risk)", 
                                  3: "Type 3 (Low Risk)", 
                                  4: "Type 4 (High Risk / Asymptomatic)"
                              }[x])

    trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
    chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "True" if x == 1 else "False")
    
    restecg = st.sidebar.selectbox("Resting ECG Results", options=[0, 1, 2], 
                                   format_func=lambda x: {
                                        0: "Normal", 
                                        1: "ST-T Wave Abnormality", 
                                        2: "Left Ventricular Hypertrophy"
                                   }[x])

    thalach = st.sidebar.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak = st.sidebar.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2],
                                 format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x])

    ca = st.sidebar.slider("Number of Major Vessels (0-3)", 0, 3, 0)
    thal = st.sidebar.selectbox("Thalassemia", options=[1, 2, 3], 
                                format_func=lambda x: {1: "Normal", 2: "Fixed Defect", 3: "Reversable Defect"}[x])

    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# 6. Main Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Patient Vitals")
    st.write(input_df)

# 7. Prediction Logic
if st.button("Analyze Risk"):
    probability = model.predict_proba(input_df)
    risk_score = probability[0][1] 
    
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
        shap_values = explainer(input_df)
        
        # Handle 2D/3D array shapes from Random Forest
        if len(shap_values.values.shape) == 3:
             # Shape: (1 sample, n_features, 2 classes) -> Take class 1
            shap_values_class1 = shap_values[:, :, 1]
        else:
            shap_values_class1 = shap_values
            
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(shap_values_class1[0], show=False)
        st.pyplot(fig, bbox_inches='tight')
        
    st.markdown("""
    **How to read this chart:**
    * **Red bars** push the risk score **HIGHER**.
    * **Blue bars** push the risk score **LOWER**.
    """)

    # ============================================================
    # 9. LLM INTEGRATION LAYER (Google Gemini)
    # ============================================================
    st.divider()
    st.subheader("🤖 AI Medical Assistant Report")

    with st.spinner("Consulting AI Specialist (Gemini)..."):
        try:
            # A. Extract Top Features for the Prompt
            # We get feature names and their SHAP values
            feature_names = input_df.columns.tolist()
            shap_vals = shap_values_class1[0].values
            patient_vals = input_df.iloc[0].values
            
            # Combine into a list of tuples (Feature, Patient Value, SHAP Impact)
            features_impact = []
            for name, val, impact in zip(feature_names, patient_vals, shap_vals):
                features_impact.append((name, val, impact))
            
            # Sort by absolute impact (biggest drivers first)
            features_impact.sort(key=lambda x: abs(x[2]), reverse=True)
            
            # Take top 5 drivers
            top_drivers_text = ""
            for name, val, impact in features_impact[:5]:
                direction = "INCREASED risk" if impact > 0 else "DECREASED risk"
                top_drivers_text += f"- {name}: Patient Value = {val} ({direction})\n"

            # B. Construct the Prompt
            prompt = f"""
            You are a senior Cardiologist Assistant.
            
            PATIENT CONTEXT:
            - Age: {input_df['age'][0]}
            - Sex: {"Male" if input_df['sex'][0] == 1 else "Female"}
            - Predicted Heart Disease Risk: {risk_score:.1%}
            
            TOP DRIVERS OF THIS PREDICTION (SHAP Analysis):
            {top_drivers_text}
            
            TASK:
            Write a short clinical note for the doctor covering:
            1. **Reasoning:** Explain medically why the model predicted this risk based on the top drivers above.
            2. **Lifestyle:** Suggest 2-3 specific lifestyle improvements relevant to these factors.
            3. **Next Steps:** Suggest 1 diagnostic test (e.g. stress test, echo) if risk is high.
            4. **Disclaimer:** Remind the doctor this is AI-assisted and requires clinical verification.
            
            Keep it professional, concise, and medical.
            """

            # C. Call Google Gemini API
            # Note: 1.5-flash is generally faster/cheaper, but you can use 1.5-pro
            model_gemini = genai.GenerativeModel('gemini-2.5-flash')
            response = model_gemini.generate_content(prompt)
            
            # # D. Display Output
            # report = response.text
            # st.markdown(f"""
            # <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;border-left:5px solid #ff4b4b;">
            #     {report}
            # </div>
            # """, unsafe_allow_html=True)
            # D. Display Output
            report = response.text
            st.markdown(f"""
            <div style="background-color:#fff3e0;color:black;padding:20px;border-radius:10px;border-left:5px solid #ff4b4b;">
                {report}
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error generating report: {e}")

# Sidebar Info
st.sidebar.markdown("---")
st.sidebar.info("AI HealthGuard v2.0\nModel: Random Forest + SHAP + Gemini LLM")