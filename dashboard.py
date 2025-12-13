import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import google.generativeai as genai
import base64

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

# --- FUNCTION TO LOAD LOCAL FONT ---
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

# Load the local fonts (UPDATE THESE FILENAMES to match your actual files)
regular_font_b64 = get_base64_of_bin_file("SF-Pro-Regular.otf")
bold_font_b64 = get_base64_of_bin_file("SF-Pro-Bold.otf")

# Fallback if files are missing
font_face_css = ""

if regular_font_b64:
    font_face_css += f"""
    @font-face {{
        font-family: 'SFProRegular';
        src: url('data:font/otf;base64,{regular_font_b64}') format('opentype');
        font-weight: normal;
        font-style: normal;
    }}
    """

if bold_font_b64:
    font_face_css += f"""
    @font-face {{
        font-family: 'SFProBold';
        src: url('data:font/otf;base64,{bold_font_b64}') format('opentype');
        font-weight: bold;
        font-style: normal;
    }}
    """

# --- CUSTOM CSS ---
st.markdown(f"""
    <style>
    /* 1. Inject the Font Definitions */
    {font_face_css}

    /* 2. Apply Fonts Globally (Use Regular for Body) */
    html, body, [class*="css"] {{
        font-family: 'SFProRegular', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    /* 3. Apply Bold Font to Headings */
    h1, h2, h3, .stsubheader {{
        font-family: 'SFProBold', sans-serif;
        font-weight: 700 !important;
    }}
    
    /* 4. Center the Main Title */
    .block-container h1 {{
        text-align: center;
        padding-bottom: 20px;
    }}

    /* 5. Custom Red Button Styling */
    div.stButton > button:first-child {{
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        transition: 0.3s;
        font-size: 18px; 
    }}
    
    div.stButton > button:first-child:hover {{
        background-color: #D32F2F; 
        color: white;
        border: none;
    }}

    /* 6. Table Styling */
    .stTable {{
        font-size: 18px !important;
    }}
    th {{
        font-weight: bold !important;
        background-color: #f0f2f6;
    }}
    </style>
    """, unsafe_allow_html=True)

# ... (Rest of your code remains exactly the same) ...

# 3. Cache the SHAP Explainer
@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = get_explainer(model)

# 4. Header (Centered via CSS/Markdown)
st.markdown("<h1 style='text-align: center;'>🛡️ AI HealthGuard 🛡️</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; margin-bottom: 30px;'>
    <strong>Welcome, Doctor.</strong> This system uses Machine Learning to assess heart disease risk.<br>
    It also provides insights to justify predictions based on patient vitals.
</div>
""", unsafe_allow_html=True)

# --- CONFIGURING GEMINI API VIA SECRETS ---
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

# 5. Input Fields
def user_input_features():
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    
    cp = st.sidebar.selectbox("Chest Pain Type", options=[1, 2, 3, 4], 
                              format_func=lambda x: {
                                  1: "Typical Angina", 
                                  2: "Atypical Angina", 
                                  3: "Non-anginal Pain", 
                                  4: "Asymptomatic"
                              }[x])

    trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
    chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "True" if x == 1 else "False")
    
    restecg = st.sidebar.selectbox("Resting ECG Results", options=[0, 1, 2], 
                                   format_func=lambda x: {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}[x])

    thalach = st.sidebar.number_input("Max Heart Rate", min_value=50, max_value=250, value=150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak = st.sidebar.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    
    slope = st.sidebar.selectbox("Slope", options=[0, 1, 2],
                                 format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x])

    ca = st.sidebar.slider("Number of Major Vessels (0-3)", 0, 3, 0)
    thal = st.sidebar.selectbox("Thalassemia", options=[1, 2, 3], 
                                format_func=lambda x: {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}[x])

    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# 6. Main Layout - TABLE CHANGES
# We remove the columns here so the table takes the FULL width of the container.
st.subheader("Patient Vitals")

# Using st.table instead of st.dataframe makes it static but "bigger" and easier to read.
# We also use the CSS above to increase the font size.
st.table(input_df)

# 7. Prediction Logic with Centered Red Button
st.write("") # Spacer

# Keep the button centered (columns used here only for the button)
b1, b2, b3 = st.columns([2, 2, 2])

with b2:
    analyze = st.button("Analyze Risk", use_container_width=True)

if analyze:
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
            shap_values_class1 = shap_values[:, :, 1]
        else:
            shap_values_class1 = shap_values
            
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 5)) # Slightly wider figure
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

    with st.spinner("Consulting AI Specialist"):
        try:
            # A. Extract Top Features
            feature_names = input_df.columns.tolist()
            shap_vals = shap_values_class1[0].values
            patient_vals = input_df.iloc[0].values
            
            features_impact = []
            for name, val, impact in zip(feature_names, patient_vals, shap_vals):
                features_impact.append((name, val, impact))
            
            features_impact.sort(key=lambda x: abs(x[2]), reverse=True)
            
            top_drivers_text = ""
            for name, val, impact in features_impact[:5]:
                direction = "INCREASED risk" if impact > 0 else "DECREASED risk"
                top_drivers_text += f"- {name}: Patient Value = {val} ({direction})\n"

            # B. Prompt
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
            3. **Next Steps:** Suggest 1 diagnostic test if risk is high.
            4. **Disclaimer:** Remind the doctor this is AI-assisted.
            
            Keep it professional, concise, and medical.
            """

            # C. Call Gemini
            model_gemini = genai.GenerativeModel('gemini-2.5-pro') 
            response = model_gemini.generate_content(prompt)
            
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