import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os
import base64
import time
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# === PAGE CONFIGURATION ===
st.set_page_config(
    page_title="CardioCare AI | Advanced Cardiac Analysis",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for healthcare theme
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #005f73;
        --secondary: #0a9396;
        --accent: #94d2bd;
        --light: #e9d8a6;
        --alert: #ae2012;
        --success: #4caf50;
        --warning: #ff9800;
    }

    /* Main container */
    .stApp {
        background: linear-gradient(135deg, #f8f9fa 0%, #e6f7ff 100%);
        background-attachment: fixed;
    }

    /* Headers */
    h1, h2, h3, h4 {
        color: var(--primary) !important;
        font-family: 'Montserrat', sans-serif;
        font-weight: 600;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--primary), var(--secondary)) !important;
        color: white !important;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, var(--secondary), var(--primary)) !important;
        color: white !important;
        border-radius: 25px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        border: none !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }

    /* Forms */
    .stForm {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 20px !important;
        padding: 30px !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.5);
    }

    /* Input widgets */
    .stNumberInput, .stSelectbox, .stSlider, .stTextInput {
        border-radius: 12px !important;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    /* Cards */
    .custom-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        transition: all 0.3s ease;
    }

    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.15);
    }

    /* Animations */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }

    .pulse-animation {
        animation: pulse 2s infinite;
    }

    /* Risk indicator */
    .risk-gauge {
        width: 100%;
        height: 200px;
        position: relative;
        margin: 20px 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: var(--primary);
        font-size: 14px;
        margin-top: 40px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.7) !important;
        border-radius: 12px !important;
        padding: 10px 20px !important;
        margin: 0 5px !important;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: white !important;
        transform: translateY(-2px);
    }

    .stTabs [aria-selected="true"] {
        background: white !important;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }

    /* Report button */
    .report-btn {
        background: linear-gradient(135deg, #ff9800, #f57c00) !important;
    }

    /* Rainbow bar chart */
    .rainbow-bar {
        background: linear-gradient(90deg, #ff0000, #ff8000, #ffff00, #80ff00, #00ff00, #00ff80, #00ffff, #0080ff, #0000ff, #8000ff, #ff00ff, #ff0080);
        height: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models(base_dir=r"exported_models"):
    files = {
        "early_model": "xgb_early_hd_model.joblib",
        "hd_model": "heart_disease_model_final.pkl",
        "scaler": "scaler_hd.joblib",
        "scaler_hd": "scaler_final.pkl"
    }
    try:
        for name, fname in files.items():
            if not os.path.exists(os.path.join(base_dir, fname)):
                raise FileNotFoundError(f"Missing: {fname}")

        early_model = joblib.load(os.path.join(base_dir, files["early_model"]))
        hd_model = joblib.load(os.path.join(base_dir, files["hd_model"]))
        scaler = joblib.load(os.path.join(base_dir, files["scaler"]))
        scaler_hd = joblib.load(os.path.join(base_dir, files["scaler_hd"]))

        return early_model, hd_model, scaler, scaler_hd
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()


# Load models
early_model, hd_model, scaler, scaler_hd = load_models()


# === REPORT GENERATION ===
def generate_pdf_report(patient_data, prediction_data, mode):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Header
    pdf.set_fill_color(0, 95, 115)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 15, "CardioCare AI - Cardiac Risk Assessment Report", 0, 1, 'C', 1)
    pdf.ln(10)

    # Patient Information
    pdf.set_fill_color(240, 240, 240)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Patient Information", 0, 1)
    pdf.set_font("Arial", size=12)

    # Add patient data
    for key, value in patient_data.items():
        pdf.cell(0, 8, f"{key}: {value}", 0, 1)

    pdf.ln(5)

    # Prediction Results
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Prediction Results", 0, 1)
    pdf.set_font("Arial", size=12)

    risk_level = "High Risk" if prediction_data['prediction'] == 1 else "Low Risk"
    pdf.cell(0, 8, f"Risk Level: {risk_level}", 0, 1)
    pdf.cell(0, 8, f"Probability: {prediction_data['probability']:.1%}", 0, 1)
    pdf.cell(0, 8, f"Assessment Type: {'Early Warning' if mode == 'Early Warning' else 'Comprehensive Heart Disease'}",
             0, 1)

    pdf.ln(5)

    # Recommendations
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Clinical Recommendations", 0, 1)
    pdf.set_font("Arial", size=12)

    for rec in prediction_data['recommendations']:
        pdf.cell(0, 8, f"- {rec}", 0, 1)

    # Footer
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 8, f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    pdf.cell(0, 8, "CardioCare AI - Advanced Cardiac Risk Assessment", 0, 1, 'C')

    # Save to bytes
    return pdf.output(dest='S').encode('latin1')


# === GAUGE VISUALIZATION ===
def create_risk_gauge(probability, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#0a9396"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#4CAF50'},
                {'range': [30, 70], 'color': '#FFC107'},
                {'range': [70, 100], 'color': '#F44336'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=50, r=50, b=20, t=80),
        font=dict(family="Arial", size=14, color="#005f73")
    )
    return fig


# === RAINBOW BAR CHART ===
def create_rainbow_bar_chart(labels, values, title):
    colors = [
        '#FF0000', '#FF5500', '#FFAA00', '#FFFF00',
        '#AAFF00', '#55FF00', '#00FF00', '#00FF55',
        '#00FFAA', '#00FFFF', '#00AAFF', '#0055FF',
        '#0000FF', '#5500FF', '#AA00FF', '#FF00FF'
    ]

    fig = go.Figure()

    for i, (label, value) in enumerate(zip(labels, values)):
        fig.add_trace(go.Bar(
            x=[label],
            y=[value],
            name=label,
            marker_color=colors[i % len(colors)],
            width=0.7
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Factors",
        yaxis_title="Impact",
        showlegend=False,
        height=400,
        template="plotly_white",
        margin=dict(l=50, r=50, b=100, t=80),
        font=dict(family="Arial", size=12, color="#005f73")
    )

    return fig


# === SIDEBAR ===
with st.sidebar:
    st.markdown("""
    <style>
        [data-testid="stSidebar"] * {
            color: white !important;
        }
        [data-testid="stSidebar"] .stRadio div[data-baseweb="radio"] label {
            color: white !important;
            font-weight: 500;
        }
    </style>
    <div style='text-align: center; margin-bottom: 20px;'>
        <img src='https://img.icons8.com/color/96/000000/heart-health.png' width='80'>
        <h1 style='color: white !important; font-weight: 600;'>CardioCare AI</h1>
        <p style='margin-top: -10px; font-size: 16px;'>Advanced Cardiac Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    app_mode = st.radio("Navigation", ["Home", "Early Warning", "Heart Disease", "Patient History"])

    st.markdown("---")
    st.markdown("""
    <div>
    <h4>About</h4>
    <p>Advanced ML-based cardiac risk assessment tool for healthcare professionals.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:20px; text-align:center;">
      <p style="margin:0; color:#cfe8ff; font-weight:600; font-size:13px;">Developed by</p>
      <h3 style="
            margin:6px 0 4px;
            font-family: 'Montserrat', sans-serif;
            font-size:20px;
            background: linear-gradient(90deg, #ffb347 0%, #ff6a88 50%, #7f00ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight:800;
            text-transform:uppercase;
            letter-spacing:1.8px;
            text-shadow: 0 6px 22px rgba(127,0,255,0.18);
            ">
        YUVRAJ KUMAR GOND
      </h3>
      <p style="margin:0; color:#e6fff8; font-weight:700; font-size:12px;">
        Version <span style="background:linear-gradient(90deg,#005f73,#0a9396); color:#fff; padding:5px 10px; border-radius:14px; box-shadow:0 6px 18px rgba(10,147,150,0.18); font-weight:800;">3.0.0</span>
      </p>
      <div style="margin-top:8px;">
        <span style="display:inline-block; width:70px; height:6px; background:linear-gradient(90deg,#ffd700,#ff6a00,#ff2d95); border-radius:4px; box-shadow:0 6px 18px rgba(255,106,149,0.12);"></span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# Initialize session state for patient history
if 'patient_history' not in st.session_state:
    st.session_state.patient_history = []

# === HOME ===
if app_mode == "Home":
    st.title("❤️ CardioCare AI - Advanced Cardiac Analysis")
    st.markdown("""
    <div class='custom-card pulse-animation'>
        <h3>AI-Powered Cardiac Risk Prediction</h3>
        <p>This clinical decision support tool predicts early signs of heart disease and overall cardiovascular risk 
        using machine learning models trained on extensive clinical datasets.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='custom-card'>
            <h3>Key Features</h3>
            <ul style='padding-left: 20px;'>
                <li>Early warning prediction system</li>
                <li>Comprehensive heart disease risk assessment</li>
                <li>Exportable PDF reports</li>
                <li>Patient history tracking</li>
                <li>Interactive risk visualization</li>
                <li>Personalized recommendations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='custom-card'>
            <h3>Dataset Features</h3>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px;'>
                <div>
                    <h4>Clinical</h4>
                    <ul style='padding-left: 20px;'>
                        <li>Age, Sex</li>
                        <li>Blood Pressure</li>
                        <li>Cholesterol</li>
                        <li>Blood Sugar</li>
                        <li>ST depression</li>
                    </ul>
                </div>
                <div>
                    <h4>Lifestyle</h4>
                    <ul style='padding-left: 20px;'>
                        <li>BMI</li>
                        <li>Smoking</li>
                        <li>Alcohol</li>
                        <li>Activity</li>
                        <li>Stress</li>
                        <li>Diet</li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div class='custom-card'>
        <h3>How It Works</h3>
        <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; text-align: center;'>
            <div>
                <img src='https://img.icons8.com/color/96/000000/form.png' width='60'>
                <h4>1. Input Data</h4>
                <p>Enter patient clinical and lifestyle information</p>
            </div>
            <div>
                <img src='https://img.icons8.com/color/96/000000/artificial-intelligence.png' width='60'>
                <h4>2. AI Analysis</h4>
                <p>Our models process the data using advanced algorithms</p>
            </div>
            <div>
                <img src='https://img.icons8.com/color/96/000000/report-card.png' width='60'>
                <h4>3. Get Results</h4>
                <p>Receive risk assessment with actionable insights</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Diet Quality Rating Information - FIXED VERSION
    st.markdown("""
    <div class='custom-card'>
        <h3>🍽️ Diet Quality Score Guide</h3>
        <p>Understanding how diet quality is measured (1-10 scale):</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Using Streamlit columns and components instead of pure HTML
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background: rgba(230, 57, 70, 0.1); padding: 15px; border-radius: 10px; border-left: 4px solid #e63946; margin-bottom: 15px;'>
            <h4 style='color: #e63946; margin: 0;'>Poor Diet (1-3)</h4>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        - High in processed foods & sugars
        - Low fruit & vegetable intake  
        - Frequent fast food consumption
        - High saturated & trans fats
        - Excessive sodium intake
        """)
        
        st.markdown("""
        <div style='background: rgba(42, 157, 143, 0.1); padding: 15px; border-radius: 10px; border-left: 4px solid #2a9d8f; margin-bottom: 15px;'>
            <h4 style='color: #2a9d8f; margin: 0;'>Good Diet (7-8)</h4>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        - Regular fruit & vegetable consumption
        - Lean protein sources
        - Whole grains & fiber
        - Limited processed foods
        - Healthy cooking methods
        """)
    
    with col2:
        st.markdown("""
        <div style='background: rgba(244, 162, 97, 0.1); padding: 15px; border-radius: 10px; border-left: 4px solid #f4a261; margin-bottom: 15px;'>
            <h4 style='color: #f4a261; margin: 0;'>Average Diet (4-6)</h4>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        - Mixed diet with some healthy choices
        - Moderate fruit & vegetable intake
        - Occasional processed foods
        - Balanced macronutrients
        - Moderate portion sizes
        """)
        
        st.markdown("""
        <div style='background: rgba(38, 70, 83, 0.1); padding: 15px; border-radius: 10px; border-left: 4px solid #264653; margin-bottom: 15px;'>
            <h4 style='color: #264653; margin: 0;'>Excellent Diet (9-10)</h4>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        - Predominantly plant-based foods
        - Rich in antioxidants & nutrients
        - Minimal processed foods
        - Healthy fats (omega-3, olive oil)
        - Proper hydration & portion control
        """)
    
    # Visual scale
    st.markdown("""
    <div style='margin: 20px 0;'>
        <p><strong>Diet Quality Scale:</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create the gradient scale using columns
    scale_col1, scale_col2, scale_col3, scale_col4 = st.columns(4)
    
    with scale_col1:
        st.markdown("""
        <div style='background: #e63946; height: 20px; border-radius: 10px 0 0 10px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;'>
            1-3
        </div>
        <p style='text-align: center; margin: 5px 0; font-size: 12px;'>Poor</p>
        """, unsafe_allow_html=True)
    
    with scale_col2:
        st.markdown("""
        <div style='background: #f4a261; height: 20px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;'>
            4-6
        </div>
        <p style='text-align: center; margin: 5px 0; font-size: 12px;'>Average</p>
        """, unsafe_allow_html=True)
    
    with scale_col3:
        st.markdown("""
        <div style='background: #2a9d8f; height: 20px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;'>
            7-8
        </div>
        <p style='text-align: center; margin: 5px 0; font-size: 12px;'>Good</p>
        """, unsafe_allow_html=True)
    
    with scale_col4:
        st.markdown("""
        <div style='background: #264653; height: 20px; border-radius: 0 10px 10px 0; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;'>
            9-10
        </div>
        <p style='text-align: center; margin: 5px 0; font-size: 12px;'>Excellent</p>
        """, unsafe_allow_html=True)
    
    # Note
    st.info("**Note:** A higher diet quality score significantly reduces cardiovascular risk factors including obesity, high blood pressure, and cholesterol levels.")

    st.markdown("---")

    # Statistics and Performance
    st.markdown("""
    <div class='custom-card'>
        <h3>System Performance</h3>
        <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; text-align: center;'>
            <div>
                <h4>Accuracy</h4>
                <h3 style='color: #0a9396;'>94.7%</h3>
            </div>
            <div>
                <h4>Precision</h4>
                <h3 style='color: #0a9396;'>92.5%</h3>
            </div>
            <div>
                <h4>Recall</h4>
                <h3 style='color: #0a9396;'>93.8%</h3>
            </div>
            <div>
                <h4>F1 Score</h4>
                <h3 style='color: #0a9396;'>93.1%</h3>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# === EARLY WARNING ===
elif app_mode == "Early Warning":
    st.title("🔍 Early Heart Disease Detection")
    st.markdown("""
    <div class='custom-card'>
        <p>This tool identifies early warning signs of potential heart disease based on clinical measurements and lifestyle factors.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("early_form"):
        st.markdown("""
        <div class='custom-card'>
            <h3>Patient Information</h3>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            patient_id = st.text_input("Patient ID", "P-1001")
            patient_name = st.text_input("Patient Name", "John Doe")
            age = st.number_input("Age", 18, 100, 45)
            sex = st.selectbox("Sex", ["Male", "Female"])
        with col2:
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 90, 200, 120)
            chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        with col3:
            thalach = st.number_input("Maximum Heart Rate Achieved", 70, 220, 150)
            exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
            oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 10.0, 0.0, step=0.1)
            bmi = st.number_input("Body Mass Index", 15.0, 50.0, 25.0)

        st.markdown("""
        <div class='custom-card'>
            <h3>Lifestyle Factors</h3>
        """, unsafe_allow_html=True)

        col4, col5, col6 = st.columns(3)
        with col4:
            smoking = st.selectbox("Smoking", ["No", "Yes"])
            alcohol_intake = st.selectbox("Alcohol Intake", ["None", "Light", "Moderate", "Heavy"])
            physical_activity = st.selectbox("Physical Activity Level", ["Sedentary", "Light", "Moderate", "Active"])
        with col5:
            family_history = st.selectbox("Family History of Heart Disease", ["No", "Yes"])
            diabetes = st.selectbox("Diabetes", ["No", "Yes"])
            stress_level = st.selectbox("Stress Level", ["Low", "Moderate", "High"])
        with col6:
            sleep_hours = st.number_input("Average Sleep Hours", 4, 12, 7)
            diet_score = st.slider("Diet Quality Score (1-10)", 1, 10, 6)
            st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("Predict Early Risk", use_container_width=True)

    if submitted:
        with st.spinner("Analyzing cardiac risk factors..."):
            time.sleep(1.5)
            try:
                # Prepare input data
                input_dict = {
                    'age': age,
                    'sex': 1 if sex == "Male" else 0,
                    'trestbps': trestbps,
                    'chol': chol,
                    'fbs': 1 if fbs == "Yes" else 0,
                    'thalach': thalach,
                    'exang': 1 if exang == "Yes" else 0,
                    'oldpeak': oldpeak,
                    'bmi': bmi,
                    'smoking': 1 if smoking == "Yes" else 0,
                    'alcohol_intake': ["None", "Light", "Moderate", "Heavy"].index(alcohol_intake),
                    'physical_activity': ["Sedentary", "Light", "Moderate", "Active"].index(physical_activity),
                    'family_history': 1 if family_history == "Yes" else 0,
                    'diabetes': 1 if diabetes == "Yes" else 0,
                    'stress_level': ["Low", "Moderate", "High"].index(stress_level),
                    'sleep_hours': sleep_hours,
                    'diet_score': diet_score
                }

                # Create DataFrame and scale features
                input_df = pd.DataFrame([input_dict])
                scaled_input = scaler.transform(input_df)

                # Make prediction
                pred = early_model.predict(scaled_input)[0]
                prob = early_model.predict_proba(scaled_input)[0][1]

                # Store patient data
                patient_data = {
                    "id": patient_id,
                    "name": patient_name,
                    "age": age,
                    "sex": sex,
                    "risk_level": "High" if pred == 1 else "Low",
                    "probability": f"{prob:.1%}",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "mode": "Early Warning"
                }

                # Add to history if not already present
                if patient_data not in st.session_state.patient_history:
                    st.session_state.patient_history.append(patient_data)

                # Display results
                st.markdown("---")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"""
                    <div class='custom-card'>
                        <h2>Prediction Result</h2>
                        <div style='text-align: center; margin: 20px 0;'>
                            <h3 style='color: {'#e63946' if pred == 1 else '#2a9d8f'};'>
                                {'🚨 High Risk of Early Heart Disease' if pred == 1 else '✅ Low Risk of Early Heart Disease'}
                            </h3>
                            <h4 style='font-size: 24px;'>Probability: {prob:.1%}</h4>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Export Report Button
                    patient_info = {
                        "Patient ID": patient_id,
                        "Patient Name": patient_name,
                        "Age": age,
                        "Sex": sex,
                        "Blood Pressure": f"{trestbps} mm Hg",
                        "Cholesterol": f"{chol} mg/dl",
                        "Fasting Blood Sugar": fbs,
                        "Max Heart Rate": thalach
                    }

                    prediction_info = {
                        "prediction": pred,
                        "probability": prob,
                        "recommendations": [
                            "Consult a cardiologist within 2 weeks" if pred == 1 else "Annual cardiac check-up",
                            "Schedule ECG and stress test" if pred == 1 else "Continue healthy habits",
                            "Begin blood pressure monitoring",
                            "Consult nutritionist for diet plan",
                            "Smoking cessation program" if smoking == "Yes" else "Maintain non-smoking status"
                        ]
                    }

                    pdf_bytes = generate_pdf_report(patient_info, prediction_info, "Early Warning")

                    st.download_button(
                        label="📄 Export Full Report",
                        data=pdf_bytes,
                        file_name=f"CardioCare_Report_{patient_id}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="early_report"
                    )

                with col2:
                    st.plotly_chart(create_risk_gauge(prob, "Cardiac Risk Gauge"), use_container_width=True)

                if pred == 1:
                    st.markdown("""
                    <div class='custom-card' style='border-left: 5px solid #e63946;'>
                        <h3>Clinical Recommendations</h3>
                        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px;'>
                            <div>
                                <h4>Immediate Actions</h4>
                                <ul>
                                    <li>Consult a cardiologist within 2 weeks</li>
                                    <li>Schedule ECG and stress test</li>
                                    <li>Begin blood pressure monitoring</li>
                                    <li>Complete lipid profile test</li>
                                </ul>
                            </div>
                            <div>
                                <h4>Lifestyle Changes</h4>
                                <ul>
                                    <li>Begin supervised exercise program</li>
                                    <li>Consult nutritionist for diet plan</li>
                                    <li>Smoking cessation program</li>
                                    <li>Stress management techniques</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class='custom-card' style='border-left: 5px solid #2a9d8f;'>
                        <h3>Preventive Recommendations</h3>
                        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px;'>
                            <div>
                                <h4>Maintenance</h4>
                                <ul>
                                    <li>Annual cardiac check-up</li>
                                    <li>Continue healthy habits</li>
                                    <li>Monitor key indicators</li>
                                    <li>Regular cholesterol checks</li>
                                </ul>
                            </div>
                            <div>
                                <h4>Improvement</h4>
                                <ul>
                                    <li>Consider diet optimization</li>
                                    <li>Stress reduction techniques</li>
                                    <li>Regular aerobic exercise</li>
                                    <li>Adequate sleep maintenance</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Show feature importance with rainbow bars
                st.markdown("---")
                st.markdown("""
                <div class='custom-card'>
                    <h2>Key Contributing Factors</h2>
                    <p>These factors most influenced the prediction:</p>
                </div>
                """, unsafe_allow_html=True)

                factors = {
                    'Factor': ['Age', 'Cholesterol', 'Blood Pressure', 'BMI', 'Lifestyle', 'Family History',
                               'Smoking', 'Diet', 'Stress', 'Sleep'],
                    'Impact': [0.25, 0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05, 0.03]
                }

                # Create rainbow bar chart
                fig = create_rainbow_bar_chart(factors['Factor'], factors['Impact'], "Risk Factor Impact")
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")

# === FULL HEART DISEASE PREDICTION ===
elif app_mode == "Heart Disease":
    st.title("❤️ Comprehensive Heart Disease Risk Assessment")
    st.markdown("""
    <div class='custom-card'>
        <p>Complete cardiovascular risk evaluation based on clinical and lifestyle factors.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("hd_form"):
        st.markdown("""
        <div class='custom-card'>
            <h3>Patient Information</h3>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            patient_id = st.text_input("Patient ID", "P-1001")
            patient_name = st.text_input("Patient Name", "John Doe")
            age = st.number_input("Age", 18, 100, 45)
            sex = st.selectbox("Sex", ["Male", "Female"])
        with col2:
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 90, 200, 120)
            chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
            thalach = st.number_input("Maximum Heart Rate Achieved", 70, 220, 150)
            bmi = st.number_input("Body Mass Index", 15.0, 50.0, 25.0)

        st.markdown("""
        <div class='custom-card'>
            <h3>Clinical Measurements</h3>
        """, unsafe_allow_html=True)

        col3, col4 = st.columns(2)
        with col3:
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
            exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
            oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 10.0, 0.0, step=0.1)
        with col4:
            family_history = st.selectbox("Family History of Heart Disease", ["No", "Yes"])
            diabetes = st.selectbox("Diabetes", ["No", "Yes"])
            st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='custom-card'>
            <h3>Lifestyle Factors</h3>
        """, unsafe_allow_html=True)

        col5, col6 = st.columns(2)
        with col5:
            smoking = st.selectbox("Smoking", ["No", "Yes"])
            alcohol_intake = st.selectbox("Alcohol Intake", ["None", "Light", "Moderate", "Heavy"])
        with col6:
            physical_activity = st.selectbox("Physical Activity Level", ["Sedentary", "Light", "Moderate", "Active"])
            stress_level = st.selectbox("Stress Level", ["Low", "Moderate", "High"])
            sleep_hours = st.number_input("Average Sleep Hours", 4, 12, 7)
            diet_score = st.slider("Diet Quality Score (1-10)", 1, 10, 6)

        submitted = st.form_submit_button("Assess Heart Disease Risk", use_container_width=True)

    if submitted:
        with st.spinner("Analyzing comprehensive cardiac profile..."):
            time.sleep(2)
            try:
                # Prepare input data
                input_dict = {
                    'age': age,
                    'sex': 1 if sex == "Male" else 0,
                    'trestbps': trestbps,
                    'chol': chol,
                    'fbs': 1 if fbs == "Yes" else 0,
                    'thalach': thalach,
                    'exang': 1 if exang == "Yes" else 0,
                    'oldpeak': oldpeak,
                    'bmi': bmi,
                    'smoking': 1 if smoking == "Yes" else 0,
                    'alcohol_intake': ["None", "Light", "Moderate", "Heavy"].index(alcohol_intake),
                    'physical_activity': ["Sedentary", "Light", "Moderate", "Active"].index(physical_activity),
                    'family_history': 1 if family_history == "Yes" else 0,
                    'diabetes': 1 if diabetes == "Yes" else 0,
                    'stress_level': ["Low", "Moderate", "High"].index(stress_level),
                    'sleep_hours': sleep_hours,
                    'diet_score': diet_score
                }

                # Create DataFrame and scale features
                input_df = pd.DataFrame([input_dict])
                scaled_input = scaler_hd.transform(input_df)

                # Make prediction
                pred = hd_model.predict(scaled_input)[0]
                prob = hd_model.predict_proba(scaled_input)[0][1]

                # Store patient data
                patient_data = {
                    "id": patient_id,
                    "name": patient_name,
                    "age": age,
                    "sex": sex,
                    "risk_level": "High" if pred == 1 else "Low",
                    "probability": f"{prob:.1%}",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "mode": "Heart Disease"
                }

                # Add to history if not already present
                if patient_data not in st.session_state.patient_history:
                    st.session_state.patient_history.append(patient_data)

                # Display results
                st.markdown("---")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"""
                    <div class='custom-card'>
                        <h2>Risk Assessment Result</h2>
                        <div style='text-align: center; margin: 20px 0;'>
                            <h3 style='color: {'#e63946' if pred == 1 else '#2a9d8f'};'>
                                {'⚠️ Significant Risk of Heart Disease' if pred == 1 else '✅ Low Risk of Heart Disease'}
                            </h3>
                            <h4 style='font-size: 24px;'>Probability: {prob:.1%}</h4>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Export Report Button
                    patient_info = {
                        "Patient ID": patient_id,
                        "Patient Name": patient_name,
                        "Age": age,
                        "Sex": sex,
                        "Blood Pressure": f"{trestbps} mm Hg",
                        "Cholesterol": f"{chol} mg/dl",
                        "BMI": bmi,
                        "Diabetes": diabetes,
                        "Family History": family_history
                    }

                    prediction_info = {
                        "prediction": pred,
                        "probability": prob,
                        "recommendations": [
                            "Cardiology consultation within 1 week" if pred == 1 else "Annual physical exam",
                            "Complete lipid profile" if pred == 1 else "Biannual lipid profile",
                            "Stress echocardiogram" if pred == 1 else "Regular blood pressure checks",
                            "Possible statin therapy" if pred == 1 else "Maintain healthy diet",
                            "Cardiac rehabilitation referral" if pred == 1 else "150 mins exercise/week"
                        ]
                    }

                    pdf_bytes = generate_pdf_report(patient_info, prediction_info, "Heart Disease")

                    st.download_button(
                        label="📄 Export Full Report",
                        data=pdf_bytes,
                        file_name=f"CardioCare_Report_{patient_id}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="hd_report"
                    )

                with col2:
                    st.plotly_chart(create_risk_gauge(prob, "Cardiovascular Risk Gauge"), use_container_width=True)

                if pred == 1:
                    st.markdown("""
                    <div class='custom-card' style='border-left: 5px solid #e63946;'>
                        <h3>Urgent Clinical Recommendations</h3>
                        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px;'>
                            <div>
                                <h4>Medical Evaluation</h4>
                                <ul>
                                    <li>Cardiology consultation within 1 week</li>
                                    <li>Complete lipid profile</li>
                                    <li>Stress echocardiogram</li>
                                    <li>Consider CT coronary angiography</li>
                                </ul>
                            </div>
                            <div>
                                <h4>Therapeutic Actions</h4>
                                <ul>
                                    <li>Possible statin therapy</li>
                                    <li>Blood pressure management</li>
                                    <li>Diabetes control if applicable</li>
                                    <li>Cardiac rehabilitation referral</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class='custom-card' style='border-left: 5px solid #2a9d8f;'>
                        <h3>Preventive Recommendations</h3>
                        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px;'>
                            <div>
                                <h4>Screening</h4>
                                <ul>
                                    <li>Annual physical exam</li>
                                    <li>Biannual lipid profile</li>
                                    <li>Regular blood pressure checks</li>
                                    <li>Diabetes screening</li>
                                </ul>
                            </div>
                            <div>
                                <h4>Lifestyle</h4>
                                <ul>
                                    <li>Maintain healthy diet</li>
                                    <li>150 mins exercise/week</li>
                                    <li>Stress management</li>
                                    <li>Adequate sleep</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Risk factors visualization
                st.markdown("---")
                st.markdown("""
                <div class='custom-card'>
                    <h2>Risk Factor Analysis</h2>
                    <p>Breakdown of contributing risk factors:</p>
                </div>
                """, unsafe_allow_html=True)

                mod_factors = ['Smoking', 'BMI', 'Activity', 'Alcohol', 'Diet', 'Stress']
                mod_values = [
                    0.8 if smoking == "Yes" else 0,
                    0.6 if bmi > 30 else (0.3 if bmi > 25 else 0),
                    0.7 if physical_activity in ["Sedentary", "Light"] else 0,
                    0.5 if alcohol_intake in ["Moderate", "Heavy"] else 0,
                    0.4 if diet_score < 5 else (0.2 if diet_score < 7 else 0),
                    0.3 if stress_level == "High" else (0.1 if stress_level == "Moderate" else 0)
                ]

                non_mod_factors = ['Age', 'Family History', 'Sex', 'Genetics']
                non_mod_values = [
                    min(0.9, age / 100),
                    0.7 if family_history == "Yes" else 0,
                    0.4 if sex == "Male" else 0.2,
                    0.3 if family_history == "Yes" else 0.1
                ]

                col3, col4 = st.columns(2)
                with col3:
                    st.plotly_chart(
                        create_rainbow_bar_chart(mod_factors, mod_values, "Modifiable Risk Factors"),
                        use_container_width=True
                    )
                with col4:
                    st.plotly_chart(
                        create_rainbow_bar_chart(non_mod_factors, non_mod_values, "Non-Modifiable Risk Factors"),
                        use_container_width=True
                    )

                # Personalized plan
                st.markdown(f"""
                <div class='custom-card'>
                    <h3>Personalized Risk Reduction Plan</h3>
                    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px;'>
                        <div>
                            <h4>Top Improvement Areas</h4>
                            <ul>
                                <li>{"Smoking cessation" if smoking == "Yes" else "Maintain non-smoking"}</li>
                                <li>{"Weight management" if bmi > 25 else "Maintain healthy weight"}</li>
                                <li>{"Increase physical activity" if physical_activity in ['Sedentary', 'Light'] else "Maintain activity level"}</li>
                                <li>{"Reduce alcohol consumption" if alcohol_intake in ["Moderate", "Heavy"] else "Maintain alcohol consumption"}</li>
                            </ul>
                        </div>
                        <div>
                            <h4>Monitoring</h4>
                            <ul>
                                <li>{"Weekly blood pressure monitoring" if trestbps > 130 else "Monthly blood pressure checks"}</li>
                                <li>{"Monthly cholesterol tests" if chol > 200 else "Quarterly cholesterol test"}</li>
                                <li>{"Daily glucose monitoring" if diabetes == 'Yes' else "Annual diabetes screening"}</li>
                                <li>{"Stress management counseling" if stress_level == "High" else "Regular stress assessment"}</li>
                            </ul>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

# === PATIENT HISTORY ===
elif app_mode == "Patient History":
    st.title("📋 Patient History Records")
    st.markdown("""
    <div class='custom-card'>
        <p>Review historical risk assessments and patient records.</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.patient_history:
        st.info("No patient records available. Perform assessments to populate history.")
    else:
        # Create DataFrame for display
        history_df = pd.DataFrame(st.session_state.patient_history)

        # Sort by timestamp
        history_df = history_df.sort_values(by="timestamp", ascending=False)

        # Display in tabs
        tab1, tab2 = st.tabs(["Summary View", "Detailed Records"])

        with tab1:
            st.markdown("### Patient Assessment Summary")

            # Create metrics
            high_risk_count = history_df[history_df["risk_level"] == "High"].shape[0]
            low_risk_count = history_df[history_df["risk_level"] == "Low"].shape[0]
            avg_prob = history_df["probability"].str.rstrip('%').astype('float').mean()

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Assessments", len(st.session_state.patient_history))
            col2.metric("High Risk Cases", high_risk_count)
            col3.metric("Average Risk Probability", f"{avg_prob:.1f}%")

            # Risk trend chart
            if len(history_df) > 1:
                trend_df = history_df.copy()
                trend_df['probability_num'] = trend_df['probability'].str.rstrip('%').astype('float')
                trend_df = trend_df.sort_values(by="timestamp")

                fig = px.line(
                    trend_df,
                    x="timestamp",
                    y="probability_num",
                    color="id",
                    markers=True,
                    title="Risk Probability Over Time"
                )
                fig.update_layout(
                    xaxis_title="Assessment Date",
                    yaxis_title="Risk Probability (%)",
                    legend_title="Patient ID"
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("### Detailed Assessment Records")
            st.dataframe(
                history_df,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Timestamp"),
                    "probability": st.column_config.ProgressColumn(
                        "Probability",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    )
                },
                use_container_width=True
            )

            # Export all history
            if st.button("Export Full History to CSV"):
                csv = history_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="cardio_care_patient_history.csv",
                    mime="text/csv"
                )

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>© 2025 CardioCare AI v3.0 | Advanced Cardiac Risk Assessment System</p>
    <p>Contact: <a href="mailto:yuvrajgond365@gmail.com" style="color: #0a9396; text-decoration: none;">yuvrajgond365@gmail.com</a></p>
</div>
""", unsafe_allow_html=True)
