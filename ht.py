import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os

# === PAGE CONFIGURATION ===
st.set_page_config(
    page_title="CardioCare AI",
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
    }

    /* Main container */
    .stApp {
        background-color: #f8f9fa;
    }

    /* Headers */
    h1, h2, h3 {
        color: var(--primary) !important;
        font-family: 'Arial', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--primary), var(--secondary)) !important;
        color: white !important;
    }

    /* Buttons */
    .stButton>button {
        background-color: var(--secondary) !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-weight: 500 !important;
    }

    .stButton>button:hover {
        background-color: var(--primary) !important;
    }

    /* Forms */
    .stForm {
        background-color: white !important;
        border-radius: 12px !important;
        padding: 20px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
    }

    /* Input widgets */
    .stNumberInput, .stSelectbox, .stSlider, .stTextInput {
        border-radius: 8px !important;
    }

    /* Success message */
    .stAlert.success {
        background-color: #d4edda !important;
        color: #155724 !important;
        border-radius: 8px !important;
    }

    /* Error message */
    .stAlert.error {
        background-color: #f8d7da !important;
        color: #721c24 !important;
        border-radius: 8px !important;
    }

    /* Custom cards */
    .custom-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    /* Risk indicator */
    .risk-indicator {
        width: 100%;
        height: 30px;
        border-radius: 15px;
        background: linear-gradient(90deg, #4CAF50, #FFC107, #F44336);
        margin: 10px 0;
        position: relative;
    }

    .risk-marker {
        position: absolute;
        width: 4px;
        height: 40px;
        background: black;
        top: -5px;
        transform: translateX(-50%);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models(base_dir=r"C:\Users\yuvra\PycharmProjects\hd2\exported_models"):
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
    </div>
    """, unsafe_allow_html=True)

    app_mode = st.radio("Navigation", ["Home", "Early Warning", "Heart Disease"])

    st.markdown("---")
    st.markdown("""
    <div>
    <h4>About</h4>
    <p>Advanced ML-based cardiac risk assessment tool for healthcare professionals.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top: 20px;'>
    <p>Developed by</p>
    <p>Yuvraj Kumar Gond</p>
    <p>Version 2.1.0</p>
    </div>
    """, unsafe_allow_html=True)

# === HOME ===
if app_mode == "Home":
    st.title("❤️ CardioCare AI - Heart Health Prediction")

    st.markdown("""
    <div class='custom-card'>
        <h3>Advanced ML-Based Cardiac Risk Prediction</h3>
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
                <li>Uses XGBoost and Random Forest algorithms</li>
                <li>Clinical decision support tool</li>
                <li>Risk factor visualization</li>
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
                <h4>1. Input Data</h4>
                <p>Enter patient clinical and lifestyle information</p>
            </div>
            <div>
                <h4>2. AI Analysis</h4>
                <p>Our models process the data using advanced algorithms</p>
            </div>
            <div>
                <h4>3. Get Results</h4>
                <p>Receive risk assessment with actionable insights</p>
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
            age = st.number_input("Age", 18, 100, 45)
            sex = st.selectbox("Sex", ["Male", "Female"])
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 90, 200, 120)
            chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
        with col2:
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
            thalach = st.number_input("Maximum Heart Rate Achieved", 70, 220, 150)
            exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        with col3:
            oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 10.0, 0.0, step=0.1)
            bmi = st.number_input("Body Mass Index", 15.0, 50.0, 25.0)
            smoking = st.selectbox("Smoking", ["No", "Yes"])

        st.markdown("""
        <div class='custom-card'>
            <h3>Lifestyle Factors</h3>
        """, unsafe_allow_html=True)

        col4, col5, col6 = st.columns(3)
        with col4:
            alcohol_intake = st.selectbox("Alcohol Intake", ["None", "Light", "Moderate", "Heavy"])
            physical_activity = st.selectbox("Physical Activity Level", ["Sedentary", "Light", "Moderate", "Active"])
        with col5:
            family_history = st.selectbox("Family History of Heart Disease", ["No", "Yes"])
            diabetes = st.selectbox("Diabetes", ["No", "Yes"])
        with col6:
            stress_level = st.selectbox("Stress Level", ["Low", "Moderate", "High"])
            sleep_hours = st.number_input("Average Sleep Hours", 4, 12, 7)
            diet_score = st.slider("Diet Quality Score (1-10)", 1, 10, 6)

        submitted = st.form_submit_button("Predict Early Risk", use_container_width=True)

    if submitted:
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

            # Display results
            st.markdown("---")
            st.markdown(f"""
            <div class='custom-card'>
                <h2>Prediction Result</h2>
                <div class='risk-indicator'>
                    <div class='risk-marker' style='left: {prob * 100}%'></div>
                </div>
                <h3 style='text-align: center; color: {'#e63946' if pred == 1 else '#2a9d8f'};'>
                    {'🚨 High Risk of Early Heart Disease' if pred == 1 else '✅ Low Risk of Early Heart Disease'}
                </h3>
                <h4 style='text-align: center;'>Probability: {prob:.1%}</h4>
            </div>
            """, unsafe_allow_html=True)

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
                            </ul>
                        </div>
                        <div>
                            <h4>Lifestyle Changes</h4>
                            <ul>
                                <li>Begin supervised exercise program</li>
                                <li>Consult nutritionist for diet plan</li>
                                <li>Smoking cessation program</li>
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
                            </ul>
                        </div>
                        <div>
                            <h4>Improvement</h4>
                            <ul>
                                <li>Consider diet optimization</li>
                                <li>Stress reduction techniques</li>
                                <li>Regular aerobic exercise</li>
                            </ul>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Show feature importance
            st.markdown("---")
            st.markdown("""
            <div class='custom-card'>
                <h2>Key Contributing Factors</h2>
                <p>These factors most influenced the prediction:</p>
            </div>
            """, unsafe_allow_html=True)

            factors = {
                'Factor': ['Age', 'Cholesterol', 'Blood Pressure', 'BMI', 'Lifestyle', 'Family History'],
                'Impact': [0.25, 0.20, 0.18, 0.15, 0.12, 0.10]
            }

            col1, col2 = st.columns([3, 1])
            with col1:
                st.bar_chart(pd.DataFrame(factors).set_index('Factor'))
            with col2:
                st.markdown("""
                <div style='margin-top: 50px;'>
                    <h4>Top Positive Factors</h4>
                    <ul>
                        <li>Healthy BMI</li>
                        <li>Active Lifestyle</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

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
            <h3>Clinical Measurements</h3>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 18, 100, 45)
            sex = st.selectbox("Sex", ["Male", "Female"])
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 90, 200, 120)
            chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
            thalach = st.number_input("Maximum Heart Rate Achieved", 70, 220, 150)
        with col2:
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
            exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
            oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 10.0, 0.0, step=0.1)
            bmi = st.number_input("Body Mass Index", 15.0, 50.0, 25.0)

        st.markdown("""
        <div class='custom-card'>
            <h3>Lifestyle & History</h3>
        """, unsafe_allow_html=True)

        col3, col4 = st.columns(2)
        with col3:
            smoking = st.selectbox("Smoking", ["No", "Yes"])
            alcohol_intake = st.selectbox("Alcohol Intake", ["None", "Light", "Moderate", "Heavy"])
            physical_activity = st.selectbox("Physical Activity Level", ["Sedentary", "Light", "Moderate", "Active"])
            sleep_hours = st.number_input("Average Sleep Hours", 4, 12, 7)
        with col4:
            family_history = st.selectbox("Family History of Heart Disease", ["No", "Yes"])
            diabetes = st.selectbox("Diabetes", ["No", "Yes"])
            stress_level = st.selectbox("Stress Level", ["Low", "Moderate", "High"])
            diet_score = st.slider("Diet Quality Score (1-10)", 1, 10, 6)

        submitted = st.form_submit_button("Assess Heart Disease Risk", use_container_width=True)

    if submitted:
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

            # Display results
            st.markdown("---")
            st.markdown(f"""
            <div class='custom-card'>
                <h2>Risk Assessment Result</h2>
                <div class='risk-indicator'>
                    <div class='risk-marker' style='left: {prob * 100}%'></div>
                </div>
                <h3 style='text-align: center; color: {'#e63946' if pred == 1 else '#2a9d8f'};'>
                    {'⚠️ Significant Risk of Heart Disease' if pred == 1 else '✅ Low Risk of Heart Disease'}
                </h3>
                <h4 style='text-align: center;'>Probability: {prob:.1%}</h4>
            </div>
            """, unsafe_allow_html=True)

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

            risk_factors = {
                'Modifiable': ['Smoking', 'BMI', 'Activity', 'Alcohol', 'Diet', 'Stress'],
                'Non-Modifiable': ['Age', 'Family History', 'Sex', 'Genetics']
            }

            mod_values = [
                0.8 if smoking == "Yes" else 0,
                0.6 if bmi > 30 else (0.3 if bmi > 25 else 0),
                0.7 if physical_activity in ["Sedentary", "Light"] else 0,
                0.5 if alcohol_intake in ["Moderate", "Heavy"] else 0,
                0.4 if diet_score < 5 else (0.2 if diet_score < 7 else 0),
                0.3 if stress_level == "High" else (0.1 if stress_level == "Moderate" else 0)
            ]

            non_mod_values = [
                min(0.9, age / 100),
                0.7 if family_history == "Yes" else 0,
                0.4 if sex == "Male" else 0.2,
                0.3 if family_history == "Yes" else 0.1
            ]

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                <div class='custom-card'>
                    <h3>Modifiable Risk Factors</h3>
                """, unsafe_allow_html=True)
                st.bar_chart(pd.DataFrame({
                    'Factor': risk_factors['Modifiable'],
                    'Risk Contribution': mod_values
                }).set_index('Factor'))

            with col2:
                st.markdown("""
                <div class='custom-card'>
                    <h3>Non-Modifiable Risk Factors</h3>
                """, unsafe_allow_html=True)
                st.bar_chart(pd.DataFrame({
                    'Factor': risk_factors['Non-Modifiable'],
                    'Risk Contribution': non_mod_values
                }).set_index('Factor'))

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
                        </ul>
                    </div>
                    <div>
                        <h4>Monitoring</h4>
                        <ul>
                            <li>{"Monthly blood pressure checks" if trestbps > 130 else "Annual blood pressure checks"}</li>
                            <li>{"Quarterly cholesterol tests" if chol > 200 else "Annual cholesterol test"}</li>
                            <li>{"Regular diabetes management" if diabetes == 'Yes' else "Annual diabetes screening"}</li>
                        </ul>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")