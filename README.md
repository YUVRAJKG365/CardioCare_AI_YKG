# 🫀 CardioCare AI – Heart Disease Early Detection & Prediction System

**CardioCare_AI_YKG** is an advanced, AI-powered healthcare application developed using **Streamlit**. It performs **early detection** and **full-scale heart disease prediction** using clinical and lifestyle indicators. The system uses machine learning to assess cardiovascular risk and provide actionable insights and visualizations for both healthcare professionals and individuals.

> ⚠️ **Note**: This is an **individual project** developed by **Yuvraj Kumar Gond** for academic and demonstration purposes. It is **not open source** and is not intended for commercial or diagnostic use.

---

## 🧠 Key Features

- 🔍 Early heart disease warning system
- ❤️ Full heart disease risk prediction
- 📈 Dynamic risk probability visualization
- 📊 Feature importance and contributing factor analysis
- 🧑‍⚕️ Personalized medical and lifestyle recommendations
- 🧬 Based on real-world clinical datasets
- 🎨 Custom-designed medical-themed UI using Streamlit

---

## 🧪 Machine Learning Models

- `XGBoostClassifier`: For early warning prediction
- `RandomForestClassifier` or `XGBoost`: For full heart disease detection
- `StandardScaler`: For input normalization
- Pre-trained models and scalers stored in `exported_models/`


yaml
Copy
Edit

---

## 🧾 Input Features

### Early Detection Model

| Category      | Features |
|---------------|----------|
| Clinical      | Age, Sex, Blood Pressure, Cholesterol, Fasting Blood Sugar |
| Lifestyle     | BMI, Smoking, Alcohol Intake, Physical Activity, Stress, Sleep Hours, Diet Score |
| History       | Family History, Diabetes |

### Full Prediction Model

Includes all above, plus:
- `thalach`: Max heart rate
- `exang`: Exercise-induced angina
- `oldpeak`: ST depression
- Additional derived indicators

---

## ⚙️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/YUVRAJKG365/CardioCare_AI_YKG.git
cd CardioCare_AI_YKG
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Launch the Streamlit App
bash
Copy
Edit
streamlit run hp.py
🧭 App Navigation
Home – Project overview and system explanation

Early Warning – Predict early cardiac risk based on lifestyle & basic health metrics

Heart Disease – Perform full cardiovascular risk assessment using clinical inputs

🔍 Example Use Cases
🏥 Hospitals and clinics for risk screening

📊 Health dashboards for telehealth apps

🧑‍💼 Individuals managing long-term cardiac wellness

📌 Future Enhancements
🌐 Web deployment (Streamlit Cloud / Hugging Face Spaces)

📲 Android or iOS integration

🫁 Real-time health monitoring integration (wearables)

🧠 LLM-based health assistant/chatbot

👨‍💻 Developer Info
🧑 Yuvraj Kumar Gond
🔗 GitHub: @YUVRAJKG365
📅 Version: 2.1.0
📍 Location: India
📘 Status: Individual academic project — not open source

🛑 Disclaimer
This software is intended for academic and educational purposes only. It should not be used for actual medical diagnosis or treatment decisions without consultation with a licensed medical professional.

less
Copy
Edit

Let me know if you also want:
- A `requirements.txt` generated from your code
- A license section (like `All Rights Reserved`)
- Deployment support (Streamlit Cloud, Hugging Face, etc.)
