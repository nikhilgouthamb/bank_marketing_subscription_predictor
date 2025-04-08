import streamlit as st
import pandas as pd
import joblib

# Load models
rf_model = joblib.load("rf_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")

# Load feature order used during training
feature_order = joblib.load("feature_columns.pkl")

# Model options
model_dict = {
    "Random Forest": rf_model,
    "XGBoost + SMOTE": xgb_model
}

# App title
st.title("📊 Bank Marketing Subscription Predictor")
st.write("Choose a model and enter customer details to predict if they’ll subscribe to a term deposit.")

# Sidebar: Model selection
selected_model_name = st.sidebar.selectbox("Choose a model", list(model_dict.keys()))
model = model_dict[selected_model_name]

# Sidebar: Customer inputs
st.sidebar.subheader("Customer Info")

user_input = {}

# Numeric inputs
user_input['age'] = st.sidebar.slider("Age", 18, 95, 30)
user_input['balance'] = st.sidebar.number_input("Account Balance (€)", value=1000)
user_input['duration'] = st.sidebar.slider("Last contact duration (seconds)", 0, 5000, 100)
user_input['campaign'] = st.sidebar.slider("Number of contacts in campaign", 1, 50, 3)

# One-hot encode 'job' selection
job_options = [
    'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
    'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed'
]
job_selected = st.sidebar.selectbox("Job", job_options)

# Add one-hot encoded job columns
for job in job_options:
    user_input[f'job_{job}'] = 1 if job == job_selected else 0

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# 🧠 Ensure input_df has same feature order as during training
input_df = input_df.reindex(columns=feature_order, fill_value=0)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    label = "✅ Subscribed" if prediction == 1 else "❌ Not Subscribed"
    st.subheader("Prediction:")
    st.success(label)

    st.write(f"📈 Confidence: **{probability:.2%}**")
