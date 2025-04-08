import streamlit as st
import joblib
import pandas as pd

# Load models
rf_model = joblib.load("rf_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")

# Model options
model_dict = {
    "Random Forest": rf_model,
    "XGBoost + SMOTE": xgb_model
}

# Define input features (must match model training)
feature_names = [
    'age', 'balance', 'duration', 'campaign'
] + [  # Add all one-hot encoded job columns used
    'job_unknown.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
    'job_management', 'job_retired', 'job_self-employed', 'job_services',
    'job_student', 'job_technician', 'job_unemployed'
]

# Streamlit UI
st.title("📊 Bank Marketing Subscription Predictor")
st.write("Select a model and enter details to predict whether the customer will subscribe to a term deposit.")

# Sidebar: Choose model
selected_model_name = st.sidebar.selectbox("Choose a model", list(model_dict.keys()))
model = model_dict[selected_model_name]

# Sidebar: User input for features
st.sidebar.subheader("Customer Info")

user_input = {}

user_input['age'] = st.sidebar.slider("Age", 18, 95, 30)
user_input['balance'] = st.sidebar.number_input("Account Balance (€)", value=1000)
user_input['duration'] = st.sidebar.slider("Last contact duration (seconds)", 0, 5000, 100)
user_input['campaign'] = st.sidebar.slider("Number of contacts in campaign", 1, 50, 3)

# One-hot encode 'job' selection
job_options = [
    'unknown', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
    'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed'
]
job = st.sidebar.selectbox("Job", job_options)

for job_cat in job_options:
    user_input[f'job_{job_cat}'] = 1 if job_cat == job else 0

# Create DataFrame for prediction
input_df = pd.DataFrame([user_input])

# Predict
if st.button("Predict"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    result = "✅ Subscribed" if pred == 1 else "❌ Not Subscribed"
    st.subheader("Prediction Result:")
    st.success(result)

    st.write(f"📈 Confidence: {prob:.2%}")
