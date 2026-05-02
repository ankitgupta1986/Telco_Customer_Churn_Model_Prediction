# app.py - Streamlit Customer Churn Prediction App
import streamlit as st
import pandas as pd
import joblib

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ==============================
# Load Model and Scaler
# ==============================
@st.cache_resource
def load_model():
    model = joblib.load('Telco_Customer_Churn.pkl')
    scaler = joblib.load('Telco_Customer_Churn_scaler.pkl')
    return model, scaler

model, scaler = load_model()

# ==============================
# Title
# ==============================
st.title("📊 Customer Churn Prediction System")
st.markdown("Predict whether a telecom customer is likely to churn or not.")

# ==============================
# Sidebar Inputs
# ==============================
st.sidebar.header("Enter Customer Details")

# Numerical Inputs
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 1000.0)

# Categorical Inputs
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

online_security = st.sidebar.selectbox("Online Security", ["Yes", "No"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

# ==============================
# Prepare Input Data
# ==============================
input_data = {
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'Contract': contract,
    'InternetService': internet_service,
    'PaymentMethod': payment_method,
    'OnlineSecurity': online_security,
    'TechSupport': tech_support,
    'PaperlessBilling': paperless_billing
}

input_df = pd.DataFrame([input_data])

# ==============================
# Encoding Function
# ==============================
def preprocess_input(df):

    categorical_cols = df.select_dtypes(include=['object']).columns

    # One-hot encoding
    df = pd.get_dummies(df, columns=categorical_cols)

    # Required columns based on training
    required_columns = [
        'tenure',
        'MonthlyCharges',
        'TotalCharges',
        'Contract_One year',
        'Contract_Two year',
        'InternetService_Fiber optic',
        'InternetService_No',
        'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check',
        'PaymentMethod_Mailed check',
        'OnlineSecurity_Yes',
        'TechSupport_Yes',
        'PaperlessBilling_Yes'
    ]

    # Add missing columns
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0

    # Keep only required columns
    df = df[required_columns]

    return df

# ==============================
# Prediction Button
# ==============================
if st.button("Predict Churn"):

    processed_data = preprocess_input(input_df)

    # Scale numerical data
    processed_data_scaled = scaler.transform(processed_data)

    # Prediction
    prediction = model.predict(processed_data_scaled)[0]
    probability = model.predict_proba(processed_data_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to CHURN")
    else:
        st.success(f"✅ Customer is likely to STAY")

    st.write(f"### Churn Probability: {probability:.2%}")

