import streamlit as st
import joblib
import pandas as pd
import numpy as np
from scorecardpy import woebin_ply, scorecard, scorecard_ply

# --- Load all models & tools ---
xgb_model = joblib.load("xgb_model.pkl")
logreg_model = joblib.load("logistic_model.pkl")
lr_woe_model = joblib.load("logistic_model_woe.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("scaler_features.pkl")
bins = joblib.load("scorecard_bins.pkl")

# --- UI Config ---
st.set_page_config(page_title="Credit Risk Dashboard", layout="centered")
st.title("💳 Credit Risk Prediction Dashboard")

# --- Model selection ---
model_option = st.selectbox("Choose a Model", (
    "XGBoost",
    "Logistic Regression",
    "Credit Scorecard (WOE)"
))

st.subheader("📋 Enter Applicant Details")

# --- User Input Form ---
input_data = {}

# Gender
input_data["gender"] = st.radio("Gender", ("Male", "Female"))

# Numeric Features
input_data["credit_score"] = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
input_data["credit_limit_used(%)"] = st.slider("Credit Limit Used (%)", 0.0, 100.0, 35.0)
input_data["no_of_days_employed"] = st.number_input("Days Employed", min_value=0, value=1000)

# Occupation one-hot encoding
occupation_options = [
    'HR staff', 'IT staff', 'Low-skill Laborers', 'Private service staff',
    'Sales', 'Secretaries', 'High skill tech staff'
]
selected_occupation = st.selectbox("Occupation Type", occupation_options)

# Create one-hot encoding manually
for occ in occupation_options:
    col_name = f"occupation_{occ}"
    input_data[col_name] = 1 if selected_occupation == occ else 0

# Convert gender to numeric
input_data["gender"] = 1 if input_data["gender"] == "Female" else 0

# Create input dataframe with feature alignment
user_df = pd.DataFrame([input_data])
user_df = user_df.reindex(columns=features, fill_value=0)

# --- Predict Button ---
if st.button("🔍 Predict"):
    st.divider()

    if model_option == "Credit Scorecard (WOE)":
        # Add dummy target for scorecardpy compatibility
        df_orig = user_df.copy()
        df_orig["credit_card_default"] = 0

        # Apply WOE binning
        woe_df = woebin_ply(df_orig, bins)
        woe_df.drop(columns=["credit_card_default"], inplace=True)

        # Predict
        pred_prob = lr_woe_model.predict_proba(woe_df)[:, 1][0]
        score = scorecard_ply(df_orig, scorecard(bins, lr_woe_model, woe_df.columns))
        final_score = int(score["score"].values[0])

        # Interpret credit score
        if final_score >= 750:
            risk_level = "Excellent"
            color = "🟢"
            msg = "Very low chance of default. Creditworthy!"
        elif final_score >= 650:
            risk_level = "Good"
            color = "🟡"
            msg = "Fairly safe, but monitor closely."
        elif final_score >= 550:
            risk_level = "Average"
            color = "🟠"
            msg = "Moderate risk. Further review recommended."
        else:
            risk_level = "Poor"
            color = "🔴"
            msg = "High risk. Proceed with caution."

        # Output
        st.subheader("📊 Prediction Result")
        st.markdown(f"**💳 Credit Score:** `{final_score}`")
        st.markdown(f"**{color} Risk Level:** `{risk_level}`")
        st.markdown(f"**📉 Probability of Default:** `{pred_prob:.2%}`")
        st.info(msg)

    else:
        # Scale numeric data
        X_scaled = scaler.transform(user_df)

        # Choose model
        model = xgb_model if model_option == "XGBoost" else logreg_model
        pred_class = model.predict(X_scaled)[0]
        pred_prob = model.predict_proba(X_scaled)[0][1]

        # Output
        st.subheader("📊 Prediction Result")

        if pred_class == 1:
            st.error("🚨 High Risk of Default")
            st.markdown(f"**🔻 Probability of Default:** `{pred_prob:.2%}`")
            st.markdown("💡 Consider reviewing applicant's credit profile carefully.")
        else:
            st.success("✅ Low Risk of Default")
            st.markdown(f"**🔹 Probability of Default:** `{pred_prob:.2%}`")
            st.markdown("👍 Applicant appears financially reliable.")
