import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
model = joblib.load("XGBClassifierr_model.pkl")

st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("💳 Online Payment Fraud Detection System")
st.write("Enter transaction details below:")

# ---- INPUT FIELDS ----

step = st.number_input("Step (Time Step)", min_value=0)

type_txn = st.selectbox(
    "Transaction Type",
    ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"]
)

amount = st.number_input("Transaction Amount", min_value=0.0)

oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0)

oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0)

# ---- PREDICTION ----

if st.button("Predict Fraud"):

    input_df = pd.DataFrame({
        "step": [step],
        "type": [type_txn],
        "amount": [amount],
        "oldbalanceOrg": [oldbalanceOrg],
        "newbalanceOrig": [newbalanceOrig],
        "oldbalanceDest": [oldbalanceDest],
        "newbalanceDest": [newbalanceDest]
    })

    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected!")
        st.write(f"Fraud Probability: {probability:.2f}")
    else:
        st.success("✅ Legitimate Transaction")
        st.write(f"Fraud Probability: {probability:.2f}")