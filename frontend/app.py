import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8007/predict"

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("Credit Card Fraud Detection")

tab1, tab2 = st.tabs(["Manual Entry", "CSV Batch Upload"])

with tab1:
    st.header("Manual Transaction Entry")
    with st.form("manual_entry"):
        col1, col2 = st.columns(2)
        time_val = col1.number_input("Time", value=0.0)
        amount_val = col2.number_input("Amount", value=150.00)
        
        st.write("Anonymized Features (V1 - V4 for demo)")
        v1 = col1.number_input("V1", value=0.0)
        v2 = col2.number_input("V2", value=0.0)
        v3 = col1.number_input("V3", value=0.0)
        v4 = col2.number_input("V4", value=0.0)
        # Add remaining V-features in reality
        
        submitted = st.form_submit_button("Analyze Transaction")
        
        if submitted:
            payload = {"Time": time_val, "Amount": amount_val, "V1": v1, "V2": v2, "V3": v3, "V4": v4}
            # Fill remaining features with 0.0 for this demo snippet to match Pydantic schema
            
            with st.spinner("Analyzing..."):
                resp = requests.post(API_URL, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    if data["prediction"] == 1:
                        st.error(f"🚨 FRAUD DETECTED (Probability: {data['probability']*100:.2f}%)")
                        st.warning(f"**Explanation:** {data['explanation']}")
                    else:
                        st.success(f"✅ Transaction Approved (Probability of fraud: {data['probability']*100:.2f}%)")
                else:
                    st.error("API Error")

with tab2:
    st.header("Batch Upload")
    uploaded_file = st.file_uploader("Upload CSV of transactions", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        results = []
        
        progress_bar = st.progress(0)
        for i, row in df.iterrows():
            payload = row.to_dict()
            resp = requests.post(API_URL, json=payload)
            if resp.status_code == 200:
                results.append(resp.json())
            progress_bar.progress((i + 1) / len(df))
            
        st.write("### Flagged Transactions")
        flagged = [r for r in results if r["prediction"] == 1]
        
        if not flagged:
            st.success("No fraud detected in this batch.")
        else:
            st.error(f"{len(flagged)} suspicious transactions found.")
            for f in flagged:
                with st.expander(f"Fraud Probability: {f['probability']*100:.2f}%"):
                    st.write(f"**Top Features:** {', '.join(f['top_features'])}")
                    st.write(f"**Analyst Summary:** {f['explanation']}")
