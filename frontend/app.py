import io
import pandas as pd
import requests
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Credit Card Fraud Detection System")
st.markdown(
    "Real-time and batch transaction inference using ML prediction models, "
    "SHAP feature importance, and LLM-powered explanations."
)

st.divider()

API_URL = "http://127.0.0.1:8000/predict"
BATCH_API_URL = "http://127.0.0.1:8000/predict_batch"

# Navigation Mode Selection
mode = st.radio(
    "Select Operating Mode:",
    ["Manual Entry Mode", "CSV Upload Mode"],
    horizontal=True,
)

st.divider()

# ==========================================
# MODE 1: MANUAL ENTRY MODE
# ==========================================
if mode == "Manual Entry Mode":
    st.subheader("📝 Manual Single Transaction Entry")

    # Sidebar inputs for Manual Mode
    st.sidebar.header("🔧 Transaction Inputs")
    time_input = st.sidebar.number_input(
        "Time (seconds elapsed)", min_value=0.0, value=100.0, step=1.0
    )
    amount_input = st.sidebar.number_input(
        "Transaction Amount ($)", min_value=0.0, value=250.0, step=5.0
    )

    with st.sidebar.expander("V1 - V28 Features (PCA Components)", expanded=False):
        v_inputs = {}
        for i in range(1, 29):
            v_inputs[f"V{i}"] = st.number_input(
                f"V{i}", value=0.0, step=0.1, format="%.4f"
            )

    payload = {
        "Time": time_input,
        "Amount": amount_input,
        **v_inputs,
    }

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.write("### Input Summary")
        st.dataframe(
            pd.DataFrame([{"Time": time_input, "Amount ($)": amount_input}]),
            use_container_width=True,
        )
        predict_btn = st.button(
            "🔍 Run Assessment", type="primary", use_container_width=True
        )

    with col_right:
        st.write("### Assessment Results")
        if predict_btn:
            with st.spinner("Analyzing transaction..."):
                try:
                    response = requests.post(API_URL, json=payload, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        prediction = data.get("prediction", 0)
                        probability = data.get("probability", 0.0)
                        top_features = data.get("top_features", [])
                        explanation = data.get("explanation", "")

                        if prediction == 1:
                            st.error(f"⚠️ **FRAUD DETECTED** (Probability: {probability:.2%})")
                        else:
                            st.success(f"✅ **APPROVED / LEGITIMATE** (Probability: {probability:.2%})")

                        col_m1, col_m2 = st.columns(2)
                        col_m1.metric("Risk Score", f"{probability:.2%}")
                        col_m2.metric("Status", "Fraud" if prediction == 1 else "Legitimate")

                        if top_features:
                            st.markdown("#### 🔑 Key Risk Indicators (SHAP)")
                            for idx, feat in enumerate(top_features, 1):
                                st.markdown(f"**{idx}.** `{feat}`")

                        if explanation:
                            st.markdown("#### 🤖 Natural Language Explanation")
                            st.info(explanation)
                    else:
                        st.error(f"API Error ({response.status_code}): {response.text}")
                except Exception as e:
                    st.error(f"Connection Error: {str(e)}")


# ==========================================
# MODE 2: CSV UPLOAD MODE (BATCH PROCESSING)
# ==========================================
else:
    st.subheader("📁 CSV Batch Processing Mode")
    st.write(
        "Upload a CSV file containing transaction records (`Time`, `Amount`, `V1`–`V28`)."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(batch_df)}** transactions from CSV.")

            required_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
            missing_cols = [col for col in required_cols if col not in batch_df.columns]

            if missing_cols:
                st.error(f"Missing required columns in CSV: {missing_cols}")
            else:
                if st.button("🚀 Process Batch Predictions", type="primary"):
                    with st.spinner(f"Processing {len(batch_df)} transactions..."):
                        try:
                            transactions = batch_df[required_cols].to_dict(orient="records")
                            res = requests.post(
                                BATCH_API_URL,
                                json={"transactions": transactions},
                                timeout=60,
                            )

                            if res.status_code == 200:
                                batch_predictions = res.json()["results"]
                                results = []
                                for idx, (row, pred) in enumerate(
                                    zip(batch_df.itertuples(), batch_predictions)
                                ):
                                    results.append({
                                        "Row": idx + 1,
                                        "Time": row.Time,
                                        "Amount": row.Amount,
                                        "Prediction": pred["prediction"],
                                        "Probability": pred["probability"],
                                        "Top_Features": ", ".join(pred["top_features"]),
                                        "Explanation": pred["explanation"],
                                    })
                                st.session_state["batch_results_df"] = pd.DataFrame(results)
                            else:
                                st.error(f"API Error ({res.status_code}): {res.text}")
                        except Exception as err:
                            st.error(f"Connection Error: {str(err)}")

            # Display results and download button if available in session_state
            if "batch_results_df" in st.session_state:
                res_df = st.session_state["batch_results_df"]

                st.markdown("### 📊 Batch Results Summary")
                fraud_count = (res_df["Prediction"] == 1).sum()

                col_metric, col_download = st.columns([2, 1])

                with col_metric:
                    st.metric(
                        "Total Flagged Fraud Transactions",
                        f"{fraud_count} / {len(res_df)}"
                    )

                # --- CSV Export Button ---
                csv_data = res_df.to_csv(index=False).encode("utf-8")

                with col_download:
                    st.write("")  # Spacing alignment
                    st.download_button(
                        label="📥 Export Results to CSV",
                        data=csv_data,
                        file_name="fraud_detection_batch_results.csv",
                        mime="text/csv",
                        type="secondary",
                        use_container_width=True,
                    )

                # Highlight Fraud Rows in Dataframe
                def highlight_fraud(val):
                    if val == 1:
                        return "background-color: #ffcdd2; color: #b71c1c; font-weight: bold;"
                    return ""

                styled_df = res_df.style.map(
                    highlight_fraud, subset=["Prediction"]
                )
                st.dataframe(styled_df, use_container_width=True)

                # Expandable Explanation Panels for Flagged Transactions
                st.markdown("### 🔍 Expandable Explanation Panels")
                fraud_rows = res_df[res_df["Prediction"] == 1]

                if fraud_rows.empty:
                    st.success("No fraudulent transactions detected in this batch.")
                else:
                    for _, f_row in fraud_rows.iterrows():
                        with st.expander(
                            f"⚠️ Row {int(f_row['Row'])}: ${f_row['Amount']:.2f} — Risk: {f_row['Probability']:.2%}"
                        ):
                            st.write(f"**Top SHAP Risk Features:** `{f_row['Top_Features']}`")
                            st.info(f"**LLM Explanation:** {f_row['Explanation']}")

        except Exception as e:
            st.error(f"Failed to parse uploaded CSV: {str(e)}")