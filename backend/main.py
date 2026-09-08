import traceback
import joblib
import numpy as np
import pandas as pd
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from explain import generate_llm_explanation, get_shap_explanation

app = FastAPI(title="Fraud Detection API")

# Load artifacts on startup
try:
    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    raise RuntimeError(f"Could not load model or scaler: {e}")


class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

class TransactionBatch(BaseModel):                         
    transactions: List[Transaction]


@app.post("/predict")
def predict_fraud(transaction: Transaction):
    try:
        data_dict = transaction.model_dump()
        df = pd.DataFrame([data_dict])

        # 1. Feature Engineering: log transform
        df['Amount_log'] = np.log1p(df['Amount'])

        # 2. Scale Time and Amount
        if hasattr(scaler, "feature_names_in_"):
            scaler_cols = list(scaler.feature_names_in_)
            df[scaler_cols] = scaler.transform(df[scaler_cols])
        else:
            df[['Time', 'Amount']] = scaler.transform(df[['Time', 'Amount']])

        # 3. Align DataFrame columns strictly with model feature requirements
        if hasattr(model, "feature_names_in_"):
            df = df[model.feature_names_in_]

        # 4. Model Prediction
        probability = float(model.predict_proba(df)[0, 1])
        prediction = int(model.predict(df)[0])

        top_features = []
        explanation = "Transaction approved based on risk threshold."

        if prediction == 1:
            top_features = get_shap_explanation(model, df)
            explanation = generate_llm_explanation(top_features, probability)

        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "message": (
                "Fraudulent transaction detected"
                if prediction == 1
                else "Transaction approved"
            ),
            "top_features": top_features,
            "explanation": explanation,
        }

    except Exception as e:
        print("\n--- TRACEBACK ERROR START ---")
        traceback.print_exc()
        print("--- TRACEBACK ERROR END ---\n")
        raise HTTPException(status_code=500, detail=str(e))

    

@app.post("/predict_batch")
def predict_fraud_batch(batch: TransactionBatch):
    try:
        records = [t.model_dump() for t in batch.transactions]
        df = pd.DataFrame(records)

        # 1. Feature Engineering: log transform
        df['Amount_log'] = np.log1p(df['Amount'])

        # 2. Scale Time and Amount
        if hasattr(scaler, "feature_names_in_"):
            scaler_cols = list(scaler.feature_names_in_)
            df[scaler_cols] = scaler.transform(df[scaler_cols])
        else:
            df[['Time', 'Amount']] = scaler.transform(df[['Time', 'Amount']])

        # 3. Align columns
        if hasattr(model, "feature_names_in_"):
            df = df[model.feature_names_in_]

        # 4. Batch prediction — one call for all rows
        probabilities = model.predict_proba(df)[:, 1]
        predictions = model.predict(df)

        results = []
        for i in range(len(df)):
            prediction = int(predictions[i])
            probability = float(probabilities[i])

            top_features = []
            explanation = "Transaction approved based on risk threshold."
            if prediction == 1:
                row_df = df.iloc[[i]]
                top_features = get_shap_explanation(model, row_df)
                explanation = generate_llm_explanation(top_features, probability)

            results.append({
                "prediction": prediction,
                "probability": round(probability, 4),
                "message": "Fraudulent transaction detected" if prediction == 1 else "Transaction approved",
                "top_features": top_features,
                "explanation": explanation,
            })

        return {"results": results}

    except Exception as e:
        print("\n--- TRACEBACK ERROR START ---")
        traceback.print_exc()
        print("--- TRACEBACK ERROR END ---\n")
        raise HTTPException(status_code=500, detail=str(e))