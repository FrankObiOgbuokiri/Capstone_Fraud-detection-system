from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from explain import get_shap_explanation, generate_llm_explanation

app = FastAPI(title="Fraud Detection API")

# Load artifacts on startup
try:
    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    raise RuntimeError(f"Could not load model or scaler: {e}")

# Assuming V1-V28 + Time + Amount
class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    # ... Include all 28 V-features here. Truncated for brevity.

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    try:
        # Convert to DataFrame
        data_dict = transaction.model_dump()
        df = pd.DataFrame([data_dict])
        
        # Preprocess Time and Amount
        df[['Time', 'Amount']] = scaler.transform(df[['Time', 'Amount']])
        
        # Predict
        probability = float(model.predict_proba(df)[0, 1])
        prediction = int(model.predict(df)[0])
        
        # Explain
        top_features = []
        explanation = "Legitimate transaction."
        
        if prediction == 1:
            top_features = get_shap_explanation(model, df)
            explanation = generate_llm_explanation(top_features, probability)

        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "message": "Fraudulent transaction detected" if prediction == 1 else "Transaction approved",
            "top_features": top_features,
            "explanation": explanation
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
