# Credit Card Fraud Detection System
A full-stack machine learning system for detecting fraudulent credit card transactions in real time — a tuned, cross-validated XGBoost classifier served through a FastAPI backend and a Streamlit dashboard, with SHAP + LLM-generated explanations for every flagged transaction.
Full methodology, model comparison, and results are documented in docs/report_obioma.docx and notebooks/.
# Overview
Credit card fraud is a classic extreme-imbalance classification problem — in the dataset this project is built on, fraudulent transactions make up only 0.17% of all records (492 out of 284,807). A naive model that predicts "legitimate" every time would score 99.8% accuracy while catching zero fraud, so this project is built around PR-AUC as the primary evaluation metric, and around class-weighting, resampling comparison, cross-validation, and threshold analysis as the core methodology, not just training a model and calling it done.
The end result is a working system, not just a notebook: a trained model served through a real API, a dashboard non-technical users can interact with, and an explainability layer that makes the model's decisions auditable rather than opaque.

# Features
•	Tuned, cross-validated XGBoost classifier — hyperparameters selected via RandomizedSearchCV with 5-fold stratified cross-validation, optimizing for PR-AUC.
•	Evidence-based threshold selection — the classification threshold was swept and analyzed rather than left at an unexamined default; see Model Performance for the finding.
•	Real-time single-transaction prediction via POST /predict.
•	Batch prediction via POST /predict_batch, processing an entire CSV upload in one API call rather than one request per row.
•	SHAP explainability — every flagged transaction returns its top contributing features, computed via TreeExplainer.
•	LLM-generated explanations — SHAP output is translated into a natural-language summary via the Groq API, with a graceful fallback to a template-based explanation if the LLM call fails.
•	Streamlit dashboard with two modes: manual single-transaction entry, and CSV batch upload with per-row results, fraud highlighting, and expandable explanation panels.
•	Comprehensive model comparison — 5 models (Logistic Regression, Random Forest, XGBoost, CatBoost, LightGBM) evaluated across 3 imbalance-handling strategies (SMOTE, Random Under-Sampling, class-weighting) before final model selection.

# Architecture
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│  Streamlit UI    │  HTTP   │   FastAPI Backend │         │   Groq LLM API   │
│  (frontend/)      │ ──────▶ │   (backend/)       │ ──────▶ │   (explanations) │
│                  │         │                  │         └─────────────────┘
│  - Manual entry  │         │  - /predict       │
│  - CSV batch     │         │  - /predict_batch │         ┌─────────────────┐
└─────────────────┘         │  - SHAP layer     │ ──────▶ │  fraud_model.pkl │
                             │  - LLM layer      │         │  scaler.pkl      │
                             └──────────────────┘         └─────────────────┘
The model and scaler are trained and exported in notebooks/, then loaded by the FastAPI backend at startup. The backend is the single source of truth for predictions — the Streamlit frontend never touches the model directly, it only calls the API.

# Project Structure
Capstone_Fraud-detection-system/
├── backend/
│   ├── main.py              # FastAPI app: /predict, /predict_batch
│   ├── explain.py           # SHAP + LLM (Groq) explanation logic
│   ├── fraud_model.pkl      # Trained model (not tracked in git)
│   ├── scaler.pkl           # Fitted StandardScaler (not tracked in git)
│   ├── .env                 # GROQ_API_KEY (not tracked in git)
│   └── .env.example         # Template for required environment variables
├── frontend/
│   └── app.py                # Streamlit dashboard
├── notebooks/
│   ├── preprocessing.ipynb   # EDA, resampling & model comparison
│   └── fraud_training_*.ipynb # Full end-to-end training notebook
├── src/
│   ├── data_prep.py          # Loading, EDA plots, split/scale
│   ├── train.py              # Training, tuning, model comparison
│   ├── evaluate.py           # Metrics, cross-validation, threshold tuning
│   ├── sampling.py           # SMOTE/RUS, class-weight ratio computation
│   └── utils.py              # Model/artifact save & load
├── data/                     # Raw dataset 
├── docs/                     # Project report
├── test_batch_sample.csv     # Small known-answer file for testing the batch endpoint
├── requirements.txt
├── .gitignore
└── README.md

# Setup
python -m venv venv
venv\Scripts\activate          # Windows (or: source venv/bin/activate)
pip install -r requirements.txt
Copy the environment template and add your Groq API key:
cd backend
copy .env.example .env         # or: cp .env.example .env
# edit .env: GROQ_API_KEY=your_actual_key_here

# Dataset
Uses the Credit Card Fraud Detection dataset from Kaggle (284,807 transactions, 492 fraudulent). Included in this repo at data/creditcard.csv via Git LFS — install Git LFS before cloning (git lfs install), or run git lfs pull after cloning if the file appears as a small pointer file rather than the actual dataset. For a quick check without pulling the full file, use test_batch_sample.csv in the dashboard's CSV Upload Mode.
Running the Project
Retrain the model: run notebooks/preprocessing.ipynb then notebooks/fraud_training_Obioma_Ogbuokiri.ipynb.
Run the live system (two terminals):
# Terminal 1
cd backend
uvicorn main:app --reload

# Terminal 2
cd frontend
streamlit run app.py
The backend must be running before the frontend can return predictions.

# Model Performance
XGBoost (class-weighted, tuned): PR-AUC 0.8539 ± 0.0191 (5-fold CV), 0.8844 (test); precision/recall 0.89/0.84 on the fraud class. Full comparison against 4 other models and 3 resampling strategies is in preprocessing.ipynb.

# API
•	POST /predict — single transaction (Time, Amount, V1–V28)
•	POST /predict_batch — {"transactions": [...]}, batched predictions
Both return prediction, probability, top_features (SHAP), and explanation (LLM-generated).

# Tech Stack
scikit-learn · XGBoost · CatBoost · LightGBM · SHAP · FastAPI · Streamlit · Groq API

# Acknowledgements
Developed as a capstone project, with structured review and mentorship shaping key fixes documented in docs/report_obioma.docx.

