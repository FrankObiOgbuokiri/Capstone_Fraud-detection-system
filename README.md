# 3MTT Mentorship: Fraud Detection Project

## Project Overview
This repository contains the **Fraud Detection** project developed under the **3 Million Technical Talent (3MTT) Mentorship Program**. Financial fraud costs organizations and individuals billions of dollars annually. This project leverages machine learning techniques to identify fraudulent transactions, minimizing financial losses while ensuring a seamless experience for legitimate users.


## Objectives
*   **Identify Patterns:** Analyze transactional data to discover anomalies and patterns indicative of fraud.
*   **Handle Data Imbalance:** Implement advanced sampling techniques (like class weight adjustment) to tackle highly imbalanced fraud datasets.
*   **Build Predictive Models:** Train, evaluate, and fine-tune machine learning models to maximize detection rates (Recall) while minimizing false alarms (Precision).
*   **Provide Actionable Insights:** Highlight the key features and risk factors that contribute most to fraudulent activity.

## Tech Stack & Tools
*   **Language:** Python
*   **Libraries:** 
    *   *Data Manipulation:* Pandas, NumPy
    *   *Data Visualization:* Matplotlib, Seaborn
    *   *Machine Learning:* Scikit-Learn, XGBoost
    
*   **Environment:** Python/Jupyter Notebook 

## Dataset Description
The dataset used in this project models typical financial transaction behaviors. Key attributes often include:
*   `Transaction_ID`: Unique identifier for each transaction.
*   `Amount`: The monetary value of the transaction.
*   `Time`: Timestamp or relative time sequence of the transaction.
*   `Features (V1, V2, ...)`: Anonymized features resulting from PCA transformation (common in financial datasets for privacy).
*   `Class`: The target variable (`0` for Legitimate, `1` for Fraudulent).


## Project Structure
```text
fraud_detection_system/
├── .gitignore
├── README.md
├── requirements.txt
├── src/                          # SHARED: Built via collaborative PRs
│   ├── __init__.py
│   ├── data_prep.py
│   ├── sampling.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/                    # INDIVIDUAL: Your own workspace
│   ├── .gitkeep
│   ├── fraud_training_obi.ipynb
│   
├── backend/                      # SHARED APPLICATION LAYER
│   ├── main.py
│   ├── explain.py
│   ├── fraud_model.pkl           # Saved production model
│   └── scaler.pkl                # Saved production scaler
├── frontend/                     # SHARED FRONTEND LAYER
│   └── app.py
└── docs/                         # INDIVIDUAL: Technical reports
    ├── report_obi.md
    
