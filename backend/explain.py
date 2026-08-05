import shap
import pandas as pd
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_shap_explanation(model, input_df):
    """Calculates SHAP values for the given input."""
    # Note: Use TreeExplainer for XGBoost/Random Forest, LinearExplainer for Logistic Regression
    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)
    
    # Extract top 3 features for this specific prediction
    feature_names = input_df.columns
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'importance': abs(shap_values.values[0])
    }).sort_values(by='importance', ascending=False)
    
    return shap_df['feature'].head(3).tolist()

def generate_llm_explanation(top_features, probability):
    """Converts SHAP features into a plain-English explanation via LLM."""
    prompt = f"""
    A fraud detection model flagged a transaction with a {(probability * 100):.1f}% probability of being fraudulent.
    The top features contributing to this decision were: {', '.join(top_features)}.
    
    Write exactly one short, plain-English sentence explaining this to a non-technical fraud analyst.
    Do not mention SHAP or the math. Keep it actionable.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Explanation unavailable at this time due to LLM timeout."