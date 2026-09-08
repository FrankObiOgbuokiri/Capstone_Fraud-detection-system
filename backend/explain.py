import os
from pathlib import Path
import pandas as pd
import shap
from dotenv import load_dotenv
from groq import Groq

# Dynamically locate and load .env file
env_path = Path(__file__).resolve().parent / ".env"
if not env_path.exists():
    env_path = Path(__file__).resolve().parent.parent / ".env"

load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key else None


def get_shap_explanation(model, input_df):
    """Calculates top SHAP feature importances for a single transaction input."""
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model

    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer(input_df)
    except Exception:
        explainer = shap.Explainer(estimator, input_df)
        shap_values = explainer(input_df)

    vals = shap_values.values

    # Handle 3D array output: (samples, features, classes)
    if vals.ndim == 3:
        vals = vals[:, :, 1]

    importance = abs(vals[0]) if vals.ndim > 1 else abs(vals)

    shap_df = pd.DataFrame({
        'feature': input_df.columns,
        'importance': importance
    }).sort_values(by='importance', ascending=False)

    return shap_df['feature'].head(3).tolist()


def generate_llm_explanation(top_features, probability):
    """Generates an LLM summary of the key features driving the fraud prediction."""
    global client
    if client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY is missing! Set it in your .env file.")
        client = Groq(api_key=key)

    candidate_models = [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "groq/compound-mini",
    ]

    for model_name in candidate_models:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a fraud analyst explaining model outputs in clear, "
                            "concise language."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Explain concisely why a transaction was flagged as fraud with "
                            f"a probability of {probability:.2%}. Key contributing "
                            f"features: {', '.join(top_features)}"
                        ),
                    },
                ],
                temperature=0.2,
                max_completion_tokens=400,
            )
            return completion.choices[0].message.content
        except Exception:
            continue

    # Graceful fallback if LLM endpoint is unreachable
    return (
        f"Transaction flagged with {probability:.2%} fraud probability based on key "
        f"risk indicators: {', '.join(top_features)}."
    )