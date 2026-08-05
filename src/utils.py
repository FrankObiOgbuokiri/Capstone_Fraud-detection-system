import os
import joblib


def save_artifact(obj, file_path):
    """
    Saves a Python object (e.g., model or scaler) to disk using joblib.
    """
    # Create target directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    joblib.dump(obj, file_path)
    print(f"[INFO] Saved artifact to: {file_path}")


def load_artifact(file_path):
    """
    Loads a saved artifact (e.g., model or scaler) from disk.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Artifact not found at: {file_path}")
    
    obj = joblib.load(file_path)
    print(f"[INFO] Loaded artifact from: {file_path}")
    return obj