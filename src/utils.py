import os
import joblib

from datetime import datetime


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


def save_versioned_artifact(obj, base_dir, name, metadata=None):
    """
    Saves an artifact with a timestamped filename for versioning, alongside
    a metadata log entry. Does not replace save_artifact — use this when you
    want a historical record of trained models; use save_artifact directly
    for the "current production" file the API loads.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_path = os.path.join(base_dir, f"{name}_{timestamp}.pkl")
    save_artifact(obj, versioned_path)

    if metadata:
        log_path = os.path.join(base_dir, "model_version_log.txt")
        with open(log_path, "a") as f:
            f.write(f"{timestamp} | {name} | {metadata}\n")
        print(f"[INFO] Logged version metadata to: {log_path}")

    return versioned_path