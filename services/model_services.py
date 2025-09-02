import os
import joblib

MODEL_DIR = "saved_models"

def save_model(model, model_id):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"{model_id}.pkl")
    joblib.dump(model, path)
    return path

def load_model(model_id):
    path = os.path.join(MODEL_DIR, f"{model_id}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model {model_id} not found")
    return joblib.load(path)

def list_models():
    if not os.path.exists(MODEL_DIR):
        return []
    return [f.replace(".pkl", "") for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]