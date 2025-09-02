
import os
import pandas as pd
from werkzeug.utils import secure_filename
import joblib
# from services.data_service import save_uploaded_file, load_dataset
import uuid
# from services.model_service import save_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


PROCESSED_DIR = "uploaded_data"
MODEL_DIR = "saved_models"
UPLOAD_FOLDER = "uploaded_data"


# data_services.py - Enhanced load_dataset function
def load_dataset(dataset_name):
    base_dir = "uploaded_data"
    file_path = os.path.join(base_dir, dataset_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset '{dataset_name}' not found at {file_path}")
    
    # Handle different file types
    if dataset_name.endswith('.csv'):
        return pd.read_csv(file_path)
    elif dataset_name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file_path)
    elif dataset_name.endswith('.json'):
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {dataset_name}")
    
# dataset upload services
def save_uploaded_file(file_storage):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    filename = secure_filename(file_storage.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file_storage.save(file_path)
    return filename


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


def preprocess_pipeline(dataset_name, config):
    df = load_dataset(dataset_name)
    if config.get("fillna", True):
        df = df.fillna(0)
    processed_name = f"processed_{dataset_name}"
    path = os.path.join(PROCESSED_DIR, processed_name)
    df.to_csv(path, index=False)
    return processed_name

# Train model services

def train_model(dataset_name, config):
    df = load_dataset(dataset_name)

    target = config.get("target")
    if not target or target not in df.columns:
        raise ValueError("Target column not found in dataset")

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    model_id = str(uuid.uuid4())
    save_model(model, model_id)

    return model_id, {"accuracy": acc}
