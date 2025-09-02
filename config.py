import os
APP_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(APP_DIR, "saved_models")
DATA_DIR = os.path.join(APP_DIR, "uploaded_data")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# In-memory registries
DATA_REGISTRY = {}
MODEL_REGISTRY = {}
