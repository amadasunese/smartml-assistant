import uuid
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from services.data_services import load_dataset
from services.model_services import save_model

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