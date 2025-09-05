import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import uuid


from services.data_services import load_dataset
from services.model_services import load_model, list_models, MODEL_DIR

# MODEL_DIR = "models"
EVAL_PLOTS_DIR = "static/evaluation_plots"

# os.makedirs(EVAL_PLOTS_DIR, exist_ok=True)


def load_model(model_id):
    model_path = os.path.join(MODEL_DIR, f"{model_id}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_id} not found")
    return joblib.load(model_path)


# import os
# import pandas as pd
# import joblib
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, f1_score,
#     confusion_matrix, roc_auc_score
# )
# import matplotlib.pyplot as plt
# import seaborn as sns
# import uuid

# from .data_service import load_dataset

# MODEL_DIR = "models"
# EVAL_PLOTS_DIR = "static/evaluation_plots"

# os.makedirs(EVAL_PLOTS_DIR, exist_ok=True)


def load_model(model_id):
    model_path = os.path.join(MODEL_DIR, f"{model_id}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_id} not found")
    return joblib.load(model_path)


def evaluate_model(model_id, dataset_name, target_column, metrics, visualization):
    # Load dataset & model
    df = load_dataset(dataset_name)
    model = load_model(model_id)

    if not target_column or target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    # Split features & target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    preds = model.predict(X)

    results = {}

    if "accuracy" in metrics:
        results["accuracy"] = accuracy_score(y, preds)
    if "precision" in metrics:
        results["precision"] = precision_score(y, preds, average="weighted", zero_division=0)
    if "recall" in metrics:
        results["recall"] = recall_score(y, preds, average="weighted", zero_division=0)
    if "f1" in metrics:
        results["f1"] = f1_score(y, preds, average="weighted", zero_division=0)

    # if "roc_auc" in metrics:
    #     try:
    #         if hasattr(model, "predict_proba"):
    #             proba = model.predict_proba(X)
    #             results["roc_auc"] = roc_auc_score(y, proba[:, 1])
    #         else:
    #             results["roc_auc"] = "Model does not support probability estimates"
    #     except Exception as e:
    #         results["roc_auc"] = f"Not supported: {str(e)}"
    
    if "roc_auc" in metrics:
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                if proba.shape[1] == 2:  # ✅ only binary
                    results["roc_auc"] = roc_auc_score(y, proba[:, 1])
                else:  # multiclass support
                    results["roc_auc"] = roc_auc_score(y, proba, multi_class="ovr")
            else:
                results["roc_auc"] = "Model does not support probability estimates"
        except Exception as e:
            results["roc_auc"] = f"Not supported: {str(e)}"

    # Visualization (only confusion matrix for now)
    if visualization == "confusion_matrix":
        cm = confusion_matrix(y, preds)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plot_id = f"{uuid.uuid4()}.png"
        plot_path = os.path.join(EVAL_PLOTS_DIR, plot_id)
        plt.savefig(plot_path)
        plt.close()
        results["visualization"] = f"/{plot_path}"
    elif visualization != "none":
        results["visualization"] = f"{visualization} not implemented yet"

    return results


# def evaluate_model(model_id, dataset_name, metrics, visualization):
#     # Load dataset & model
#     df = load_dataset(dataset_name)
#     model = load_model(model_id)

#     # Separate features and target
#     if "target" not in df.columns:
#         raise ValueError("Dataset must include a 'target' column for evaluation")
#     X = df.drop(columns=["target"])
#     y = df["target"]

#     preds = model.predict(X)

#     results = {}

#     if "accuracy" in metrics:
#         results["accuracy"] = accuracy_score(y, preds)
#     if "precision" in metrics:
#         results["precision"] = precision_score(y, preds, average="weighted", zero_division=0)
#     if "recall" in metrics:
#         results["recall"] = recall_score(y, preds, average="weighted", zero_division=0)
#     if "f1" in metrics:
#         results["f1"] = f1_score(y, preds, average="weighted", zero_division=0)
#     if "roc_auc" in metrics:
#         try:
#             proba = model.predict_proba(X)
#             results["roc_auc"] = roc_auc_score(y, proba[:, 1])
#         except Exception:
#             results["roc_auc"] = "Not supported for this model"

#     # Visualization
#     plot_path = None
#     if visualization == "confusion_matrix":
#         cm = confusion_matrix(y, preds)
#         plt.figure(figsize=(6, 4))
#         sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
#         plt.xlabel("Predicted")
#         plt.ylabel("Actual")
#         plot_id = f"{uuid.uuid4()}.png"
#         plot_path = os.path.join(EVAL_PLOTS_DIR, plot_id)
#         plt.savefig(plot_path)
#         plt.close()
#         results["visualization"] = f"/{plot_path}"
#     elif visualization != "none":
#         results["visualization"] = f"{visualization} not implemented yet"

#     return results
