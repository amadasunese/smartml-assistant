import pandas as pd
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from services.data_services import load_dataset, PROCESSED_DIR, MODEL_DIR, UPLOAD_FOLDER


# DATASET_DIR = "datasets"
# PROCESSED_DIR = "uploaded_data"
# os.makedirs(PROCESSED_DIR, exist_ok=True)


# PROCESSED_DIR = "uploaded_data"
# MODEL_DIR = "saved_models"
# UPLOAD_FOLDER = "uploaded_data"
# os.makedirs(PROCESSED_DIR, exist_ok=True)

# PROCESSED_DIR = "processed_data"
# MODEL_DIR = "saved_models"
# UPLOAD_FOLDER = "uploaded_data"

def _load_dataset(dataset_name):
    """Load dataset based on file extension (CSV, Excel, JSON)."""
    dataset_path = os.path.join(UPLOAD_FOLDER, dataset_name)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset not found")

    ext = dataset_name.split(".")[-1].lower()

    if ext == "csv":
        df = pd.read_csv(dataset_path)
    elif ext in ["xls", "xlsx"]:
        df = pd.read_excel(dataset_path)
    elif ext == "json":
        df = pd.read_json(dataset_path)
    else:
        raise ValueError("Unsupported file format. Only CSV, Excel, and JSON are supported.")

    return df, ext

def _save_dataset(df, dataset_name, ext):
    """Save processed dataset in the same format as input."""
    processed_name = dataset_name.replace(f".{ext}", f"_processed.{ext}")
    processed_path = os.path.join(PROCESSED_DIR, processed_name)

    if ext == "csv":
        df.to_csv(processed_path, index=False)
    elif ext in ["xls", "xlsx"]:
        df.to_excel(processed_path, index=False)
    elif ext == "json":
        df.to_json(processed_path, orient="records", indent=2)
    else:
        raise ValueError("Unsupported file format for saving.")

    return processed_name


def apply_feature_engineering(dataset_name, feature_generation, feature_selection, dimensionality_reduction):
    df, ext = _load_dataset(dataset_name)

    # -------- Feature Generation -------- #
    if feature_generation == "polynomial":
        poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
        poly_features = poly.fit_transform(df.select_dtypes(include=["int64", "float64"]))
        df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(df.select_dtypes(include=["int64", "float64"]).columns))

    elif feature_generation == "interaction":
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        inter_features = poly.fit_transform(df.select_dtypes(include=["int64", "float64"]))
        df = pd.DataFrame(inter_features, columns=poly.get_feature_names_out(df.select_dtypes(include=["int64", "float64"]).columns))

    elif feature_generation == "datetime":
        for col in df.select_dtypes(include=["datetime64"]):
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day

    # -------- Feature Selection -------- #
    if feature_selection == "variance":
        selector = VarianceThreshold(threshold=0.01)
        df = pd.DataFrame(selector.fit_transform(df), columns=df.columns[selector.get_support()])

    elif feature_selection == "kbest":
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        selector = SelectKBest(score_func=f_classif, k=min(5, X.shape[1]))
        X_new = selector.fit_transform(X, y)
        df = pd.DataFrame(X_new, columns=X.columns[selector.get_support()])
        df["target"] = y

    elif feature_selection == "rfe":
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        model = LogisticRegression(max_iter=500)
        selector = RFE(model, n_features_to_select=min(5, X.shape[1]))
        X_new = selector.fit_transform(X, y)
        df = pd.DataFrame(X_new, columns=X.columns[selector.support_])
        df["target"] = y

    # -------- Dimensionality Reduction -------- #
    if dimensionality_reduction == "pca":
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        pca = PCA(n_components=min(3, X.shape[1]))
        X_pca = pca.fit_transform(X)
        df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
        df["target"] = y

    elif dimensionality_reduction == "lda":
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        lda = LDA(n_components=1)
        X_lda = lda.fit_transform(X, y)
        df = pd.DataFrame(X_lda, columns=["LD1"])
        df["target"] = y

    elif dimensionality_reduction == "tsne":
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)
        df = pd.DataFrame(X_tsne, columns=["TSNE1", "TSNE2"])
        df["target"] = y

    # -------- Save processed dataset -------- #
    processed_name = _save_dataset(df, dataset_name, ext)

    return {
        "message": "Feature engineering applied successfully",
        "processed_dataset": processed_name,
        "shape": df.shape,
        "preview": df.head(5).to_dict(orient="records")  # 👈 dataset sample for frontend
    }