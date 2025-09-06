# views.py - Fix imports
from flask import Blueprint, jsonify, request, render_template, send_file
from services.data_services import load_dataset, save_uploaded_file, UPLOAD_FOLDER, PROCESSED_DIR
from services.model_services import list_models, load_model, save_model
from services.preprocess_services import preprocess_pipeline
from services.train_services import train_model, get_dataset_columns
from services.feature_engineering_service import apply_feature_engineering
import requests
import os
import re




import uuid
import os
import logging
logging.basicConfig(level=logging.DEBUG)
# from services.data_services import load_dataset, save_uploaded_file, list_models, load_model, preprocess_pipeline, train_model
import scipy.stats as stats
import logging
from flask import jsonify, request
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from io import StringIO  # Import StringIO
from services.evaluate_services import evaluate_model




bp = Blueprint("main", __name__)


@bp.route("/")
def index():
    return render_template("index.html")


# Upload data
@bp.route("/upload", methods=["POST"])
def upload_dataset():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    filename = save_uploaded_file(file)
    return jsonify({"message": "File uploaded", "filename": filename})



@bp.route("/upload/url", methods=["POST"])
def upload_dataset_url():
    data = request.get_json()
    url = data.get("url")

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        # Fetch the dataset from the URL
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Extract filename from URL
        filename = url.split("/")[-1]
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        # Save file locally
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return jsonify({"message": "File downloaded", "filename": filename})
    except Exception as e:
        return jsonify({"error": f"Failed to fetch dataset: {str(e)}"}), 500


# @bp.route("/upload/cloud", methods=["POST"])
# def upload_dataset_cloud():
#     data = request.get_json()
#     source = data.get("provider")   # e.g., "gdrive", "dropbox", "s3"
#     url = data.get("url")

#     if not url or not source:
#         return jsonify({"error": "Missing cloud source or URL"}), 400

#     try:
#         # Download file from cloud link (same as URL import)
#         response = requests.get(url, stream=True)
#         response.raise_for_status()

#         filename = f"{source}_{url.split('/')[-1]}"
#         filepath = os.path.join(UPLOAD_FOLDER, filename)

#         with open(filepath, "wb") as f:
#             for chunk in response.iter_content(chunk_size=8192):
#                 f.write(chunk)

#         return jsonify({"message": f"Imported from {source}", "filename": filename})
#     except Exception as e:
#         return jsonify({"error": f"Failed to import from {source}: {str(e)}"}), 500

@bp.route("/upload/cloud", methods=["POST"])
def upload_dataset_cloud():
    data = request.get_json()
    provider = data.get("provider")
    url = data.get("url")

    if not provider or not url:
        return jsonify({"error": "Missing cloud source or URL"}), 400

    # --- Validate and normalize direct download URLs ---
    if provider.lower() == "google drive":
        # User must provide FILE_ID, not the full link
        if "drive.google.com" in url:
            return jsonify({"error": "Please provide only the FILE_ID, not the full Google Drive link"}), 400
        url = f"https://drive.google.com/uc?export=download&id={url.strip()}"  # build direct link

    elif provider.lower() == "dropbox":
        # Ensure dl=1 for direct download
        if url.endswith("?dl=0"):
            url = url.replace("?dl=0", "?dl=1")
        elif "dropbox.com" in url and "?dl=1" not in url:
            url += "?dl=1"

    elif provider.lower() == "aws s3":
        # Expect a valid S3 object link (public or pre-signed)
        if not url.startswith("http"):
            return jsonify({"error": "Please provide a valid AWS S3 object URL"}), 400

    else:
        return jsonify({"error": f"Unsupported provider: {provider}"}), 400

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Extract filename from headers, fallback to URL
        cd = response.headers.get("content-disposition")
        if cd:
            match = re.findall('filename="?([^"]+)"?', cd)
            filename = match[0] if match else None
        else:
            filename = None

        if not filename:
            filename = url.split("/")[-1].split("?")[0]

        filename = f"{provider.lower().replace(' ', '_')}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return jsonify({"message": f"Imported from {provider}", "filename": filename}), 200

    except Exception as e:
        return jsonify({"error": f"Failed to import from {provider}: {str(e)}"}), 500

# views.py - Enhanced EDA function
@bp.route("/eda/summary/<dataset_name>", methods=["GET"])
def dataset_summary(dataset_name):
    logging.debug(f"EDA requested for dataset: {dataset_name}")
    try:
        df = load_dataset(dataset_name)
        logging.debug(f"Dataset loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        return jsonify({"error": f"Dataset '{dataset_name}' not found"}), 404
    except Exception as e:
        return jsonify({"error": f"Error loading dataset: {str(e)}"}), 500

    try:
        # Handle potential serialization issues with NaN values
        head_data = df.head(5).replace({float('nan'): None}).to_dict(orient="records")
        describe_data = df.describe(include="all").replace({float('nan'): None}).to_dict()
        
        return jsonify({
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "head": head_data,
            "describe": describe_data
        })
    except Exception as e:
        return jsonify({"error": f"Error processing data: {str(e)}"}), 500





# @bp.route("/eda/<analysis_type>/<dataset_name>", methods=["GET"])
# def perform_eda(analysis_type, dataset_name):
#     logging.debug(f"EDA requested for dataset: {dataset_name}, type: {analysis_type}")
#     try:
#         df = load_dataset(dataset_name)
#     except FileNotFoundError:
#         return jsonify({"error": f"Dataset '{dataset_name}' not found"}), 404
#     except Exception as e:
#         return jsonify({"error": f"Error loading dataset: {str(e)}"}), 500

#     try:
#         if analysis_type == 'summary':
#             head_data = df.head(5).replace({np.nan: None}).to_dict(orient="records")
#             describe_data = df.describe(include="all").replace({np.nan: None}).to_dict()
#             return jsonify({
#                 "shape": df.shape,
#                 "columns": df.columns.tolist(),
#                 "head": head_data,
#                 "describe": describe_data
#             })

#         elif analysis_type == 'correlation':
#             numeric_df = df.select_dtypes(include=np.number)
#             corr_matrix = numeric_df.corr().replace({np.nan: None}).values.tolist()
#             return jsonify({
#                 "columns": numeric_df.columns.tolist(),
#                 "correlation": corr_matrix
#             })

#         elif analysis_type == 'distribution':
#             distribution_data = {}
#             for col in df.columns:
#                 if pd.api.types.is_numeric_dtype(df[col]):
#                     dropna_series = df[col].dropna()
#                     if len(dropna_series) > 1:
#                         distribution_data[col] = {
#                             "skewness": float(dropna_series.skew()),
#                             "kurtosis": float(dropna_series.kurtosis()),
#                             "normality": float(stats.shapiro(dropna_series)[1])
#                         }
#                     else:
#                         distribution_data[col] = {
#                             "skewness": None,
#                             "kurtosis": None,
#                             "normality": None
#                         }
#             return jsonify({
#                 "columns": df.columns.tolist(),
#                 "distribution": distribution_data
#             })

#         elif analysis_type == 'missing':
#             missing_data = {}
#             total_rows = len(df)
#             for col in df.columns:
#                 missing_count = int(df[col].isnull().sum())
#                 missing_data[col] = {
#                     "count": missing_count,
#                     "percentage": (missing_count / total_rows) * 100
#                 }
#             return jsonify({
#                 "columns": df.columns.tolist(),
#                 "missing": missing_data
#             })
        
#         else:
#             return jsonify({"error": "Invalid analysis type"}), 400

#     except Exception as e:
#         logging.error(f"Error processing data for {analysis_type}: {str(e)}")
#         return jsonify({"error": f"Error processing data: {str(e)}"}), 500



# bp = Blueprint("main", __name__)

# DATA_DIR = "datasets"

# def load_dataset(dataset_name):
#     """Loads a dataset from the specified directory."""
#     file_path = os.path.join(DATA_DIR, dataset_name)
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"File not found: {file_path}")

#     # Handle different file types
#     if file_path.endswith('.csv'):
#         return pd.read_csv(file_path)
#     elif file_path.endswith(('.xls', '.xlsx')):
#         return pd.read_excel(file_path)
#     elif file_path.endswith('.json'):
#         return pd.read_json(file_path)
#     else:
#         raise ValueError("Unsupported file type")

# @bp.route("/eda/<analysis_type>/<dataset_name>", methods=["GET"])
# def perform_eda(analysis_type, dataset_name):
#     logging.debug(f"EDA requested for dataset: {dataset_name}, type: {analysis_type}")
#     try:
#         df = load_dataset(dataset_name)
#     except FileNotFoundError:
#         return jsonify({"error": f"Dataset '{dataset_name}' not found"}), 404
#     except Exception as e:
#         return jsonify({"error": f"Error loading dataset: {str(e)}"}), 500

#     try:
#         if analysis_type == 'summary':
#             # Capture the output of df.info()
#             buf = StringIO()
#             df.info(buf=buf)
#             info_text = buf.getvalue()

#             head_data = df.head(5).replace({np.nan: None}).to_dict(orient="records")
#             describe_data = df.describe(include="all").replace({np.nan: None}).to_dict()
#             return jsonify({
#                 "shape": df.shape,
#                 "columns": df.columns.tolist(),
#                 "head": head_data,
#                 "describe": describe_data,
#                 "info": info_text  # Include the captured info() output
#             })

#         elif analysis_type == 'correlation':
#             numeric_df = df.select_dtypes(include=np.number)
#             if numeric_df.empty:
#                 return jsonify({"error": "No numeric columns for correlation analysis."}), 400
#             corr_matrix = numeric_df.corr().replace({np.nan: None}).values.tolist()
#             return jsonify({
#                 "columns": numeric_df.columns.tolist(),
#                 "correlation": corr_matrix
#             })

#         elif analysis_type == 'distribution':
#             distribution_data = {}
#             for col in df.columns:
#                 if pd.api.types.is_numeric_dtype(df[col]):
#                     dropna_series = df[col].dropna()
#                     if len(dropna_series) > 1:
#                         distribution_data[col] = {
#                             "skewness": float(dropna_series.skew()),
#                             "kurtosis": float(dropna_series.kurtosis()),
#                             "normality": float(stats.shapiro(dropna_series)[1])
#                         }
#                     else:
#                         distribution_data[col] = {
#                             "skewness": None,
#                             "kurtosis": None,
#                             "normality": None
#                         }
#             return jsonify({
#                 "columns": df.columns.tolist(),
#                 "distribution": distribution_data
#             })

#         elif analysis_type == 'missing':
#             missing_data = {}
#             total_rows = len(df)
#             for col in df.columns:
#                 missing_count = int(df[col].isnull().sum())
#                 missing_data[col] = {
#                     "count": missing_count,
#                     "percentage": (missing_count / total_rows) * 100
#                 }
#             return jsonify({
#                 "columns": df.columns.tolist(),
#                 "missing": missing_data
#             })
            
#         # New EDA analysis for categorical variables
#         elif analysis_type == 'categorical':
#             categorical_data = {}
#             for col in df.columns:
#                 if not pd.api.types.is_numeric_dtype(df[col]):
#                     categorical_data[col] = df[col].value_counts().to_dict()
#             return jsonify({
#                 "columns": df.columns.tolist(),
#                 "categorical_counts": categorical_data
#             })
            
#         # New EDA analysis for outlier detection
#         elif analysis_type == 'outliers':
#             outlier_data = {}
#             for col in df.columns:
#                 if pd.api.types.is_numeric_dtype(df[col]):
#                     Q1 = df[col].quantile(0.25)
#                     Q3 = df[col].quantile(0.75)
#                     IQR = Q3 - Q1
#                     lower_bound = Q1 - 1.5 * IQR
#                     upper_bound = Q3 + 1.5 * IQR
                    
#                     outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
#                     outlier_data[col] = {
#                         "count": int(outliers.count()),
#                         "percentage": float((outliers.count() / len(df)) * 100)
#                     }
#             return jsonify({
#                 "columns": df.columns.tolist(),
#                 "outliers": outlier_data
#             })
            
#         # New EDA analysis for Bivariate plots (Pair Plot)
#         elif analysis_type == 'pairplot':
#             numeric_df = df.select_dtypes(include=np.number)
#             if numeric_df.empty:
#                 return jsonify({"error": "No numeric columns for pairplot"}), 400
                
#             fig = sns.pairplot(numeric_df).fig
            
#             buf = io.BytesIO()
#             fig.savefig(buf, format='png')
#             buf.seek(0)
            
#             img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            
#             plt.close(fig)
            
#             return jsonify({"image": f"data:image/png;base64,{img_base64}"})

#         else:
#             return jsonify({"error": "Invalid analysis type"}), 400

#     except Exception as e:
#         logging.error(f"Error processing data for {analysis_type}: {str(e)}")
#         return jsonify({"error": f"Error processing data: {str(e)}"}), 500


@bp.route("/eda/<analysis_type>/<dataset_name>", methods=["GET"])
def perform_eda(analysis_type, dataset_name):
    logging.debug(f"EDA requested for dataset: {dataset_name}, type: {analysis_type}")
    try:
        df = load_dataset(dataset_name)
    except FileNotFoundError:
        return jsonify({"error": f"Dataset '{dataset_name}' not found"}), 404
    except Exception as e:
        return jsonify({"error": f"Error loading dataset: {str(e)}"}), 500

    try:
        if analysis_type == 'summary':
            # Capture the output of df.info()
            buf = StringIO()
            df.info(buf=buf)
            info_text = buf.getvalue()

            head_data = df.head(5).replace({np.nan: None}).to_dict(orient="records")
            describe_data = df.describe(include="all").replace({np.nan: None}).to_dict()
            return jsonify({
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "head": head_data,
                "describe": describe_data,
                "info": info_text
            })

        elif analysis_type == 'correlation':
            numeric_df = df.select_dtypes(include=np.number)
            if numeric_df.empty:
                return jsonify({"error": "No numeric columns for correlation analysis."}), 400
            corr_matrix = numeric_df.corr().replace({np.nan: None}).values.tolist()
            return jsonify({
                "columns": numeric_df.columns.tolist(),
                "correlation": corr_matrix
            })

        elif analysis_type == 'distribution':
            distribution_data = {}
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    dropna_series = df[col].dropna()
                    if len(dropna_series) > 1:
                        distribution_data[col] = {
                            "skewness": float(dropna_series.skew()),
                            "kurtosis": float(dropna_series.kurtosis()),
                            "normality": float(stats.shapiro(dropna_series)[1])
                        }
                    else:
                        distribution_data[col] = {
                            "skewness": None,
                            "kurtosis": None,
                            "normality": None
                        }
            return jsonify({
                "columns": df.columns.tolist(),
                "distribution": distribution_data
            })

        elif analysis_type == 'missing':
            missing_data = {}
            total_rows = len(df)
            for col in df.columns:
                missing_count = int(df[col].isnull().sum())
                missing_data[col] = {
                    "count": missing_count,
                    "percentage": (missing_count / total_rows) * 100
                }
            return jsonify({
                "columns": df.columns.tolist(),
                "missing": missing_data
            })
            
        elif analysis_type == 'categorical':
            categorical_data = {}
            for col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    categorical_data[col] = df[col].value_counts().to_dict()
            return jsonify({
                "columns": df.columns.tolist(),
                "categorical_counts": categorical_data
            })
            
        elif analysis_type == 'outliers':
            outlier_data = {}
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                    outlier_data[col] = {
                        "count": int(outliers.count()),
                        "percentage": float((outliers.count() / len(df)) * 100)
                    }
            return jsonify({
                "columns": df.columns.tolist(),
                "outliers": outlier_data
            })
            
        # elif analysis_type == 'pairplot':
        #     # Check for pandas version to avoid deprecated option issue
        #     if pd.__version__ >= '2.0.0':
        #         # No specific setting needed, pandas >= 2.0 handles this automatically
        #         pass
            
        #     numeric_df = df.select_dtypes(include=np.number)
        #     if numeric_df.empty:
        #         return jsonify({"error": "No numeric columns for pairplot"}), 400
                
        #     fig = sns.pairplot(numeric_df).figure
            
        #     buf = io.BytesIO()
        #     fig.savefig(buf, format='png')
        #     buf.seek(0)
            
        #     img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            
        #     plt.close(fig)  # Crucial to close the plot after saving
            
        #     return jsonify({"image": f"data:image/png;base64,{img_base64}"})

        elif analysis_type == 'pairplot':
            # Check for pandas version to avoid deprecated option issue
            if pd.__version__ >= '2.0.0':
                # No specific setting needed, pandas >= 2.0 handles this automatically
                pass

            numeric_df = df.select_dtypes(include=np.number)
            if numeric_df.empty:
                return jsonify({"error": "No numeric columns for pairplot"}), 400

            # Create pairplot
            g = sns.pairplot(numeric_df)
            g.figure.tight_layout()  # fixes "title not moved" warnings

            # Save to buffer
            buf = io.BytesIO()
            g.figure.savefig(buf, format='png', bbox_inches='tight')  # bbox ensures content fits
            buf.seek(0)

            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            plt.close(g.figure)  # Close the figure to free memory

            return jsonify({"image": f"data:image/png;base64,{img_base64}"})
        
        elif analysis_type == 'duplicates':
            duplicate_count = df.duplicated().sum()
            return jsonify({
                "total_rows": len(df),
                "duplicates": int(duplicate_count),
                "percentage": float((duplicate_count / len(df)) * 100)
            })

        elif analysis_type == 'imbalance':
            imbalance_data = {}
            if 'target' in df.columns:  # adjust if target is configurable
                counts = df['target'].value_counts()
                imbalance_data = counts.to_dict()
            else:
                return jsonify({"error": "No 'target' column found for imbalance check"}), 400

            return jsonify({
                "target_column": "target",
                "class_distribution": imbalance_data
            })

        elif analysis_type == 'skewness':
            skew_data = {}
            for col in df.select_dtypes(include=np.number).columns:
                skew_val = df[col].skew()
                skew_data[col] = {
                    "skewness": float(skew_val),
                    "boxcox_applicable": bool((df[col] > 0).all() and abs(skew_val) > 0.5)
                }
            return jsonify({
                "columns": df.select_dtypes(include=np.number).columns.tolist(),
                "skewness": skew_data
            })

        else:
            return jsonify({"error": "Invalid analysis type"}), 400

    except Exception as e:
        logging.error(f"Error processing data for {analysis_type}: {str(e)}")
        return jsonify({"error": f"Error processing data: {str(e)}"}), 500

# views.py - Add this function
# @bp.route("/datasets", methods=["GET"])
# def list_datasets():
#     try:
#         datasets = [f for f in os.listdir(UPLOAD_FOLDER) 
#                    if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]
#         return jsonify({"datasets": datasets})
#     except Exception as e:
#         return jsonify({"error": f"Error listing datasets: {str(e)}"}), 500

@bp.route("/datasets", methods=["GET"])
def list_datasets():
    try:
        # Ensure both directories exist
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)

        # Raw uploaded datasets
        raw_datasets = [
            f for f in os.listdir(UPLOAD_FOLDER) 
            if os.path.isfile(os.path.join(UPLOAD_FOLDER, f)) and f.lower().endswith((".csv", ".xls", ".xlsx", ".json"))
        ]

        # Processed datasets
        processed_datasets = [
            f for f in os.listdir(PROCESSED_DIR)
            if os.path.isfile(os.path.join(PROCESSED_DIR, f)) and f.lower().endswith(".csv")
        ]

        # Merge both lists
        datasets = raw_datasets + processed_datasets

        return jsonify({"datasets": datasets})
    except Exception as e:
        return jsonify({"error": f"Error listing datasets: {str(e)}"}), 500


# Models

@bp.route("/models", methods=["GET"])
def get_models():
    return jsonify({"models": list_models()})


@bp.route("/preprocess/<dataset_name>", methods=["POST"])
def preprocess(dataset_name):
    config = request.json or {}
    processed_name = preprocess_pipeline(dataset_name, config)
    return jsonify({"message": "Dataset preprocessed", "processed_name": processed_name})


@bp.route("/feature-engineering/<dataset_name>", methods=["POST"])
def feature_engineering(dataset_name):
    try:
        form_data = request.get_json()
        feature_generation = form_data.get("feature_generation", "none")
        feature_selection = form_data.get("feature_selection", "none")
        dimensionality_reduction = form_data.get("dimensionality_reduction", "none")

        result = apply_feature_engineering(
            dataset_name,
            feature_generation,
            feature_selection,
            dimensionality_reduction
        )

        return jsonify(result)

    except FileNotFoundError as fnf:
        return jsonify({"error": str(fnf)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@bp.route("/train/<dataset_name>", methods=["POST"])
def train(dataset_name):
    config = request.json or {}
    model_id, metrics = train_model(dataset_name, config)
    return jsonify({"model_id": model_id, "metrics": metrics})


@bp.route("/dataset/columns/<dataset_name>", methods=["GET"])
def dataset_columns(dataset_name):
    try:
        columns = get_dataset_columns(dataset_name)
        return jsonify({"columns": columns})
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@bp.route("/get-columns/<dataset_name>", methods=["GET"])
def get_columns(dataset_name):
    columns = get_dataset_columns(dataset_name)
    if columns:
        return jsonify(columns)
    else:
        return jsonify({"error": "Dataset not found"}), 404




@bp.route("/evaluate/<model_id>", methods=["POST"])
def evaluate(model_id):
    try:
        data = request.get_json()
        print("📥 Incoming request:", data)  # <--- DEBUG
        dataset_name = data.get("dataset")
        target_column = data.get("target")
        metrics = data.get("metrics", [])
        visualization = data.get("visualization", "none")

        if not dataset_name or not target_column:
            print("❌ Missing values:", dataset_name, target_column)  # <--- DEBUG
            return jsonify({"error": "Dataset and target column are required"}), 400

        results = evaluate_model(model_id, dataset_name, target_column, metrics, visualization)
        return jsonify(results)

    except Exception as e:
        print("🔥 Server error in /evaluate:", str(e))  # <--- DEBUG
        return jsonify({"error": str(e)}), 500




# A simplified Flask route example
from flask import abort
import joblib
from services.model_services import MODEL_DIR



def load_model_features(model_id):
    """
    Loads a model and returns its feature names.
    """
    # Support both joblib and pickle files
    joblib_path = os.path.join(MODEL_DIR, f"{model_id}.joblib")
    pickle_path = os.path.join(MODEL_DIR, f"{model_id}.pkl")

    model_path = None
    if os.path.exists(joblib_path):
        model_path = joblib_path
    elif os.path.exists(pickle_path):
        model_path = pickle_path

    if not model_path:
        return None

    try:
        model = joblib.load(model_path)
        if hasattr(model, 'feature_names_in_'):
            return model.feature_names_in_.tolist()
        else:
            return ["feature_1", "feature_2", "feature_3"]
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        return None


@bp.route("/model/features/<model_id>", methods=["GET"])
def get_model_features(model_id):
    """
    Backend route to get a trained model's feature names.
    """
    features = load_model_features(model_id)

    if features:
        # Return the feature list in a JSON object
        return jsonify({"features": features})
    else:
        # This will prevent the SyntaxError on the frontend
        return abort(404, "Model or features not found.")
    

@bp.route("/predict/<model_id>", methods=["POST"])
def predict(model_id):
    model = load_model(model_id)
    input_data = request.json.get("input_data", {})

    df = pd.DataFrame([input_data], columns=model.feature_names)
    preds = model.predict(df)

    return jsonify({"prediction": preds.tolist()})


@bp.route('/api/download-model/<model_id>')
def download_model(model_id):
    """
    A Flask route to download a specific model.
    The model_id is captured from the URL path.
    """
    # Construct the file path for the requested model
    model_path = os.path.join(MODEL_DIR, f"{model_id}.pkl")
    
    try:
        # Check if the file exists and is a file (not a directory)
        if not os.path.isfile(model_path):
            return jsonify({'error': 'Model not found'}), 404
            

        return send_file(model_path, as_attachment=True, download_name=f'{model_id}.pkl')
        
    except FileNotFoundError:
        # This block is for handling errors if the file is not found
        return jsonify({'error': 'Model not found'}), 404
        
    except Exception as e:
        # Catch any other potential exceptions and return a server error
        return jsonify({'error': str(e)}), 500


# def has_ext(filename, extensions):
#     return any(filename.lower().endswith(ext) for ext in extensions)

# track predictions (in memory for now)
prediction_counter = 0

def has_ext(filename, exts):
    return any(filename.lower().endswith(ext) for ext in exts)

@bp.route("/stats", methods=["GET"])
def stats():
    try:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)

        datasets = [f for f in os.listdir(UPLOAD_FOLDER) if has_ext(f, [".csv", ".xls", ".xlsx", ".json"])]
        processes = [f for f in os.listdir(PROCESSED_DIR) if has_ext(f, [".csv", ".xls", ".xlsx", ".json"]) and f.startswith("processed_")]
        models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]

        return jsonify({
            "datasets": len(datasets),
            "models": len(models),
            "processes": len(processes),
            "predictions": prediction_counter
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# @bp.route("/stats", methods=["GET"])
# def stats():
#     try:
#         # datasets = raw uploaded files in UPLOAD_FOLDER
#         datasets = [
#             f for f in os.listdir(UPLOAD_FOLDER)
#             if has_ext(f, [".csv", ".xls", ".xlsx", ".json"])
#         ]

#         print('this is the dataset', datasets)

#         # processes = processed datasets in PROCESSED_DIR
#         processes = [
#             f for f in os.listdir(PROCESSED_DIR)
#             if has_ext(f, [".csv", ".xls", ".xlsx", ".json"]) and f.startswith("processed_")
#         ]

#         # models = saved model files in MODEL_DIR
#         models = [
#             f for f in os.listdir(MODEL_DIR)
#             if f.endswith(".pkl")
#         ]

#         print('this is the train model', models)

#         return jsonify({
#             "datasets": len(datasets),
#             "models": len(models),
#             "processes": len(processes),
#             "predictions": prediction_counter
#         })
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500