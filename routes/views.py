# views.py - Fix imports
from flask import Blueprint, jsonify, request, render_template
from services.data_services import load_dataset, save_uploaded_file, UPLOAD_FOLDER
from services.model_services import list_models, load_model, save_model
from services.preprocess_services import preprocess_pipeline
from services.train_services import train_model
import pandas as pd
import uuid
import os
import logging
logging.basicConfig(level=logging.DEBUG)
# from services.data_services import load_dataset, save_uploaded_file, list_models, load_model, preprocess_pipeline, train_model


import pandas as pd

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
#             head_data = df.head(5).replace({float('nan'): None}).to_dict(orient="records")
#             describe_data = df.describe(include="all").replace({float('nan'): None}).to_dict()
#             return jsonify({
#                 "shape": df.shape,
#                 "columns": df.columns.tolist(),
#                 "head": head_data,
#                 "describe": describe_data
#             })
#         elif analysis_type == 'correlation':
#             corr_matrix = df.corr(numeric_only=True).replace({float('nan'): None}).values.tolist()
#             return jsonify({
#                 "columns": df.columns[df.select_dtypes(include=np.number).columns].tolist(),
#                 "correlation": corr_matrix
#             })
#         elif analysis_type == 'distribution':
#             import scipy.stats as stats
#             distribution_data = {}
#             for col in df.columns:
#                 if pd.api.types.is_numeric_dtype(df[col]):
#                     distribution_data[col] = {
#                         "skewness": df[col].skew(),
#                         "kurtosis": df[col].kurtosis(),
#                         "normality": stats.shapiro(df[col].dropna())[1] # p-value from Shapiro-Wilk test
#                     }
#             return jsonify({
#                 "columns": df.columns.tolist(),
#                 "distribution": distribution_data
#             })
#         elif analysis_type == 'missing':
#             missing_data = {}
#             for col in df.columns:
#                 missing_count = df[col].isnull().sum()
#                 total_count = len(df[col])
#                 missing_data[col] = {
#                     "count": missing_count,
#                     "percentage": (missing_count / total_count) * 100
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

import scipy.stats as stats
import logging
from flask import jsonify, request
import pandas as pd
import numpy as np

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
            head_data = df.head(5).replace({np.nan: None}).to_dict(orient="records")
            describe_data = df.describe(include="all").replace({np.nan: None}).to_dict()
            return jsonify({
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "head": head_data,
                "describe": describe_data
            })

        elif analysis_type == 'correlation':
            numeric_df = df.select_dtypes(include=np.number)
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
        
        else:
            return jsonify({"error": "Invalid analysis type"}), 400

    except Exception as e:
        logging.error(f"Error processing data for {analysis_type}: {str(e)}")
        return jsonify({"error": f"Error processing data: {str(e)}"}), 500


# views.py - Add this function
@bp.route("/datasets", methods=["GET"])
def list_datasets():
    try:
        datasets = [f for f in os.listdir(UPLOAD_FOLDER) 
                   if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]
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

@bp.route("/train/<dataset_name>", methods=["POST"])
def train(dataset_name):
    config = request.json or {}
    model_id, metrics = train_model(dataset_name, config)
    return jsonify({"model_id": model_id, "metrics": metrics})

@bp.route("/predict/<model_id>", methods=["POST"])
def predict(model_id):
    model = load_model(model_id)
    data = request.json.get("data")
    df = pd.DataFrame([data])
    preds = model.predict(df).tolist()
    return jsonify({"predictions": preds})