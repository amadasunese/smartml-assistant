import os
import pandas as pd
from services.data_services import load_dataset

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np

PROCESSED_DIR = "uploaded_data"

def preprocess_pipeline(dataset_name, config):
    try:
        df = load_dataset(dataset_name)

        # Standardize Missing Values
        df.replace(["", " ", "null", "N/A", "na", "-"], np.nan, inplace=True)
        
        # 2. Missing Values Strategy
        missing_strategy = config.get("missing_strategy")
        if missing_strategy == "mean" or missing_strategy == "median" or missing_strategy == "most_frequent":
            imputer = SimpleImputer(strategy=missing_strategy)
            # Identify numeric and categorical columns for imputation
            numeric_cols = df.select_dtypes(include=np.number).columns
            categorical_cols = df.select_dtypes(exclude=np.number).columns
            
            if not numeric_cols.empty:
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            
            if not categorical_cols.empty and missing_strategy == "most_frequent":
                imputer_cat = SimpleImputer(strategy="most_frequent")
                df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

        elif missing_strategy == "drop":
            df = df.dropna()
        elif missing_strategy == "interpolate":
            numeric_cols = df.select_dtypes(include=np.number).columns
            df[numeric_cols] = df[numeric_cols].interpolate()
        
        # 3. Outlier Handling
        outlier_method = config.get("outlier_method")
        numeric_cols = df.select_dtypes(include=np.number).columns
        if outlier_method == "clip":
            for col in numeric_cols:
                lower_bound = df[col].quantile(0.05)
                upper_bound = df[col].quantile(0.95)
                df[col] = np.clip(df[col], lower_bound, upper_bound)
        elif outlier_method == "remove":
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        # 4. Categorical Encoding
        encoding_method = config.get("encoding_method")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if encoding_method == "onehot":
            if not categorical_cols.empty:
                encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                encoded_cols = pd.DataFrame(encoder.fit_transform(df[categorical_cols]))
                encoded_cols.columns = encoder.get_feature_names_out(categorical_cols)
                df = pd.concat([df.drop(columns=categorical_cols), encoded_cols], axis=1)
        elif encoding_method == "label" or encoding_method == "ordinal":
            if not categorical_cols.empty:
                for col in categorical_cols:
                    df[col] = LabelEncoder().fit_transform(df[col])
                
        # 5. Feature Scaling
        scaler_method = config.get("scaler")
        numeric_cols = df.select_dtypes(include=np.number).columns
        if scaler_method == "standard":
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        elif scaler_method == "minmax":
            scaler = MinMaxScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        elif scaler_method == "robust":
            scaler = RobustScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        # Save the processed dataset
        processed_name = f"processed_{dataset_name}"
        path = os.path.join(PROCESSED_DIR, processed_name)
        df.to_csv(path, index=False)
        return processed_name

    except Exception as e:
        raise Exception(f"Preprocessing failed: {str(e)}")
