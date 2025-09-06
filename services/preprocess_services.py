import os
import pandas as pd
from services.data_services import load_dataset

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import boxcox
import scipy.stats as stats
from services.data_services import PROCESSED_DIR

# PROCESSED_DIR = "uploaded_data"

# def preprocess_pipeline(dataset_name, config):
#     try:
#         df = load_dataset(dataset_name)

#         # Standardize Missing Values
#         df.replace(["", " ", "null", "N/A", "na", "-"], np.nan, inplace=True)
        
#         # 2. Missing Values Strategy
#         missing_strategy = config.get("missing_strategy")
#         if missing_strategy == "mean" or missing_strategy == "median" or missing_strategy == "most_frequent":
#             imputer = SimpleImputer(strategy=missing_strategy)
#             # Identify numeric and categorical columns for imputation
#             numeric_cols = df.select_dtypes(include=np.number).columns
#             categorical_cols = df.select_dtypes(exclude=np.number).columns
            
#             if not numeric_cols.empty:
#                 df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            
#             if not categorical_cols.empty and missing_strategy == "most_frequent":
#                 imputer_cat = SimpleImputer(strategy="most_frequent")
#                 df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

#         elif missing_strategy == "drop":
#             df = df.dropna()
#         elif missing_strategy == "interpolate":
#             numeric_cols = df.select_dtypes(include=np.number).columns
#             df[numeric_cols] = df[numeric_cols].interpolate()
        
#         # 3. Outlier Handling
#         outlier_method = config.get("outlier_method")
#         numeric_cols = df.select_dtypes(include=np.number).columns
#         if outlier_method == "clip":
#             for col in numeric_cols:
#                 lower_bound = df[col].quantile(0.05)
#                 upper_bound = df[col].quantile(0.95)
#                 df[col] = np.clip(df[col], lower_bound, upper_bound)
#         elif outlier_method == "remove":
#             for col in numeric_cols:
#                 Q1 = df[col].quantile(0.25)
#                 Q3 = df[col].quantile(0.75)
#                 IQR = Q3 - Q1
#                 lower_bound = Q1 - 1.5 * IQR
#                 upper_bound = Q3 + 1.5 * IQR
#                 df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

#         # 4. Categorical Encoding
#         encoding_method = config.get("encoding_method")
#         categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
#         if encoding_method == "onehot":
#             if not categorical_cols.empty:
#                 encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
#                 encoded_cols = pd.DataFrame(encoder.fit_transform(df[categorical_cols]))
#                 encoded_cols.columns = encoder.get_feature_names_out(categorical_cols)
#                 df = pd.concat([df.drop(columns=categorical_cols), encoded_cols], axis=1)
#         elif encoding_method == "label" or encoding_method == "ordinal":
#             if not categorical_cols.empty:
#                 for col in categorical_cols:
#                     df[col] = LabelEncoder().fit_transform(df[col])
                
#         # 5. Feature Scaling
#         scaler_method = config.get("scaler")
#         numeric_cols = df.select_dtypes(include=np.number).columns
#         if scaler_method == "standard":
#             scaler = StandardScaler()
#             df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
#         elif scaler_method == "minmax":
#             scaler = MinMaxScaler()
#             df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
#         elif scaler_method == "robust":
#             scaler = RobustScaler()
#             df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
#         # Save the processed dataset
#         processed_name = f"processed_{dataset_name}"
#         path = os.path.join(PROCESSED_DIR, processed_name)
#         df.to_csv(path, index=False)
#         return processed_name

#     except Exception as e:
#         raise Exception(f"Preprocessing failed: {str(e)}")


# def preprocess_pipeline(dataset_name, config):
#     try:
#         df = load_dataset(dataset_name)
        
#         # Store original column names and dtypes for reference
#         original_columns = df.columns.tolist()
#         original_dtypes = df.dtypes.to_dict()
        
#         # Standardize Missing Values
#         df.replace(["", " ", "null", "N/A", "na", "-", "unknown", "NaN"], np.nan, inplace=True)
        
#         # Get target column if specified (for target encoding and imbalance handling)
#         target_column = config.get("target_column")
        
#         # 1. Handle Missing Values with column-specific options
#         missing_strategy = config.get("missing_strategy")
#         missing_columns = config.get("missing_columns", [])  # List of columns to apply to
        
#         if missing_strategy and missing_strategy != "none":
#             # If no specific columns selected, apply to all appropriate columns
#             if not missing_columns:
#                 numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
#                 categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
#             else:
#                 # Apply only to selected columns
#                 numeric_cols = [col for col in missing_columns if col in df.select_dtypes(include=np.number).columns]
#                 categorical_cols = [col for col in missing_columns if col in df.select_dtypes(exclude=np.number).columns]
            
#             # Add missing indicator if requested
#             if config.get("missing_indicator", False):
#                 for col in numeric_cols + categorical_cols:
#                     if df[col].isnull().any():
#                         df[f"{col}_missing"] = df[col].isnull().astype(int)
            
#             # Apply the selected strategy
#             if missing_strategy == "mean" or missing_strategy == "median":
#                 if numeric_cols:
#                     imputer = SimpleImputer(strategy=missing_strategy)
#                     df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
#             elif missing_strategy == "most_frequent":
#                 if numeric_cols:
#                     imputer_num = SimpleImputer(strategy="most_frequent")
#                     df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])
#                 if categorical_cols:
#                     imputer_cat = SimpleImputer(strategy="most_frequent")
#                     df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])
#             elif missing_strategy == "drop":
#                 df = df.dropna(subset=numeric_cols + categorical_cols)
#             elif missing_strategy == "interpolate":
#                 if numeric_cols:
#                     df[numeric_cols] = df[numeric_cols].interpolate()
#                     # For any remaining missing values after interpolation, use mean
#                     for col in numeric_cols:
#                         if df[col].isnull().any():
#                             df[col].fillna(df[col].mean(), inplace=True)
        
#         # 2. Handle Outliers with advanced options
#         outlier_method = config.get("outlier_method")
#         outlier_columns = config.get("outlier_columns", [])
#         outlier_detection = config.get("outlier_detection", "iqr")
#         outlier_transform = config.get("outlier_transform_method", "log")
        
#         if outlier_method and outlier_method != "none":
#             # If no specific columns selected, apply to all numeric columns
#             if not outlier_columns:
#                 outlier_columns = df.select_dtypes(include=np.number).columns.tolist()
            
#             for col in outlier_columns:
#                 if col not in df.columns:
#                     continue
                    
#                 if outlier_method == "clip":
#                     if outlier_detection == "iqr":
#                         Q1 = df[col].quantile(0.25)
#                         Q3 = df[col].quantile(0.75)
#                         IQR = Q3 - Q1
#                         lower_bound = Q1 - 1.5 * IQR
#                         upper_bound = Q3 + 1.5 * IQR
#                     elif outlier_detection == "zscore":
#                         z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
#                         lower_bound = df[col].min()  # Not typically used with z-score
#                         upper_bound = df[col][z_scores < 3].max() if len(df[col][z_scores < 3]) > 0 else df[col].max()
                    
#                     df[col] = np.clip(df[col], lower_bound, upper_bound)
                
#                 elif outlier_method == "remove":
#                     if outlier_detection == "iqr":
#                         Q1 = df[col].quantile(0.25)
#                         Q3 = df[col].quantile(0.75)
#                         IQR = Q3 - Q1
#                         lower_bound = Q1 - 1.5 * IQR
#                         upper_bound = Q3 + 1.5 * IQR
#                         df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
#                     elif outlier_detection == "zscore":
#                         z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
#                         df = df[z_scores < 3]
                
#                 elif outlier_method == "transform":
#                     if outlier_transform == "log":
#                         # Add 1 to handle zeros
#                         df[col] = np.log1p(df[col])
#                     elif outlier_transform == "sqrt":
#                         df[col] = np.sqrt(df[col])
#                     elif outlier_transform == "boxcox":
#                         # Box-Cox requires positive values
#                         if df[col].min() <= 0:
#                             # Shift to positive
#                             df[col] = df[col] - df[col].min() + 1
#                         df[col], _ = boxcox(df[col])
        
#         # 3. Handle Categorical Encoding with advanced options
#         encoding_method = config.get("encoding_method")
#         encoding_columns = config.get("encoding_columns", [])
        
#         if encoding_method and encoding_method != "none":
#             # If no specific columns selected, apply to all categorical columns
#             if not encoding_columns:
#                 encoding_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
#             if encoding_method == "onehot":
#                 encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
#                 encoded_cols = pd.DataFrame(encoder.fit_transform(df[encoding_columns]))
#                 encoded_cols.columns = encoder.get_feature_names_out(encoding_columns)
#                 df = pd.concat([df.drop(columns=encoding_columns), encoded_cols], axis=1)
            
#             elif encoding_method == "label" or encoding_method == "ordinal":
#                 for col in encoding_columns:
#                     df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            
#             elif encoding_method == "target" and target_column:
#                 for col in encoding_columns:
#                     # Calculate mean target value for each category
#                     target_means = df.groupby(col)[target_column].mean()
#                     df[col] = df[col].map(target_means)
#                     # For unseen categories, use global mean
#                     df[col].fillna(df[target_column].mean(), inplace=True)
            
#             elif encoding_method == "frequency":
#                 for col in encoding_columns:
#                     freq = df[col].value_counts(normalize=True)
#                     df[col] = df[col].map(freq)
        
#         # 4. Feature Scaling with column-specific options
#         scaler_method = config.get("scaler")
#         scaling_columns = config.get("scaling_columns", [])
        
#         if scaler_method and scaler_method != "none":
#             # If no specific columns selected, apply to all numeric columns
#             if not scaling_columns:
#                 scaling_columns = df.select_dtypes(include=np.number).columns.tolist()
            
#             # Don't scale the target column if it's specified
#             if target_column and target_column in scaling_columns:
#                 scaling_columns.remove(target_column)
            
#             if scaler_method == "standard":
#                 scaler = StandardScaler()
#                 df[scaling_columns] = scaler.fit_transform(df[scaling_columns])
#             elif scaler_method == "minmax":
#                 scaler = MinMaxScaler()
#                 df[scaling_columns] = scaler.fit_transform(df[scaling_columns])
#             elif scaler_method == "robust":
#                 scaler = RobustScaler()
#                 df[scaling_columns] = scaler.fit_transform(df[scaling_columns])
#             elif scaler_method == "normalize":
#                 df[scaling_columns] = df[scaling_columns].apply(
#                     lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0
#                 )
        
#         # 5. Feature Selection
#         feature_selection_method = config.get("feature_selection")
#         if feature_selection_method and feature_selection_method != "none" and target_column:
#             X = df.drop(columns=[target_column])
#             y = df[target_column]
            
#             if feature_selection_method == "variance":
#                 selector = VarianceThreshold(threshold=0.1)
#                 X_selected = selector.fit_transform(X)
#                 selected_columns = X.columns[selector.get_support()]
#                 df = pd.concat([pd.DataFrame(X_selected, columns=selected_columns), y], axis=1)
            
#             elif feature_selection_method == "kbest":
#                 k = min(10, len(X.columns))  # Select top 10 features or all if fewer than 10
#                 selector = SelectKBest(score_func=f_classif, k=k)
#                 X_selected = selector.fit_transform(X, y)
#                 selected_columns = X.columns[selector.get_support()]
#                 df = pd.concat([pd.DataFrame(X_selected, columns=selected_columns), y], axis=1)
            
#             elif feature_selection_method == "rfe":
#                 estimator = LogisticRegression(max_iter=1000)
#                 selector = RFE(estimator, n_features_to_select=10, step=1)
#                 X_selected = selector.fit_transform(X, y)
#                 selected_columns = X.columns[selector.get_support()]
#                 df = pd.concat([pd.DataFrame(X_selected, columns=selected_columns), y], axis=1)
        
#         # 6. Handle Class Imbalance if target is specified
#         imbalance_method = config.get("imbalance_method")
#         if imbalance_method and imbalance_method != "none" and target_column:
#             X = df.drop(columns=[target_column])
#             y = df[target_column]
            
#             if imbalance_method == "smote":
#                 smote = SMOTE(random_state=config.get("random_state", 42))
#                 X_resampled, y_resampled = smote.fit_resample(X, y)
#                 df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), 
#                                pd.Series(y_resampled, name=target_column)], axis=1)
            
#             elif imbalance_method == "undersample":
#                 rus = RandomUnderSampler(random_state=config.get("random_state", 42))
#                 X_resampled, y_resampled = rus.fit_resample(X, y)
#                 df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), 
#                                pd.Series(y_resampled, name=target_column)], axis=1)
            
#             elif imbalance_method == "oversample":
#                 ros = RandomOverSampler(random_state=config.get("random_state", 42))
#                 X_resampled, y_resampled = ros.fit_resample(X, y)
#                 df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), 
#                                pd.Series(y_resampled, name=target_column)], axis=1)
        
#         # 7. Train-Test Split if requested
#         test_size = config.get("train_test_split")
#         random_state = config.get("random_state", 42)
        
#         if test_size and test_size != "none" and target_column:
#             test_size = float(test_size)
#             X = df.drop(columns=[target_column])
#             y = df[target_column]
            
#             X_train, X_test, y_train, y_test = train_test_split(
#                 X, y, test_size=test_size, random_state=random_state, stratify=y
#             )
            
#             # Save both train and test datasets
#             train_df = pd.concat([X_train, y_train], axis=1)
#             test_df = pd.concat([X_test, y_test], axis=1)
            
#             train_name = f"train_processed_{dataset_name}"
#             test_name = f"test_processed_{dataset_name}"
            
#             train_path = os.path.join(PROCESSED_DIR, train_name)
#             test_path = os.path.join(PROCESSED_DIR, test_name)
            
#             train_df.to_csv(train_path, index=False)
#             test_df.to_csv(test_path, index=False)
            
#             return train_name, test_name
        
#         # Save the processed dataset
#         processed_name = f"processed_{dataset_name}"
#         path = os.path.join(PROCESSED_DIR, processed_name)
#         df.to_csv(path, index=False)
        
#         return processed_name

#     except Exception as e:
#         raise Exception(f"Preprocessing failed: {str(e)}")



# def preprocess_pipeline(dataset_name, config):
#     try:
#         df = load_dataset(dataset_name)
        
#         # Store original column names and dtypes for reference
#         original_columns = df.columns.tolist()
#         original_dtypes = df.dtypes.to_dict()
        
#         # Standardize Missing Values
#         df.replace(["", " ", "null", "N/A", "na", "-", "unknown", "NaN"], np.nan, inplace=True)
        
#         # Get target column if specified (for target encoding and imbalance handling)
#         target_column = config.get("target_column")
        
#         # 1. Handle Missing Values with column-specific options
#         missing_strategy = config.get("missing_strategy")
#         missing_columns = config.get("missing_columns", [])  # List of columns to apply to
        
#         if missing_strategy and missing_strategy != "none":
#             # If no specific columns selected, apply to all appropriate columns
#             if not missing_columns:
#                 numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
#                 categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
#             else:
#                 # Apply only to selected columns
#                 numeric_cols = [col for col in missing_columns if col in df.select_dtypes(include=np.number).columns]
#                 categorical_cols = [col for col in missing_columns if col in df.select_dtypes(exclude=np.number).columns]
            
#             # Add missing indicator if requested
#             if config.get("missing_indicator", False):
#                 for col in numeric_cols + categorical_cols:
#                     if df[col].isnull().any():
#                         df[f"{col}_missing"] = df[col].isnull().astype(int)
            
#             # Apply the selected strategy
#             if missing_strategy == "mean" or missing_strategy == "median":
#                 if numeric_cols:
#                     imputer = SimpleImputer(strategy=missing_strategy)
#                     df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
#             elif missing_strategy == "most_frequent":
#                 if numeric_cols:
#                     imputer_num = SimpleImputer(strategy="most_frequent")
#                     df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])
#                 if categorical_cols:
#                     imputer_cat = SimpleImputer(strategy="most_frequent")
#                     df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])
#             elif missing_strategy == "drop":
#                 df = df.dropna(subset=numeric_cols + categorical_cols)
#             elif missing_strategy == "interpolate":
#                 if numeric_cols:
#                     df[numeric_cols] = df[numeric_cols].interpolate()
#                     # For any remaining missing values after interpolation, use mean
#                     for col in numeric_cols:
#                         if df[col].isnull().any():
#                             df[col].fillna(df[col].mean(), inplace=True)
        
#         # 2. Handle Outliers with advanced options
#         outlier_method = config.get("outlier_method")
#         outlier_columns = config.get("outlier_columns", [])
#         outlier_detection = config.get("outlier_detection", "iqr")
#         outlier_transform = config.get("outlier_transform_method", "log")
        
#         if outlier_method and outlier_method != "none":
#             # If no specific columns selected, apply to all numeric columns
#             if not outlier_columns:
#                 outlier_columns = df.select_dtypes(include=np.number).columns.tolist()
            
#             for col in outlier_columns:
#                 if col not in df.columns:
#                     continue
                    
#                 if outlier_method == "clip":
#                     if outlier_detection == "iqr":
#                         Q1 = df[col].quantile(0.25)
#                         Q3 = df[col].quantile(0.75)
#                         IQR = Q3 - Q1
#                         lower_bound = Q1 - 1.5 * IQR
#                         upper_bound = Q3 + 1.5 * IQR
#                     elif outlier_detection == "zscore":
#                         z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
#                         lower_bound = df[col].min()  # Not typically used with z-score
#                         upper_bound = df[col][z_scores < 3].max() if len(df[col][z_scores < 3]) > 0 else df[col].max()
                    
#                     df[col] = np.clip(df[col], lower_bound, upper_bound)
                
#                 elif outlier_method == "remove":
#                     if outlier_detection == "iqr":
#                         Q1 = df[col].quantile(0.25)
#                         Q3 = df[col].quantile(0.75)
#                         IQR = Q3 - Q1
#                         lower_bound = Q1 - 1.5 * IQR
#                         upper_bound = Q3 + 1.5 * IQR
#                         df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
#                     elif outlier_detection == "zscore":
#                         z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
#                         df = df[z_scores < 3]
                
#                 elif outlier_method == "transform":
#                     if outlier_transform == "log":
#                         # Add 1 to handle zeros
#                         df[col] = np.log1p(df[col])
#                     elif outlier_transform == "sqrt":
#                         df[col] = np.sqrt(df[col])
#                     elif outlier_transform == "boxcox":
#                         # Box-Cox requires positive values
#                         if df[col].min() <= 0:
#                             # Shift to positive
#                             df[col] = df[col] - df[col].min() + 1
#                         df[col], _ = boxcox(df[col])
        
#         # 3. Handle Categorical Encoding with advanced options
#         encoding_method = config.get("encoding_method")
#         encoding_columns = config.get("encoding_columns", [])
        
#         if encoding_method and encoding_method != "none":
#             # If no specific columns selected, apply to all categorical columns
#             if not encoding_columns:
#                 encoding_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
#             if encoding_method == "onehot":
#                 encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
#                 encoded_cols = pd.DataFrame(encoder.fit_transform(df[encoding_columns]))
#                 encoded_cols.columns = encoder.get_feature_names_out(encoding_columns)
#                 df = pd.concat([df.drop(columns=encoding_columns), encoded_cols], axis=1)
            
#             elif encoding_method == "label" or encoding_method == "ordinal":
#                 for col in encoding_columns:
#                     df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            
#             elif encoding_method == "target" and target_column:
#                 for col in encoding_columns:
#                     # Calculate mean target value for each category
#                     target_means = df.groupby(col)[target_column].mean()
#                     df[col] = df[col].map(target_means)
#                     # For unseen categories, use global mean
#                     df[col].fillna(df[target_column].mean(), inplace=True)
            
#             elif encoding_method == "frequency":
#                 for col in encoding_columns:
#                     freq = df[col].value_counts(normalize=True)
#                     df[col] = df[col].map(freq)
        
#         # 4. Feature Scaling with column-specific options
#         scaler_method = config.get("scaler")
#         scaling_columns = config.get("scaling_columns", [])
        
#         if scaler_method and scaler_method != "none":
#             # If no specific columns selected, apply to all numeric columns
#             if not scaling_columns:
#                 scaling_columns = df.select_dtypes(include=np.number).columns.tolist()
            
#             # Don't scale the target column if it's specified
#             if target_column and target_column in scaling_columns:
#                 scaling_columns.remove(target_column)
            
#             if scaler_method == "standard":
#                 scaler = StandardScaler()
#                 df[scaling_columns] = scaler.fit_transform(df[scaling_columns])
#             elif scaler_method == "minmax":
#                 scaler = MinMaxScaler()
#                 df[scaling_columns] = scaler.fit_transform(df[scaling_columns])
#             elif scaler_method == "robust":
#                 scaler = RobustScaler()
#                 df[scaling_columns] = scaler.fit_transform(df[scaling_columns])
#             elif scaler_method == "normalize":
#                 df[scaling_columns] = df[scaling_columns].apply(
#                     lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0
#                 )
        
        
#         # Handle Class Imbalance if target is specified
#         imbalance_method = config.get("imbalance_method")
#         if imbalance_method and imbalance_method != "none" and target_column:
#             X = df.drop(columns=[target_column])
#             y = df[target_column]
            
#             if imbalance_method == "smote":
#                 smote = SMOTE(random_state=config.get("random_state", 42))
#                 X_resampled, y_resampled = smote.fit_resample(X, y)
#                 df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), 
#                                pd.Series(y_resampled, name=target_column)], axis=1)
            
#             elif imbalance_method == "undersample":
#                 rus = RandomUnderSampler(random_state=config.get("random_state", 42))
#                 X_resampled, y_resampled = rus.fit_resample(X, y)
#                 df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), 
#                                pd.Series(y_resampled, name=target_column)], axis=1)
            
#             elif imbalance_method == "oversample":
#                 ros = RandomOverSampler(random_state=config.get("random_state", 42))
#                 X_resampled, y_resampled = ros.fit_resample(X, y)
#                 df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), 
#                                pd.Series(y_resampled, name=target_column)], axis=1)
        
        
        
#         # Save the processed dataset
#         processed_name = f"processed_{dataset_name}"
#         path = os.path.join(PROCESSED_DIR, processed_name)
#         df.to_csv(path, index=False)
        
#         return processed_name

#     except Exception as e:
#         raise Exception(f"Preprocessing failed: {str(e)}")



def preprocess_pipeline(dataset_name, config):
    try:
        df = load_dataset(dataset_name)
        
        # Store original column names and dtypes for reference
        original_columns = df.columns.tolist()
        original_dtypes = df.dtypes.to_dict()
        
        # Standardize Missing Values
        df.replace(["", " ", "null", "N/A", "na", "-", "unknown", "NaN"], np.nan, inplace=True)
        
        # Get target column if specified (for target encoding and imbalance handling)
        target_column = config.get("target_column")

        # 🔹 0. Drop Columns if specified
        drop_columns = config.get("drop_columns", [])
        if drop_columns:
            df = df.drop(columns=[c for c in drop_columns if c in df.columns], errors="ignore")
        
        # 1. Handle Missing Values with column-specific options
        missing_strategy = config.get("missing_strategy")
        missing_columns = config.get("missing_columns", [])  # List of columns to apply to
        
        if missing_strategy and missing_strategy != "none":
            # If no specific columns selected, apply to all appropriate columns
            if not missing_columns:
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
            else:
                # Apply only to selected columns
                numeric_cols = [col for col in missing_columns if col in df.select_dtypes(include=np.number).columns]
                categorical_cols = [col for col in missing_columns if col in df.select_dtypes(exclude=np.number).columns]
            
            # Add missing indicator if requested
            if config.get("missing_indicator", False):
                for col in numeric_cols + categorical_cols:
                    if df[col].isnull().any():
                        df[f"{col}_missing"] = df[col].isnull().astype(int)
            
            # Apply the selected strategy
            if missing_strategy == "mean" or missing_strategy == "median":
                if numeric_cols:
                    imputer = SimpleImputer(strategy=missing_strategy)
                    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            elif missing_strategy == "most_frequent":
                if numeric_cols:
                    imputer_num = SimpleImputer(strategy="most_frequent")
                    df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])
                if categorical_cols:
                    imputer_cat = SimpleImputer(strategy="most_frequent")
                    df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])
            elif missing_strategy == "drop":
                df = df.dropna(subset=numeric_cols + categorical_cols)
            elif missing_strategy == "interpolate":
                if numeric_cols:
                    df[numeric_cols] = df[numeric_cols].interpolate()
                    # For any remaining missing values after interpolation, use mean
                    for col in numeric_cols:
                        if df[col].isnull().any():
                            df[col].fillna(df[col].mean(), inplace=True)
        
        # 2. Handle Outliers with advanced options
        outlier_method = config.get("outlier_method")
        outlier_columns = config.get("outlier_columns", [])
        outlier_detection = config.get("outlier_detection", "iqr")
        outlier_transform = config.get("outlier_transform_method", "log")
        
        if outlier_method and outlier_method != "none":
            if not outlier_columns:
                outlier_columns = df.select_dtypes(include=np.number).columns.tolist()
            
            for col in outlier_columns:
                if col not in df.columns:
                    continue
                    
                if outlier_method == "clip":
                    if outlier_detection == "iqr":
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                    elif outlier_detection == "zscore":
                        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                        lower_bound = df[col].min()
                        upper_bound = df[col][z_scores < 3].max() if len(df[col][z_scores < 3]) > 0 else df[col].max()
                    
                    df[col] = np.clip(df[col], lower_bound, upper_bound)
                
                elif outlier_method == "remove":
                    if outlier_detection == "iqr":
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                    elif outlier_detection == "zscore":
                        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                        df = df[z_scores < 3]
                
                elif outlier_method == "transform":
                    if outlier_transform == "log":
                        df[col] = np.log1p(df[col])
                    elif outlier_transform == "sqrt":
                        df[col] = np.sqrt(df[col])
                    elif outlier_transform == "boxcox":
                        if df[col].min() <= 0:
                            df[col] = df[col] - df[col].min() + 1
                        df[col], _ = boxcox(df[col])
        
        # 2b. Handle Skewness
        skewness_method = config.get("skewness_method", "none")
        skewness_columns = config.get("skewness_columns", [])

        if skewness_method != "none":
            if not skewness_columns:
                skewness_columns = df.select_dtypes(include=np.number).columns.tolist()
                if target_column and target_column in skewness_columns:
                    skewness_columns.remove(target_column)
            
            for col in skewness_columns:
                if col not in df.columns:
                    continue
                if skewness_method == "log":
                    df[col] = np.log1p(df[col] - df[col].min() + 1)
                elif skewness_method == "sqrt":
                    df[col] = np.sqrt(df[col] - df[col].min())
                elif skewness_method == "boxcox":
                    series = df[col].dropna()
                    if series.min() <= 0:
                        series = series - series.min() + 1
                    transformed, _ = boxcox(series)
                    df.loc[series.index, col] = transformed
                elif skewness_method == "yeojohnson":
                    series = df[col].dropna()
                    transformed, _ = stats.yeojohnson(series)
                    df.loc[series.index, col] = transformed
                    
        
        # 3. Encoding
        encoding_method = config.get("encoding_method")
        encoding_columns = config.get("encoding_columns", [])
        
        if encoding_method and encoding_method != "none":
            if not encoding_columns:
                encoding_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if encoding_method == "onehot":
                encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                encoded_cols = pd.DataFrame(encoder.fit_transform(df[encoding_columns]))
                encoded_cols.columns = encoder.get_feature_names_out(encoding_columns)
                df = pd.concat([df.drop(columns=encoding_columns), encoded_cols], axis=1)
            
            elif encoding_method in ("label", "ordinal"):
                for col in encoding_columns:
                    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            
            elif encoding_method == "target" and target_column:
                for col in encoding_columns:
                    target_means = df.groupby(col)[target_column].mean()
                    df[col] = df[col].map(target_means)
                    df[col].fillna(df[target_column].mean(), inplace=True)
            
            elif encoding_method == "frequency":
                for col in encoding_columns:
                    freq = df[col].value_counts(normalize=True)
                    df[col] = df[col].map(freq)
        
        # 4. Scaling
        scaler_method = config.get("scaler")
        scaling_columns = config.get("scaling_columns", [])
        
        if scaler_method and scaler_method != "none":
            if not scaling_columns:
                scaling_columns = df.select_dtypes(include=np.number).columns.tolist()
            if target_column and target_column in scaling_columns:
                scaling_columns.remove(target_column)
            
            if scaler_method == "standard":
                scaler = StandardScaler()
                df[scaling_columns] = scaler.fit_transform(df[scaling_columns])
            elif scaler_method == "minmax":
                scaler = MinMaxScaler()
                df[scaling_columns] = scaler.fit_transform(df[scaling_columns])
            elif scaler_method == "robust":
                scaler = RobustScaler()
                df[scaling_columns] = scaler.fit_transform(df[scaling_columns])
            elif scaler_method == "normalize":
                df[scaling_columns] = df[scaling_columns].apply(
                    lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0
                )
        
        # 5. Handle Class Imbalance
        imbalance_method = config.get("imbalance_method")
        if imbalance_method and imbalance_method != "none" and target_column:
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            if imbalance_method == "smote":
                smote = SMOTE(random_state=config.get("random_state", 42))
                X_resampled, y_resampled = smote.fit_resample(X, y)
                df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), 
                               pd.Series(y_resampled, name=target_column)], axis=1)
            elif imbalance_method == "undersample":
                rus = RandomUnderSampler(random_state=config.get("random_state", 42))
                X_resampled, y_resampled = rus.fit_resample(X, y)
                df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), 
                               pd.Series(y_resampled, name=target_column)], axis=1)
            elif imbalance_method == "oversample":
                ros = RandomOverSampler(random_state=config.get("random_state", 42))
                X_resampled, y_resampled = ros.fit_resample(X, y)
                df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), 
                               pd.Series(y_resampled, name=target_column)], axis=1)
        
        # 0b. Handle Duplicates
        duplicate_handling = config.get("duplicate_handling", "none")
        if duplicate_handling != "none":
            if duplicate_handling == "drop":
                df = df.drop_duplicates()
            elif duplicate_handling == "keep_first":
                df = df.drop_duplicates(keep="first")
            elif duplicate_handling == "keep_last":
                df = df.drop_duplicates(keep="last")
        
       # Save the processed dataset
        processed_name = f"processed_{dataset_name}"
        os.makedirs(PROCESSED_DIR, exist_ok=True)  # ensure dir exists
        path = os.path.join(PROCESSED_DIR, processed_name)
        df.to_csv(path, index=False)
        
        return processed_name

    except Exception as e:
        raise Exception(f"Preprocessing failed: {str(e)}")
