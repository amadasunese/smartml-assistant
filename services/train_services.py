
import uuid
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

from services.data_services import load_dataset
from services.model_services import save_model



from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def train_model(dataset_name, config):
    df = load_dataset(dataset_name)

    target = config.get("target")
    if not target or target not in df.columns:
        raise ValueError("Target column not found in dataset")

    X = df.drop(columns=[target])
    y = df[target]

    # Configurable test size
    test_size = float(config.get("test_size", 0.2))

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Select algorithm
    algo = config.get("algorithm", "random_forest")
    if algo == "logistic_regression":
        clf = LogisticRegression(max_iter=1000)
    elif algo == "random_forest":
        clf = RandomForestClassifier(n_estimators=100)
    elif algo == "xgboost":
        clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    elif algo == "svm":
        clf = SVC(probability=True)
    elif algo == "decision_tree":
        clf = DecisionTreeClassifier()
    elif algo == "knn":
        clf = KNeighborsClassifier()
    elif algo == "neural_network":
        clf = MLPClassifier(max_iter=1000)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    # ✅ Preprocessing
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object", "category"]).columns

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Build pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    # Handle cross-validation
    cross_val = config.get("cross_validation", "none")
    cv_scores = None
    if cross_val and cross_val != "none":
        try:
            if cross_val == "stratified":
                from sklearn.model_selection import StratifiedKFold
                cv = StratifiedKFold(n_splits=5)
            else:
                cv = int(cross_val)
            cv_scores = cross_val_score(model, X, y, cv=cv)
        except Exception as e:
            cv_scores = f"Cross-validation failed: {str(e)}"
    
    # Train & evaluate
    # Train & evaluate
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    model_id = str(uuid.uuid4())

    # ✅ Store feature names for later use in prediction
    model.feature_names = X.columns.tolist()

    save_model(model, model_id)

    return model_id, {
        "accuracy": acc,
        "cross_validation_scores": cv_scores.tolist() if hasattr(cv_scores, "tolist") else cv_scores
    }

    # # Train & evaluate
    # model.fit(X_train, y_train)
    # preds = model.predict(X_test)
    # acc = accuracy_score(y_test, preds)

    # # ✅ Save model with feature info
    # model.feature_names_in_ = X.columns
    # model_id = str(uuid.uuid4())
    # save_model(model, model_id)

    # return model_id, {
    #     "accuracy": acc,
    #     "cross_validation_scores": cv_scores.tolist() if hasattr(cv_scores, "tolist") else cv_scores
    # }



def get_dataset_columns(dataset_name):
    df = load_dataset(dataset_name)
    return df.columns.tolist()