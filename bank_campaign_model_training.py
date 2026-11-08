
import json
from datetime import datetime, timezone

import pandas as pd
from google.cloud import bigquery, storage
from joblib import dump
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

PROJECT_ID = "mlops-dev-513820"
BUCKET_NAME = "mlops-dev-513820-data"
BIGQUERY_TABLE = (
    "mlops-dev-513820.ml_ops.bank_campaign_model_metrics"
)
ARTIFACT_PATH = "bank_campaign_artifact/xgboost_model.joblib"

storage_client = storage.Client(project=PROJECT_ID)
bucket = storage_client.bucket(BUCKET_NAME)
bigquery_client = bigquery.Client(project=PROJECT_ID)


def load_data(path):
    return pd.read_csv(path, sep=";")


def encode_categorical(df, categorical_cols):
    df = df.copy()

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


def bucket_pdays(pdays):
    if pdays == 999:
        return 0
    elif pdays <= 30:
        return 1
    else:
        return 2


def apply_bucketing(df):
    df = df.copy()

    df["pdays_bucketed"] = df["pdays"].apply(bucket_pdays)
    df = df.drop(columns=["pdays", "duration"])

    return df


def preprocess_features(df):
    df = df.copy()

    y = df["y"].map({"yes": 1, "no": 0})

    if y.isna().any():
        raise ValueError("Unexpected target values in column 'y'.")

    X = df.drop(columns=["y"])

    return X, y


def train_model(model_name, X_train, y_train):
    categorical_cols = [
        "job", "marital", "education", "default", "housing",
        "loan", "contact", "month", "day_of_week", "poutcome",
    ]

    categorical_cols = [
        col for col in categorical_cols if col in X_train.columns
    ]
    numeric_cols = [
        col for col in X_train.columns if col not in categorical_cols
    ]

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
        ),
    ])

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ("categorical", categorical_pipeline, categorical_cols),
        ("numeric", numeric_pipeline, numeric_cols),
    ])

    if model_name == "logistic":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42)

    elif model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=42)

    elif model_name == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier()

    elif model_name == "xgboost":
        model = XGBClassifier(
            random_state=42,
            eval_metric="logloss",
        )

    else:
        raise ValueError(f"Invalid model name: {model_name}")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    pipeline.fit(X_train, y_train)

    return pipeline


def get_classification_report(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)

    return classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )


def save_model_artifact(model_name, pipeline):
    local_path = f"/tmp/{model_name}_model.joblib"

    dump(pipeline, local_path)

    blob = bucket.blob(ARTIFACT_PATH)
    blob.upload_from_filename(local_path)

    print(
        f"Model uploaded to gs://{BUCKET_NAME}/{ARTIFACT_PATH}"
    )


def load_model_artifact(file_name):
    blob = bucket.blob(file_name)
    local_path = f"/tmp/{file_name.split('/')[-1]}"

    blob.download_to_filename(local_path)

    from joblib import load
    return load(local_path)


def write_metrics_to_bigquery(algo_name, training_time, model_metrics):
    row = {
        "algo_name": algo_name,
        "training_time": training_time.isoformat(),
        "model_metrics": json.dumps(model_metrics),
    }

    errors = bigquery_client.insert_rows_json(
        BIGQUERY_TABLE,
        [row],
    )

    if errors:
        raise RuntimeError(
            f"BigQuery insert failed: {errors}"
        )

    print(f"Metrics inserted into {BIGQUERY_TABLE}")


def main():
    # Confirm this object exists in your Cloud Storage bucket first.
    input_data_path = (
        f"gs://{BUCKET_NAME}/bank_campaign_data/"
        "bank-campaign-training-data.csv"
    )

    model_name = "xgboost"

    df = load_data(input_data_path)

    categorical_cols = [
        "job", "marital", "education", "default", "housing",
        "loan", "contact", "month", "day_of_week", "poutcome",
    ]

    df = encode_categorical(df, categorical_cols)
    df = apply_bucketing(df)

    X, y = preprocess_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = train_model(model_name, X_train, y_train)

    metrics = get_classification_report(
        pipeline,
        X_test,
        y_test,
    )

    training_time = datetime.now(timezone.utc)

    write_metrics_to_bigquery(
        model_name,
        training_time,
        metrics,
    )
    

    save_model_artifact(model_name, pipeline)


if __name__ == "__main__":
    main()
