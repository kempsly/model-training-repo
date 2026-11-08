import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from bank_campaign_model_training import (
    load_data,
    encode_categorical,
    apply_bucketing,
    preprocess_features,
    train_model,
    get_classification_report,
)

@pytest.fixture
def dummy_data():
    data = {
        "age": [30, 40, 50, 60, 35, 45, 55, 65],
        "job": [
            "admin.", "technician", "self-employed", "management",
            "admin.", "technician", "management", "self-employed",
        ],
        "marital": [
            "married", "single", "married", "divorced",
            "single", "married", "divorced", "single",
        ],
        "education": [
            "university.degree", "basic.9y", "high.school", "basic.4y",
            "high.school", "university.degree", "basic.4y", "basic.9y",
        ],
        "default": ["no", "no", "no", "yes", "no", "no", "yes", "no"],
        "housing": ["yes", "yes", "no", "no", "yes", "no", "yes", "no"],
        "loan": ["no", "yes", "no", "yes", "no", "yes", "no", "yes"],
        "contact": [
            "cellular", "telephone", "cellular", "telephone",
            "cellular", "telephone", "cellular", "telephone",
        ],
        "month": ["may", "jun", "jul", "aug", "may", "jun", "jul", "aug"],
        "day_of_week": ["mon", "tue", "wed", "thu", "fri", "mon", "tue", "wed"],
        "duration": [200, 300, 400, 500, 250, 350, 450, 550],
        "campaign": [1, 2, 3, 4, 2, 3, 1, 4],
        "pdays": [20, 30, 40, 999, 10, 999, 50, 20],
        "previous": [1, 2, 3, 4, 0, 1, 2, 3],
        "poutcome": [
            "success", "failure", "nonexistent", "failure",
            "success", "nonexistent", "failure", "success",
        ],
        "emp.var.rate": [1.1, 2.2, 3.3, 4.4, 1.2, 2.3, 3.4, 4.5],
        "cons.price.idx": [90.1, 90.2, 90.3, 90.4, 90.5, 90.6, 90.7, 90.8],
        "cons.conf.idx": [-30.1, -30.2, -30.3, -30.4, -30.5, -30.6, -30.7, -30.8],
        "euribor3m": [1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5],
        "nr.employed": [5000, 6000, 7000, 8000, 5200, 6200, 7200, 8200],
        "y": ["yes", "no", "yes", "no", "yes", "no", "yes", "no"],
    }

    return pd.DataFrame(data)

CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "day_of_week", "poutcome",
]

@pytest.fixture
def prepared_data(dummy_data):
    df = encode_categorical(dummy_data, CATEGORICAL_COLS)
    df = apply_bucketing(df)
    return preprocess_features(df)

def test_preprocess_features(prepared_data):
    X, y = prepared_data

    assert X.shape == (8, 19)
    assert y.shape == (8,)
    assert set(y.unique()) == {0, 1}

def test_preprocess_features_invalid_target():
    df = pd.DataFrame({
        "age": [30, 40],
        "y": ["yes", "maybe"],
    })

    with pytest.raises(ValueError, match="Unexpected target values"):
        preprocess_features(df)

def test_data_loading(dummy_data, tmp_path):
    csv_path = tmp_path / "test_data.csv"
    # Use sep=";" to match load_data() configuration
    dummy_data.to_csv(csv_path, index=False, sep=";")

    loaded_df = load_data(str(csv_path))

    assert isinstance(loaded_df, pd.DataFrame)
    assert loaded_df.shape == dummy_data.shape
    assert list(loaded_df.columns) == list(dummy_data.columns)

def test_categorical_encoding(dummy_data):
    encoded_df = encode_categorical(dummy_data, CATEGORICAL_COLS)

    assert encoded_df.shape == dummy_data.shape
    assert all(
        pd.api.types.is_string_dtype(encoded_df[col])
        for col in CATEGORICAL_COLS
    )

def test_apply_bucketing(dummy_data):
    result = apply_bucketing(dummy_data)

    assert result["pdays_bucketed"].tolist() == [1, 1, 2, 0, 1, 0, 2, 1]
    assert "pdays" not in result.columns
    assert "duration" not in result.columns

def test_train_model(prepared_data):
    X, y = prepared_data

    model = train_model("xgboost", X, y)

    assert isinstance(model, Pipeline)
    predictions = model.predict(X)
    assert len(predictions) == len(y)

def test_train_model_unsupported_algorithm(prepared_data):
    X, y = prepared_data

    with pytest.raises(ValueError, match="Invalid model name"):
        train_model("unsupported_model", X, y)

def test_get_classification_report(prepared_data):
    X, y = prepared_data

    model = train_model("xgboost", X, y)
    report = get_classification_report(model, X, y)

    assert isinstance(report, dict)
    assert "0" in report
    assert "1" in report