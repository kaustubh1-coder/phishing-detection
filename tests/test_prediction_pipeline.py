import os
import json
import pytest
import numpy as np
import pandas as pd
from networksecurity.constants.training_pipeline import (
    CANONICAL_FEATURE_NAMES,
    LABEL_MAPPING,
    RAW_TARGET_MAPPING,
    TARGET_COLUMN,
)
from networksecurity.pipeline.batch_prediction import PredictionPipeline
from networksecurity.utils.url_feature_extractor import extract_url_features
from networksecurity.utils.main_utils.utils import load_object


def test_canonical_feature_names():
    assert len(CANONICAL_FEATURE_NAMES) == 30
    assert "having_IP_Address" in CANONICAL_FEATURE_NAMES
    assert "Prefix_Suffix" in CANONICAL_FEATURE_NAMES
    assert "URL_of_Anchor" in CANONICAL_FEATURE_NAMES
    assert "SSLfinal_State" in CANONICAL_FEATURE_NAMES
    assert TARGET_COLUMN not in CANONICAL_FEATURE_NAMES


def test_label_mapping():
    assert LABEL_MAPPING[0] == "Phishing"
    assert LABEL_MAPPING[1] == "Legitimate"
    assert RAW_TARGET_MAPPING[-1] == 0
    assert RAW_TARGET_MAPPING[1] == 1


def test_prediction_pipeline_validation():
    pipeline = PredictionPipeline()

    # Missing columns
    with pytest.raises(ValueError) as excinfo:
        pipeline.predict(pd.DataFrame([{"having_IP_Address": 1}]))
    assert "Missing features" in str(excinfo.value)
    assert "Expected features: 30" in str(excinfo.value)

    # Extra columns
    extra_row = {c: 1 for c in CANONICAL_FEATURE_NAMES}
    extra_row["bogus_column"] = 42
    with pytest.raises(ValueError) as excinfo:
        pipeline.predict(pd.DataFrame([extra_row]))
    assert "Extra features: ['bogus_column']" in str(excinfo.value)


def test_suspicious_url_prediction():
    pipeline = PredictionPipeline()
    url = "http://secure-login.example.test@192.168.1.10/verify-account"
    feature_df, features, metadata = extract_url_features(url)

    assert list(feature_df.columns) == CANONICAL_FEATURE_NAMES
    assert len(feature_df.columns) == 30

    res = pipeline.predict(feature_df)
    y_pred = res["predictions"]
    y_proba = res["probabilities"]

    assert y_pred[0] == 0  # 0 is Phishing
    assert y_proba[0][0] >= 0.75  # High confidence phishing
    assert LABEL_MAPPING[int(y_pred[0])] == "Phishing"


def test_legitimate_url_prediction():
    pipeline = PredictionPipeline()
    url = "https://www.google.com"
    feature_df, features, metadata = extract_url_features(url)

    res = pipeline.predict(feature_df)
    y_pred = res["predictions"]
    y_proba = res["probabilities"]

    assert y_pred[0] == 1  # 1 is Legitimate
    assert y_proba[0][1] >= 0.75  # High confidence legitimate
    assert LABEL_MAPPING[int(y_pred[0])] == "Legitimate"


def test_batch_prediction_consistency():
    pipeline = PredictionPipeline()
    raw_df = pd.read_csv("Network_Data/phisingData.csv").head(10)
    res = pipeline.predict(raw_df)

    assert len(res["predictions"]) == 10
    assert len(res["probabilities"]) == 10
    assert set(res["predictions"]).issubset({0, 1})


def test_model_metadata_verification():
    import json
    import os

    metadata_path = os.path.join("final_model", "model_metadata.json")
    assert os.path.exists(metadata_path), "final_model/model_metadata.json must exist"

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["model_name"] == "Random Forest"
    assert meta["feature_count"] == 30
    assert len(meta["features"]) == 30
    assert meta["features"] == CANONICAL_FEATURE_NAMES
    assert meta["label_mapping"] == {"0": "Phishing", "1": "Legitimate"}
    assert "training_date" in meta
    assert meta["test_accuracy"] > 0.90
    assert meta["test_f1"] > 0.90
    assert "confusion_matrix" in meta
    assert "matrix" in meta["confusion_matrix"]
    cm = meta["confusion_matrix"]["matrix"]
    assert len(cm) == 2 and len(cm[0]) == 2


@pytest.mark.parametrize(
    "url",
    [
        "https://www.google.com",
        "http://192.168.1.10/login",
        "http://secure-login.example.test@192.168.1.10/verify-account",
        "https://example.com",
        "http://example.com/login?redirect=http://evil.example.test",
    ],
)
def test_specific_live_urls_production_consistency(url: str):
    """
    Verifies that for every required URL:
    - feature extraction succeeds
    - feature count is correct (30)
    - feature names are correct
    - feature order is correct
    - no NaN/inf values reach the model
    - prediction executes successfully with valid probabilities
    """
    pipeline = PredictionPipeline()
    network_model = pipeline.load_pipeline()

    # 1. Feature extraction succeeds
    df, features, metadata = extract_url_features(url)
    assert df is not None
    assert len(df) == 1
    assert isinstance(features, dict)
    assert isinstance(metadata, dict)
    assert "feature_categories" in metadata
    assert "feature_availability" in metadata

    # 2. Feature count is correct
    assert len(df.columns) == len(CANONICAL_FEATURE_NAMES)
    assert len(df.columns) == 30

    # 3. Feature names are correct
    assert list(df.columns) == CANONICAL_FEATURE_NAMES

    # 4. Feature order is correct
    assert list(df.columns) == CANONICAL_FEATURE_NAMES

    # 5. No NaN/inf values reach the model
    transformed = network_model.preprocessor.transform(df)
    assert not np.isnan(transformed).any(), f"NaN found in transformed matrix for {url}"
    assert not np.isinf(transformed).any(), f"Inf found in transformed matrix for {url}"
    assert transformed.shape == (1, 30)

    # 6. Prediction executes successfully
    res = pipeline.predict(df)
    assert "predictions" in res
    assert "probabilities" in res
    assert len(res["predictions"]) == 1
    assert res["predictions"][0] in {0, 1}
    assert res["probabilities"] is not None
    assert res["probabilities"].shape == (1, 2)
    assert np.isclose(res["probabilities"][0].sum(), 1.0)


def test_predict_url_structured_api_response():
    """
    Verifies that predict_url returns the exact structured API response required by Requirement 10:
    {
      "url": "...",
      "prediction": "Phishing" or "Legitimate",
      "label": 0 or 1,
      "confidence": ...,
      "probabilities": {...},
      "risk_level": "...",
      "features": {...}
    }
    """
    pipeline = PredictionPipeline()

    # 1. Test Legitimate URL
    legit_res = pipeline.predict_url("https://www.google.com")
    required_keys = ["url", "prediction", "label", "confidence", "probabilities", "risk_level", "features"]
    for key in required_keys:
        assert key in legit_res, f"Missing required key: {key}"

    assert legit_res["prediction"] in {"Phishing", "Legitimate"}
    assert legit_res["label"] in {0, 1}
    assert isinstance(legit_res["confidence"], float)
    assert 0.5 <= legit_res["confidence"] <= 1.0
    assert "phishing" in legit_res["probabilities"]
    assert "legitimate" in legit_res["probabilities"]
    assert np.isclose(
        sum(legit_res["probabilities"].values()), 1.0, atol=1e-3
    )
    # Confidence must equal maximum predicted class probability
    expected_conf = max(legit_res["probabilities"].values())
    assert np.isclose(legit_res["confidence"], expected_conf, atol=1e-4)

    assert legit_res["risk_level"] in {"Low", "Medium", "High", "Critical"}
    assert isinstance(legit_res["features"], dict)
    assert len(legit_res["features"]) == 30
    assert list(legit_res["features"].keys()) == CANONICAL_FEATURE_NAMES

    # 2. Test Phishing / Suspicious URL
    phish_res = pipeline.predict_url(
        "http://secure-login.example.test@192.168.1.10/verify-account"
    )
    for key in required_keys:
        assert key in phish_res

    assert phish_res["prediction"] == "Phishing"
    assert phish_res["label"] == 0
    assert phish_res["probabilities"]["phishing"] >= 0.75
    assert phish_res["risk_level"] in {"High", "Critical"}
    # Confidence must equal maximum predicted class probability
    expected_phish_conf = max(phish_res["probabilities"].values())
    assert np.isclose(phish_res["confidence"], expected_phish_conf, atol=1e-4)


def test_predict_url_malformed_input():
    """
    Verifies error handling for malformed, empty, or invalid URLs.
    """
    pipeline = PredictionPipeline()

    # Empty string
    with pytest.raises(ValueError):
        pipeline.predict_url("")

    # None
    with pytest.raises(ValueError):
        pipeline.predict_url(None)

    # Whitespace only
    with pytest.raises(ValueError):
        pipeline.predict_url("   ")

    # Invalid URL without hostname
    with pytest.raises(ValueError):
        pipeline.predict_url("http://")


def test_missing_artifacts_error_handling(tmp_path):
    """
    Verifies error handling when model, preprocessor, or metadata artifact is missing.
    """
    # Missing model
    p_no_model = PredictionPipeline(
        model_path=str(tmp_path / "nonexistent_model.pkl"),
        preprocessor_path="final_model/preprocessor.pkl",
        metadata_path="final_model/model_metadata.json",
    )
    with pytest.raises(Exception):
        p_no_model.load_pipeline()

    # Missing preprocessor
    p_no_prep = PredictionPipeline(
        model_path="final_model/model.pkl",
        preprocessor_path=str(tmp_path / "nonexistent_prep.pkl"),
        metadata_path="final_model/model_metadata.json",
    )
    with pytest.raises(Exception):
        p_no_prep.load_pipeline()

    # Missing metadata
    p_no_meta = PredictionPipeline(
        model_path="final_model/model.pkl",
        preprocessor_path="final_model/preprocessor.pkl",
        metadata_path=str(tmp_path / "nonexistent_meta.json"),
    )
    with pytest.raises(Exception):
        p_no_meta.load_pipeline()


def test_gemini_fallback_behavior():
    """
    Verifies that if Gemini fails or times out, the fallback summary is returned
    and the prediction is not marked as failed.
    """
    from app import explain_prediction_with_gemini, local_prediction_summary

    url = "https://example.com/test"
    label = "Legitimate"
    features = {c: 1 for c in CANONICAL_FEATURE_NAMES}
    metadata = {
        "hostname": "example.com",
        "html_status": "Available",
        "dns_status": "Resolved",
    }

    # Fallback function itself generates descriptive explanation
    summary = local_prediction_summary(url, label, features)
    assert "The model predicted this URL as Legitimate" in summary

    # explain_prediction_with_gemini returns a valid explanation string even if Gemini API is disabled/fails
    explained = explain_prediction_with_gemini(url, label, features, metadata)
    assert isinstance(explained, str)
    assert len(explained) > 20


def test_predict_batch_csv_functionality(tmp_path):
    """
    Verifies predict_batch_csv processes input CSV, adds predicted columns,
    confidence, probabilities, and writes the output file.
    """
    from networksecurity.pipeline.batch_prediction import predict_batch_csv

    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "output.csv"

    # Sample dataframe with canonical features
    sample_df = pd.read_csv("Network_Data/phisingData.csv").head(5)
    sample_df.to_csv(input_csv, index=False)

    scored_df = predict_batch_csv(str(input_csv), str(output_csv))
    assert os.path.exists(output_csv)
    assert "predicted_column" in scored_df.columns
    assert "predicted_label" in scored_df.columns
    assert "confidence" in scored_df.columns
    assert "probability_phishing" in scored_df.columns
    assert "probability_legitimate" in scored_df.columns
    assert len(scored_df) == 5


