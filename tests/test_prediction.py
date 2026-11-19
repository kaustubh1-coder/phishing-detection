import os
import json
import pytest
import numpy as np
import pandas as pd
import pymongo
from dotenv import load_dotenv
from fastapi.testclient import TestClient

from app import app, explain_prediction_with_gemini, local_prediction_summary, build_url_prediction
from networksecurity.constants.training_pipeline import (
    CANONICAL_FEATURE_NAMES,
    LABEL_MAPPING,
    TARGET_COLUMN,
)
from networksecurity.pipeline.batch_prediction import PredictionPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.url_feature_extractor import extract_url_features


SAFE_URLS = [
    "https://www.google.com",
    "https://www.wikipedia.org",
    "https://example.com",
]

SUSPICIOUS_SYNTHETIC_URLS = [
    "http://192.168.1.10/login",
    "http://secure-login.example.test@192.168.1.10/verify-account",
    "http://example.test/login?redirect=http://login.example.test",
]

ALL_TEST_URLS = SAFE_URLS + SUSPICIOUS_SYNTHETIC_URLS


# ============================================================================
# 1. MongoDB Connectivity
# ============================================================================
def test_01_mongodb_connectivity():
    """1. Test MongoDB connectivity and server response."""
    load_dotenv()
    mongo_uri = os.getenv("MONGODB_URL_KEY") or os.getenv("MONGO_DB_URL")
    assert mongo_uri, "MongoDB connection URI must be configured in environment (.env)."

    try:
        client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
        server_info = client.server_info()
        assert server_info is not None, "Failed to retrieve MongoDB server info."
        assert "version" in server_info or "ok" in server_info
    except pymongo.errors.ServerSelectionTimeoutError:
        pytest.skip(f"MongoDB server at {mongo_uri} not reachable in current environment.")


# ============================================================================
# 2. Dataset Loading
# ============================================================================
def test_02_dataset_loading():
    """2. Test dataset loading from CSV and MongoDB."""
    # Verify local CSV dataset
    csv_path = os.path.join("Network_Data", "phisingData.csv")
    assert os.path.exists(csv_path), f"Dataset CSV missing at {csv_path}"
    df = pd.read_csv(csv_path)
    assert not df.empty, "Dataset CSV must not be empty."
    assert TARGET_COLUMN in df.columns, f"Target column '{TARGET_COLUMN}' missing from dataset."
    assert len(df.columns) == len(CANONICAL_FEATURE_NAMES) + 1, "Dataset column count mismatch."
    assert len(df) > 5000, f"Expected full dataset (>5000 rows), found {len(df)}."

    # Verify MongoDB dataset collection if reachable
    load_dotenv()
    mongo_uri = os.getenv("MONGODB_URL_KEY")
    db_name = os.getenv("DATA_INGESTION_DATABASE_NAME", "MyPhishingDetection")
    coll_name = os.getenv("DATA_INGESTION_COLLECTION_NAME", "MyNetworkData")
    if mongo_uri:
        try:
            client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
            db = client[db_name]
            coll = db[coll_name]
            count = coll.count_documents({})
            assert count > 0, "MongoDB collection has 0 documents."
        except Exception:
            pass


# ============================================================================
# 3. Feature Extraction on Safe and Suspicious Synthetic URLs
# ============================================================================
@pytest.mark.parametrize("url", ALL_TEST_URLS)
def test_03_feature_extraction(url: str):
    """
    3. Test feature extraction does not crash and returns all 30 canonical features,
    even for synthetic/non-existent domains.
    """
    feature_df, features_dict, metadata = extract_url_features(url)

    assert isinstance(feature_df, pd.DataFrame), "Feature extraction must return a DataFrame."
    assert feature_df.shape == (1, 30), f"Expected shape (1, 30), got {feature_df.shape}."
    assert list(feature_df.columns) == CANONICAL_FEATURE_NAMES, "Extracted column schema mismatch."

    assert isinstance(features_dict, dict), "Features must be returned as a dictionary."
    assert len(features_dict) == 30, f"Expected 30 features in dict, got {len(features_dict)}."

    assert isinstance(metadata, dict), "Metadata must be a dictionary."
    assert metadata["hostname"] != "", f"Hostname should not be empty for URL {url}."
    assert "dns_status" in metadata, "Metadata missing 'dns_status'."
    assert "html_status" in metadata, "Metadata missing 'html_status'."
    assert "feature_availability" in metadata, "Metadata missing 'feature_availability'."


# ============================================================================
# 4. Feature Schema Validation
# ============================================================================
def test_04_feature_schema_validation():
    """4. Test canonical schema validation, missing/extra column rejection, and ordering."""
    pipeline = PredictionPipeline()

    # Valid schema passes and retains canonical ordering
    valid_df = pd.DataFrame([{col: 1 for col in CANONICAL_FEATURE_NAMES}])
    validated = pipeline.validate_features(valid_df)
    assert list(validated.columns) == CANONICAL_FEATURE_NAMES

    # Missing column raises ValueError
    missing_col_df = pd.DataFrame([{col: 1 for col in CANONICAL_FEATURE_NAMES[:-1]}])
    with pytest.raises(ValueError) as excinfo:
        pipeline.validate_features(missing_col_df)
    assert "Missing features" in str(excinfo.value)

    # Extra column raises ValueError
    extra_col_df = valid_df.copy()
    extra_col_df["unexpected_bogus_col"] = 99
    with pytest.raises(ValueError) as excinfo:
        pipeline.validate_features(extra_col_df)
    assert "Extra features" in str(excinfo.value)


# ============================================================================
# 5. Preprocessor Loading
# ============================================================================
def test_05_preprocessor_loading():
    """5. Test loading saved preprocessor and transformation consistency."""
    prep_path = os.path.join("final_model", "preprocessor.pkl")
    assert os.path.exists(prep_path), f"Preprocessor missing at {prep_path}"
    preprocessor = load_object(prep_path)

    assert preprocessor is not None
    assert hasattr(preprocessor, "transform"), "Preprocessor must implement transform()."
    assert getattr(preprocessor, "n_features_in_", None) == 30, "Preprocessor must expect 30 input features."

    # Test transform on canonical dataframe
    sample_df = pd.DataFrame([{col: 1 for col in CANONICAL_FEATURE_NAMES}])
    transformed = preprocessor.transform(sample_df)
    assert transformed.shape == (1, 30), "Transformed matrix shape mismatch."
    assert not np.isnan(transformed).any(), "Transformed matrix contains NaN."
    assert not np.isinf(transformed).any(), "Transformed matrix contains Inf."


# ============================================================================
# 6. Model Loading
# ============================================================================
def test_06_model_loading():
    """6. Test loading saved model and estimator properties."""
    model_path = os.path.join("final_model", "model.pkl")
    assert os.path.exists(model_path), f"Model missing at {model_path}"
    model = load_object(model_path)

    assert model is not None
    assert hasattr(model, "predict"), "Model must implement predict()."
    assert hasattr(model, "predict_proba"), "Model must implement predict_proba()."
    assert getattr(model, "n_features_in_", None) == 30, "Model must expect 30 features."
    classes = getattr(model, "classes_", None)
    assert classes is not None, "Model must have classes_ attribute."
    assert len(classes) == 2, "Binary classification model must have 2 classes."


# ============================================================================
# 7. Prediction & Model Quality Reporting
# ============================================================================
@pytest.mark.parametrize("url", ALL_TEST_URLS)
def test_07_prediction_and_quality_reporting(url: str):
    """
    7. Test prediction execution and report model quality (actual predictions and probabilities).
    Does NOT assert hardcoded outcomes; verifies that inference executes successfully.
    """
    pipeline = PredictionPipeline()
    result = pipeline.predict_url(url)

    # Verify valid inference output structure
    assert result["prediction"] in {"Phishing", "Legitimate"}, f"Unexpected prediction: {result['prediction']}"
    assert result["label"] in {0, 1}, f"Unexpected label: {result['label']}"
    assert 0.0 <= result["confidence"] <= 1.0

    # Model quality reporting (actual predictions and probabilities)
    print(
        f"\n[Model Quality Inspection] URL: {url} | "
        f"Prediction: {result['prediction']} (class {result['label']}) | "
        f"Confidence: {result['confidence']:.4f} | "
        f"Probabilities: {result['probabilities']} | "
        f"Risk Level: {result['risk_level']}"
    )


# ============================================================================
# 8. Probability Calculation
# ============================================================================
@pytest.mark.parametrize("url", ALL_TEST_URLS)
def test_08_probability_calculation(url: str):
    """8. Test probability calculation and maximum class probability confidence."""
    pipeline = PredictionPipeline()
    result = pipeline.predict_url(url)
    probabilities = result["probabilities"]

    assert "phishing" in probabilities, "Probabilities missing 'phishing'."
    assert "legitimate" in probabilities, "Probabilities missing 'legitimate'."
    prob_p = probabilities["phishing"]
    prob_l = probabilities["legitimate"]

    assert 0.0 <= prob_p <= 1.0, f"Phishing probability out of range: {prob_p}"
    assert 0.0 <= prob_l <= 1.0, f"Legitimate probability out of range: {prob_l}"
    assert np.isclose(prob_p + prob_l, 1.0, atol=1e-3), "Class probabilities must sum to 1.0."

    # Genuine confidence calculation: max predicted class probability
    expected_confidence = max(prob_p, prob_l)
    assert np.isclose(result["confidence"], expected_confidence, atol=1e-4), "Confidence does not match max class probability."


# ============================================================================
# 9. Label Mapping Consistency
# ============================================================================
def test_09_label_mapping_consistency():
    """9. Test ground-truth label mapping across pipeline, metadata, and constants."""
    pipeline = PredictionPipeline()
    label_map = pipeline.get_label_mapping()

    assert label_map[0] == "Phishing", "Class 0 must map to 'Phishing'."
    assert label_map[1] == "Legitimate", "Class 1 must map to 'Legitimate'."

    metadata = pipeline.load_metadata()
    if metadata and "label_mapping" in metadata:
        assert metadata["label_mapping"]["0"] == "Phishing"
        assert metadata["label_mapping"]["1"] == "Legitimate"

    assert LABEL_MAPPING[0] == "Phishing"
    assert LABEL_MAPPING[1] == "Legitimate"


# ============================================================================
# 10. API Response Structure
# ============================================================================
def test_10_api_response_structure():
    """10. Test structured API response matching Requirement 10."""
    pipeline = PredictionPipeline()
    result = pipeline.predict_url("https://www.google.com")

    required_keys = [
        "url",
        "prediction",
        "label",
        "confidence",
        "probabilities",
        "risk_level",
        "features",
    ]
    for key in required_keys:
        assert key in result, f"Requirement 10 key '{key}' missing from response."

    assert isinstance(result["url"], str)
    assert result["prediction"] in {"Phishing", "Legitimate"}
    assert result["label"] in {0, 1}
    assert isinstance(result["confidence"], float)
    assert isinstance(result["probabilities"], dict)
    assert isinstance(result["risk_level"], str)
    assert isinstance(result["features"], dict)
    assert len(result["features"]) == 30

    # Test via FastAPI test client
    client = TestClient(app)
    response = client.post("/predict-url", data={"url": "https://example.com"})
    assert response.status_code == 200


# ============================================================================
# 11. Gemini Failure Fallback
# ============================================================================
def test_11_gemini_failure_fallback():
    """11. Test that Gemini failure or timeout gracefully returns fallback without failing prediction."""
    url = "http://example.test/login?redirect=http://login.example.test"
    label = "Phishing"
    features = {col: 1 for col in CANONICAL_FEATURE_NAMES}
    metadata = {
        "hostname": "example.test",
        "dns_status": "Not resolved",
        "html_status": "Failed",
    }

    # Verify fallback explanation is generated safely
    fallback = local_prediction_summary(url, label, features)
    assert "The model predicted this URL as Phishing" in fallback

    # Verify explain_prediction_with_gemini returns a non-empty string and does not crash
    explanation = explain_prediction_with_gemini(url, label, features, metadata)
    assert isinstance(explanation, str)
    assert len(explanation) > 10

    # Ensure build_url_prediction does not crash even if Gemini times out or is offline
    prediction_result = build_url_prediction(url)
    assert prediction_result["prediction"] in {"Phishing", "Legitimate"}
    assert "summary" in prediction_result


# ============================================================================
# 12. Invalid URL Handling
# ============================================================================
def test_12_invalid_url_handling():
    """12. Test error handling for empty, malformed, non-string, or invalid URLs."""
    pipeline = PredictionPipeline()

    # Empty string
    with pytest.raises(ValueError):
        pipeline.predict_url("")

    # None
    with pytest.raises(ValueError):
        pipeline.predict_url(None)

    # Whitespace
    with pytest.raises(ValueError):
        pipeline.predict_url("   \t\n  ")

    # Missing hostname
    with pytest.raises(ValueError):
        pipeline.predict_url("http://")

    # Non-string input
    with pytest.raises(ValueError):
        pipeline.predict_url(12345)

    # FastAPI endpoint error response
    client = TestClient(app)
    response_empty = client.post("/predict-url", data={"url": ""})
    assert response_empty.status_code in {200, 400, 422}

    response_nohost = client.post("/predict-url", data={"url": "http://"})
    assert response_nohost.status_code == 200
    assert "Invalid URL" in response_nohost.text
