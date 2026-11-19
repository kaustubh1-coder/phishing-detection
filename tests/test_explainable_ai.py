import json
import os
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app import app, local_prediction_summary, explain_prediction_with_gemini
from networksecurity.constants.training_pipeline import CANONICAL_FEATURE_NAMES, TARGET_COLUMN
from networksecurity.pipeline.batch_prediction import PredictionPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.evaluation import evaluate_model_robustness
from networksecurity.utils.ml_utils.explainability import ModelExplainer
from networksecurity.utils.url_feature_extractor import extract_url_features


# ============================================================================
# 1. SHAP Model Explainer Unit Tests
# ============================================================================
def test_shap_model_explainer_initialization():
    """Verify SHAP TreeExplainer initializes properly for the trained Random Forest model."""
    model_path = os.path.join("final_model", "model.pkl")
    assert os.path.exists(model_path)
    model = load_object(model_path)
    explainer = ModelExplainer(model=model, feature_names=CANONICAL_FEATURE_NAMES)

    assert explainer is not None
    assert explainer._tree_explainer is not None


def test_shap_explanation_structure_and_ordering():
    """Verify that explain_instance produces exactly top 5 features ordered by importance."""
    model_path = os.path.join("final_model", "model.pkl")
    prep_path = os.path.join("final_model", "preprocessor.pkl")
    model = load_object(model_path)
    preprocessor = load_object(prep_path)
    explainer = ModelExplainer(model=model, feature_names=CANONICAL_FEATURE_NAMES)

    sample_df = pd.DataFrame([{col: 1.0 for col in CANONICAL_FEATURE_NAMES}])
    x_trans = preprocessor.transform(sample_df)
    raw_features = {col: 1.0 for col in CANONICAL_FEATURE_NAMES}
    pred_class = int(model.predict(x_trans)[0])
    probabilities = {"legitimate": 0.99, "phishing": 0.01}

    res = explainer.explain_instance(
        transformed_features=x_trans,
        raw_features=raw_features,
        predicted_class=pred_class,
        probabilities=probabilities,
        top_k=5,
    )

    assert "top_contributions" in res
    assert "all_contributions" in res
    assert "method" in res
    assert res["method"] == "SHAP (TreeExplainer)"
    assert len(res["top_contributions"]) == 5

    importances = [c["importance"] for c in res["top_contributions"]]
    assert importances == sorted(importances, reverse=True), "Top contributions must be sorted descending by importance."

    for contrib in res["top_contributions"]:
        assert "feature" in contrib
        assert "feature_name" in contrib
        assert "shap_value" in contrib
        assert "importance" in contrib
        assert "direction" in contrib
        assert contrib["direction"] in {"Phishing", "Legitimate"}
        assert "explanation" in contrib
        assert len(contrib["explanation"]) > 10
        assert "is_unavailable" in contrib
        assert isinstance(contrib["is_unavailable"], bool)


# ============================================================================
# 2. Missing-Feature Robustness Tests
# ============================================================================
def test_missing_feature_robustness_on_unreachable_webpage():
    """Verify that simulated access failure (HTML unretrieved) does NOT crash and is handled safely."""
    pipeline = PredictionPipeline()
    # Test with synthetic test domain
    url = "https://safe-internal-test.example.corp/portal"
    feature_df, raw_dict, metadata = extract_url_features(url)

    # HTML is not fetched for non-existent domain
    assert metadata["html_status"] == "Not fetched"
    assert metadata["missing_features_count"] > 0
    assert metadata["unavailable_warning"] is not None

    # Prediction must execute smoothly via KNNImputer
    res = pipeline.predict_url(url)
    assert res["prediction"] in {"Phishing", "Legitimate"}
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["top_contributions"]) == 5


def test_missing_feature_robustness_on_partial_features():
    """Verify that an input dataframe with multiple NaNs is imputed without error."""
    pipeline = PredictionPipeline()
    row = {col: 1.0 for col in CANONICAL_FEATURE_NAMES}
    # Simulate 10 missing HTML features
    for col in ["Favicon", "Request_URL", "URL_of_Anchor", "Links_in_tags", "SFH", "Iframe"]:
        row[col] = np.nan

    df = pd.DataFrame([row])
    result = pipeline.predict(df)
    assert len(result["predictions"]) == 1
    assert result["predictions"][0] in {0, 1}
    assert result["probabilities"] is not None


# ============================================================================
# 3. Evidence-Grounded Explanation Tests
# ============================================================================
def test_evidence_grounded_fallback_explanation():
    """Verify that the local fallback explanation uses genuine confidence and top SHAP features."""
    url = "https://example.com"
    label = "Legitimate"
    features = {col: 1 for col in CANONICAL_FEATURE_NAMES}
    metadata = {
        "hostname": "example.com",
        "unavailable_warning": "14 signals were estimated via KNN imputation.",
    }
    top_contribs = [
        {
            "feature": "SSLfinal_State",
            "feature_name": "SSL/TLS Certificate State",
            "shap_value": 0.115,
            "direction": "Legitimate",
        },
        {
            "feature": "Prefix_Suffix",
            "feature_name": "Domain Prefix/Suffix Hyphen",
            "shap_value": 0.101,
            "direction": "Legitimate",
        },
    ]

    summary = local_prediction_summary(
        url=url,
        label=label,
        features=features,
        metadata=metadata,
        confidence=0.985,
        probabilities={"legitimate": 0.985, "phishing": 0.015},
        top_contributions=top_contribs,
    )

    assert "98.5% confidence" in summary
    assert "SSL/TLS Certificate State" in summary
    assert "KNN imputation" in summary


# ============================================================================
# 4. Model Robustness Evaluation Module Tests
# ============================================================================
def test_model_robustness_evaluation():
    """Verify that evaluate_model_robustness generates valid metrics across all 4 scenarios."""
    model_path = os.path.join("final_model", "model.pkl")
    prep_path = os.path.join("final_model", "preprocessor.pkl")
    test_path = os.path.join("Artifacts", "10_03_2026_22_16_37", "data_ingestion", "ingested", "test.csv")
    if not os.path.exists(test_path):
        test_path = os.path.join("Network_Data", "phisingData.csv")

    model = load_object(model_path)
    preprocessor = load_object(prep_path)
    test_df = pd.read_csv(test_path).head(100)  # Quick slice for test

    report = evaluate_model_robustness(
        model=model,
        preprocessor=preprocessor,
        test_df=test_df,
        output_dir=None,  # Do not overwrite saved production artifacts during unit test
    )

    assert "scenarios" in report
    assert len(report["scenarios"]) == 4

    scenario_names = [s["name"] for s in report["scenarios"]]
    assert "Full Feature Baseline" in scenario_names
    assert "Missing Webpage / HTML Signals" in scenario_names
    assert "Missing External / WHOIS APIs" in scenario_names
    assert "Lexical-Only (URL Syntax Alone)" in scenario_names

    for sc in report["scenarios"]:
        assert 0.0 <= sc["accuracy"] <= 1.0
        assert 0.0 <= sc["f1_score"] <= 1.0
        assert 0.0 <= sc["false_positive_rate"] <= 1.0
        assert "confusion_matrix" in sc


# ============================================================================
# 5. Integration: API Endpoints for Explainability & Robustness
# ============================================================================
def test_api_predict_url_contains_explainability():
    """Verify that predict_url returns top_contributions and unavailable warning."""
    client = TestClient(app)
    pipeline = PredictionPipeline()
    result = pipeline.predict_url("https://www.google.com")

    assert "top_contributions" in result
    assert isinstance(result["top_contributions"], list)
    assert len(result["top_contributions"]) == 5

    # Test FastAPI HTML endpoint
    resp = client.post("/predict-url", data={"url": "https://www.google.com"})
    assert resp.status_code == 200
    assert "Top 5 Contributing Features (SHAP Explainability)" in resp.text


def test_api_model_robustness_endpoint():
    """Verify that /api/model-robustness returns the JSON evaluation report."""
    client = TestClient(app)
    resp = client.get("/api/model-robustness")
    assert resp.status_code == 200
    data = resp.json()
    assert "scenarios" in data
    assert len(data["scenarios"]) == 4
