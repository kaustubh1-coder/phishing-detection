import json
import os
import sys
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union

from networksecurity.constants.training_pipeline import (
    CANONICAL_FEATURE_NAMES,
    LABEL_MAPPING,
    TARGET_COLUMN,
)
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.explainability import ModelExplainer
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.url_feature_extractor import extract_url_features


class PredictionPipeline:
    """
    Reusable prediction service for both live URL scoring and batch CSV prediction.
    Enforces canonical feature schema, single-pass preprocessing, logging,
    artifact verification, and structured response formatting.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        preprocessor_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
    ):
        self.model_path = model_path or os.path.join("final_model", "model.pkl")
        self.preprocessor_path = preprocessor_path or os.path.join(
            "final_model", "preprocessor.pkl"
        )
        self.metadata_path = metadata_path or os.path.join(
            "final_model", "model_metadata.json"
        )
        self.network_model: Optional[NetworkModel] = None
        self.metadata: Optional[Dict[str, Any]] = None
        self.explainer: Optional[ModelExplainer] = None

    def get_explainer(self) -> ModelExplainer:
        """
        Loads and caches the explainability service (SHAP TreeExplainer) for the trained model.
        """
        if self.explainer is None:
            pipeline = self.load_pipeline()
            self.explainer = ModelExplainer(
                model=pipeline.model,
                feature_names=CANONICAL_FEATURE_NAMES,
            )
        return self.explainer

    def load_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Loads and caches training metadata from the configured path.
        """
        if self.metadata is None and os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logging.info(
                    f"Loaded model metadata: {self.metadata.get('model_name')}, "
                    f"features={self.metadata.get('feature_count')}, "
                    f"trained_at={self.metadata.get('training_date')}"
                )
            except Exception as e:
                logging.warning(f"Could not load metadata from {self.metadata_path}: {e}")
        return self.metadata

    def get_label_mapping(self) -> Dict[int, str]:
        """
        Determines the label mapping from training metadata.
        Never manually inverts prediction values.
        """
        metadata = self.load_metadata()
        if metadata and "label_mapping" in metadata:
            return {int(k): v for k, v in metadata["label_mapping"].items()}
        return {int(k): v for k, v in LABEL_MAPPING.items()}

    def verify_artifacts_consistency(self, preprocessor: Any, model: Any) -> None:
        """
        Verifies that the model and preprocessor belong to the same training run
        and share the exact same feature expectations.
        """
        metadata = self.load_metadata()
        if not metadata:
            raise FileNotFoundError(
                f"Model metadata file not found at: {self.metadata_path}. "
                "Cannot verify artifact consistency."
            )

        expected_count = metadata.get("feature_count")
        if expected_count is None:
            raise ValueError("Corrupt metadata: 'feature_count' is missing.")

        features = metadata.get("features", [])
        if len(features) != expected_count:
            raise ValueError(
                f"Metadata corrupt: feature_count ({expected_count}) does not match "
                f"length of features list ({len(features)})."
            )

        # 1. Verify preprocessor feature count
        prep_features = getattr(preprocessor, "n_features_in_", None)
        if prep_features is not None and prep_features != expected_count:
            raise ValueError(
                f"Preprocessor / Metadata mismatch: Preprocessor was fit on {prep_features} features, "
                f"but metadata specifies {expected_count} features."
            )

        # 2. Verify model feature count
        model_features = getattr(model, "n_features_in_", None)
        if model_features is not None and model_features != expected_count:
            raise ValueError(
                f"Model / Metadata mismatch: Model expects {model_features} features, "
                f"but metadata specifies {expected_count} features."
            )

        # 3. Verify preprocessor output dimensionality matches model input
        if prep_features is not None and model_features is not None:
            if prep_features != model_features:
                raise ValueError(
                    f"Artifact mismatch: Preprocessor feature dimension ({prep_features}) "
                    f"does not match Model feature dimension ({model_features})."
                )

        logging.info(
            f"Artifact consistency verified: model='{metadata.get('model_name')}', "
            f"features={expected_count}, trained_at={metadata.get('training_date')}"
        )

    def load_pipeline(self) -> NetworkModel:
        """
        Loads the model only from the configured final_model path and the matching preprocessor.
        Verifies that both artifacts belong to the same training run.
        """
        try:
            if self.network_model is None:
                if not os.path.exists(self.model_path):
                    raise FileNotFoundError(
                        f"Trained model artifact not found at: {self.model_path}"
                    )
                if not os.path.exists(self.preprocessor_path):
                    raise FileNotFoundError(
                        f"Preprocessor artifact not found at: {self.preprocessor_path}"
                    )

                preprocessor = load_object(self.preprocessor_path)
                model = load_object(self.model_path)

                # Verify artifacts belong to the same training run
                self.verify_artifacts_consistency(preprocessor=preprocessor, model=model)

                self.network_model = NetworkModel(preprocessor=preprocessor, model=model)

            return self.network_model
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates that the input dataframe contains the exact canonical feature schema.
        Raises ValueError with expected, received, missing, and extra feature details on mismatch.
        Returns a DataFrame strictly ordered by expected features.
        """
        metadata = self.load_metadata()
        expected_cols = (
            metadata.get("features", CANONICAL_FEATURE_NAMES)
            if metadata
            else CANONICAL_FEATURE_NAMES
        )
        expected_count = (
            metadata.get("feature_count", len(expected_cols))
            if metadata
            else len(expected_cols)
        )

        # Exclude target column if present in evaluation/test CSVs
        input_df = df.drop(columns=[TARGET_COLUMN], errors="ignore")
        received_cols = list(input_df.columns)

        missing_cols = [c for c in expected_cols if c not in received_cols]
        extra_cols = [c for c in received_cols if c not in expected_cols]

        if missing_cols or extra_cols or len(received_cols) != expected_count:
            error_msg = (
                f"Feature schema validation failed.\n"
                f"Expected features: {expected_count}\n"
                f"Received features: {len(received_cols)}\n"
                f"Missing features: {missing_cols}\n"
                f"Extra features: {extra_cols}"
            )
            logging.error(error_msg)
            raise ValueError(error_msg)

        # Enforce exact canonical column order
        return input_df[expected_cols].copy()

    def predict(self, input_df: pd.DataFrame) -> Dict[str, Union[np.ndarray, List[str], Any]]:
        """
        Executes single-pass transformation and prediction with probability output.
        """
        # Validate features directly (raises ValueError on mismatch)
        validated_df = self.validate_features(input_df)

        try:
            network_model = self.load_pipeline()

            # Safe operational logging (strictly no secrets/credentials)
            logging.info(f"Executing prediction with model: {self.model_path}")
            logging.info(f"Executing preprocessing with: {self.preprocessor_path}")
            logging.info(f"Input feature count: {len(validated_df.columns)}")
            logging.info(f"Input feature names: {list(validated_df.columns)}")

            # Single-pass transform and prediction
            y_pred, y_proba = network_model.predict_with_proba(validated_df)

            logging.info(f"Prediction completed for {len(y_pred)} sample(s)")
            if len(y_pred) <= 10:
                logging.info(f"Predicted class: {y_pred.tolist()}")
                if y_proba is not None:
                    logging.info(f"Prediction probabilities: {y_proba.tolist()}")

            return {
                "predictions": y_pred,
                "probabilities": y_proba,
                "feature_names": list(validated_df.columns),
                "metadata": self.metadata,
            }
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def predict_url(self, raw_url: str) -> Dict[str, Any]:
        """
        Executes end-to-end prediction for a single live URL:
        URL -> Canonical feature extractor -> Canonical DataFrame -> Saved preprocessor
        -> Saved model -> Prediction -> Probability -> Structured Response.

        Returns structured API response meeting Requirement 10:
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
        if not raw_url or not isinstance(raw_url, str) or not raw_url.strip():
            raise ValueError("URL must be a non-empty string.")

        # 1. URL -> Canonical feature extractor & metadata
        feature_df, features_dict, url_metadata = extract_url_features(raw_url)

        # 2. Canonical DataFrame validation (validates feature count, names, and canonical order)
        validated_df = self.validate_features(feature_df)

        # 3. Load pipeline (strictly from configured final_model path with artifact consistency check)
        network_model = self.load_pipeline()

        # 4. Saved preprocessor & model -> Single-pass predict and predict_proba
        y_pred, y_proba = network_model.predict_with_proba(validated_df)

        # 5. Label mapping determined strictly from training metadata (never manually inverted)
        label_mapping = self.get_label_mapping()
        label_int = int(y_pred[0])
        prediction_str = label_mapping.get(label_int, "Phishing" if label_int == 0 else "Legitimate")

        # 6. Confidence and probabilities calculation
        probabilities: Dict[str, float] = {}
        if y_proba is not None:
            # Confidence is the maximum predicted class probability for binary classification
            confidence = float(round(float(np.max(y_proba[0])), 4))

            # Map probabilities to class names from metadata/model
            classes = getattr(network_model.model, "classes_", [0, 1])
            for idx, cls in enumerate(classes):
                cls_int = int(cls)
                class_name = label_mapping.get(cls_int, f"class_{cls_int}").lower()
                probabilities[class_name] = float(round(float(y_proba[0][idx]), 4))
        else:
            confidence = 1.0
            probabilities = {
                "phishing": 1.0 if label_int == 0 else 0.0,
                "legitimate": 1.0 if label_int == 1 else 0.0,
            }

        # 7. Risk level derived from phishing probability
        phishing_prob = probabilities.get("phishing", 1.0 if label_int == 0 else 0.0)
        if phishing_prob >= 0.85:
            risk_level = "Critical"
        elif phishing_prob >= 0.60:
            risk_level = "High"
        elif phishing_prob >= 0.40:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # 8. Clean feature values for serialization (convert np.nan to None)
        serializable_features = {}
        for k, v in features_dict.items():
            if pd.isna(v) or v is None:
                serializable_features[k] = None
            elif isinstance(v, (np.integer, int)):
                serializable_features[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                serializable_features[k] = float(v)
            else:
                serializable_features[k] = v

        normalized_url = url_metadata.get("normalized_url", raw_url.strip())

        # 9. Explainable AI: Compute SHAP feature contributions for the predicted class
        transformed_features = network_model.transform(validated_df)
        explainer = self.get_explainer()
        explainability = explainer.explain_instance(
            transformed_features=transformed_features,
            raw_features=features_dict,
            predicted_class=label_int,
            probabilities=probabilities,
            top_k=5,
        )
        top_contributions = explainability.get("top_contributions", [])

        return {
            "url": normalized_url,
            "prediction": prediction_str,
            "label": label_int,
            "confidence": confidence,
            "probabilities": probabilities,
            "risk_level": risk_level,
            "features": serializable_features,
            "top_contributions": top_contributions,
            "topContributions": top_contributions,
            "explainability": explainability,
            "metadata": url_metadata,
        }


def predict_batch_csv(
    input_csv_path: str,
    output_csv_path: str,
    pipeline: Optional[PredictionPipeline] = None,
) -> pd.DataFrame:
    """
    Utility function for batch CSV prediction. Reads CSV, scores it, appends predictions, and saves output.
    """
    try:
        pipeline = pipeline or PredictionPipeline()
        df = pd.read_csv(input_csv_path)
        result = pipeline.predict(df)

        label_map = pipeline.get_label_mapping()
        df["predicted_column"] = result["predictions"]
        df["predicted_label"] = [
            label_map.get(int(p), "Phishing" if int(p) == 0 else "Legitimate")
            for p in result["predictions"]
        ]

        if result["probabilities"] is not None:
            network_model = pipeline.load_pipeline()
            classes = getattr(network_model.model, "classes_", [0, 1])
            for idx, cls in enumerate(classes):
                cls_int = int(cls)
                class_name = label_map.get(cls_int, f"class_{cls_int}").lower()
                df[f"probability_{class_name}"] = result["probabilities"][:, idx].round(4)
            df["confidence"] = np.max(result["probabilities"], axis=1).round(4)

        dir_name = os.path.dirname(output_csv_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        return df
    except Exception as e:
        raise NetworkSecurityException(e, sys)
