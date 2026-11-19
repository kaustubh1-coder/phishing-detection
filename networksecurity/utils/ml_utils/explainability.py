import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import shap

from networksecurity.constants.training_pipeline import CANONICAL_FEATURE_NAMES, LABEL_MAPPING
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


FEATURE_METADATA: Dict[str, Dict[str, str]] = {
    "having_IP_Address": {
        "friendly_name": "IP Address in Hostname",
        "category": "URL Syntax",
        "phishing_reason": "URL uses a raw IP address instead of a domain name, commonly used to bypass reputation filters.",
        "legitimate_reason": "URL uses a standard registered domain name.",
        "neutral_reason": "IP presence could not be determined.",
    },
    "URL_Length": {
        "friendly_name": "URL Character Length",
        "category": "URL Syntax",
        "phishing_reason": "Excessively long URL (>75 characters), typical in phishing links concealing payload or token strings.",
        "legitimate_reason": "Standard compact URL length (<54 characters).",
        "neutral_reason": "Moderate URL length (54-75 characters).",
    },
    "Shortining_Service": {
        "friendly_name": "URL Shortening Service",
        "category": "URL Syntax",
        "phishing_reason": "Uses a link shortener (e.g., bit.ly, tinyurl) which obscures the true destination hostname.",
        "legitimate_reason": "Direct domain destination without URL shortening obfuscation.",
        "neutral_reason": "Shortening service usage is neutral.",
    },
    "having_At_Symbol": {
        "friendly_name": "@ Symbol in Authority",
        "category": "URL Syntax",
        "phishing_reason": "Contains '@' symbol, which causes browsers to ignore preceding credentials and navigate to the following host.",
        "legitimate_reason": "Standard URL syntax without deceptive '@' symbols.",
        "neutral_reason": "Standard authority formatting.",
    },
    "double_slash_redirecting": {
        "friendly_name": "Double Slash Path Redirection",
        "category": "URL Syntax",
        "phishing_reason": "Contains '//' in the URL path after the protocol, triggering unvalidated browser redirects.",
        "legitimate_reason": "Clean path structure without embedded protocol redirects.",
        "neutral_reason": "Standard path slashes.",
    },
    "Prefix_Suffix": {
        "friendly_name": "Domain Prefix/Suffix Hyphen",
        "category": "URL Syntax",
        "phishing_reason": "Domain name contains hyphens, a frequent technique in brand impersonation and typosquatting.",
        "legitimate_reason": "Domain name has no deceptive hyphens.",
        "neutral_reason": "Standard domain naming structure.",
    },
    "having_Sub_Domain": {
        "friendly_name": "Subdomain Depth & Anomaly",
        "category": "URL Syntax",
        "phishing_reason": "Multiple nested subdomains, often mimicking legitimate brand names on attacker infrastructure.",
        "legitimate_reason": "Standard single subdomain or naked domain structure.",
        "neutral_reason": "Moderate subdomain depth.",
    },
    "SSLfinal_State": {
        "friendly_name": "SSL/TLS Certificate State",
        "category": "Network & Protocols",
        "phishing_reason": "Missing HTTPS, expired certificate, or untrusted certificate authority.",
        "legitimate_reason": "Valid SSL certificate issued by a trusted certificate authority.",
        "neutral_reason": "Certificate status unverified or network unreachable.",
    },
    "Domain_registeration_length": {
        "friendly_name": "Domain Registration Term",
        "category": "Domain & Identity",
        "phishing_reason": "Short domain registration duration (<= 1 year), characteristic of temporary phishing sites.",
        "legitimate_reason": "Long-term registered domain (> 1 year), reflecting established ownership.",
        "neutral_reason": "Domain registration term imputed from typical domain distributions.",
    },
    "Favicon": {
        "friendly_name": "Favicon Origin Integrity",
        "category": "Webpage Content",
        "phishing_reason": "Favicon loaded from an external untrusted domain, indicating brand asset theft.",
        "legitimate_reason": "Favicon hosted on same origin or reputable CDN.",
        "neutral_reason": "Favicon signal imputed due to offline or unretrieved webpage.",
    },
    "port": {
        "friendly_name": "Non-Standard Network Port",
        "category": "Network & Protocols",
        "phishing_reason": "Runs on a non-standard port outside 80/443, frequently used to evade corporate web proxies.",
        "legitimate_reason": "Uses standard web ports (80 HTTP / 443 HTTPS).",
        "neutral_reason": "Standard port configuration.",
    },
    "HTTPS_token": {
        "friendly_name": "HTTPS Token in Domain Name",
        "category": "URL Syntax",
        "phishing_reason": "Deceptively embeds 'https' inside domain name (e.g., https-secure-login.com).",
        "legitimate_reason": "No deceptive HTTPS tokens embedded in the hostname.",
        "neutral_reason": "Normal hostname naming.",
    },
    "Request_URL": {
        "friendly_name": "External Asset Request Ratio",
        "category": "Webpage Content",
        "phishing_reason": "High percentage of media and script assets loaded from external domains.",
        "legitimate_reason": "Assets loaded internally from the authoritative domain.",
        "neutral_reason": "Asset ratio imputed from observed URL signals.",
    },
    "URL_of_Anchor": {
        "friendly_name": "Anchor Link Destination Ratio",
        "category": "Webpage Content",
        "phishing_reason": "Anchor links point to empty actions ('#'), 'about:blank', or unrelated external domains.",
        "legitimate_reason": "Anchor tags navigate to valid internal site pages.",
        "neutral_reason": "Anchor link profile imputed from nearest known site profiles.",
    },
    "Links_in_tags": {
        "friendly_name": "Meta / Script Tag Origin Ratio",
        "category": "Webpage Content",
        "phishing_reason": "Significant proportion of meta and script tags reference external third-party origins.",
        "legitimate_reason": "Meta, script, and link tags are properly anchored to the site origin.",
        "neutral_reason": "Tag link distribution estimated by imputer.",
    },
    "SFH": {
        "friendly_name": "Server Form Handler (SFH)",
        "category": "Webpage Content",
        "phishing_reason": "Form submission handler is empty or sends sensitive form data to an untrusted external server.",
        "legitimate_reason": "Form handler submits securely to the verified first-party domain.",
        "neutral_reason": "Server form handler estimated via imputer.",
    },
    "Submitting_to_email": {
        "friendly_name": "Form Submission to Email",
        "category": "Webpage Content",
        "phishing_reason": "Webpage uses mailto: function to transmit user credentials directly to an attacker email.",
        "legitimate_reason": "Standard server-side database authentication without email leaks.",
        "neutral_reason": "Email submission behavior estimated by imputer.",
    },
    "Abnormal_URL": {
        "friendly_name": "Host Identity Consistency",
        "category": "Domain & Identity",
        "phishing_reason": "Host does not match WHOIS registry or redirects to an entirely distinct identity.",
        "legitimate_reason": "Host identity matches WHOIS registration and redirection chain.",
        "neutral_reason": "Host identity consistency inferred from URL structure.",
    },
    "Redirect": {
        "friendly_name": "HTTP Redirect Chain",
        "category": "Network & Protocols",
        "phishing_reason": "Excessive redirection hops or open redirect parameter detected.",
        "legitimate_reason": "Clean direct navigation with minimal redirect hops (<= 1).",
        "neutral_reason": "Redirect status is normal.",
    },
    "on_mouseover": {
        "friendly_name": "Mouseover Status Bar Manipulation",
        "category": "Webpage Content",
        "phishing_reason": "JavaScript onmouseover event alters the status bar to disguise link destination.",
        "legitimate_reason": "Standard transparent hyperlink mouseover behavior.",
        "neutral_reason": "Mouseover behavior estimated by imputer.",
    },
    "RightClick": {
        "friendly_name": "Disabled Right-Click Context Menu",
        "category": "Webpage Content",
        "phishing_reason": "Context menu is disabled via JavaScript to hinder inspection of page source code.",
        "legitimate_reason": "Browser right-click and source inspection enabled normally.",
        "neutral_reason": "Context menu behavior estimated by imputer.",
    },
    "popUpWidnow": {
        "friendly_name": "Credential Pop-Up Window",
        "category": "Webpage Content",
        "phishing_reason": "Spawns automated pop-up dialogs prompting for passwords or credit card numbers.",
        "legitimate_reason": "Clean user interface without intrusive credential popups.",
        "neutral_reason": "Popup behavior estimated by imputer.",
    },
    "Iframe": {
        "friendly_name": "Invisible / Overlay Iframe",
        "category": "Webpage Content",
        "phishing_reason": "Utilizes hidden iframe overlays to spoof login forms or hijack clicks.",
        "legitimate_reason": "No deceptive iframe overlay detected.",
        "neutral_reason": "Iframe usage estimated by imputer.",
    },
    "age_of_domain": {
        "friendly_name": "Domain Age / Maturity",
        "category": "Domain & Identity",
        "phishing_reason": "Newly registered domain (<6 months), strongly associated with disposable phishing campaigns.",
        "legitimate_reason": "Established domain with substantial operational tenure (>6 months).",
        "neutral_reason": "Domain age imputed based on similar URL characteristics.",
    },
    "DNSRecord": {
        "friendly_name": "DNS Record Availability",
        "category": "Network & Protocols",
        "phishing_reason": "Domain has no valid DNS A/AAAA record resolved in authoritative nameservers.",
        "legitimate_reason": "Authoritative DNS record resolved successfully.",
        "neutral_reason": "DNS record lookup neutral or unverified.",
    },
    "web_traffic": {
        "friendly_name": "Global Web Traffic Rank",
        "category": "External Reputation",
        "phishing_reason": "Extremely low or non-existent global web traffic.",
        "legitimate_reason": "High global web traffic ranking, reflecting an established user base.",
        "neutral_reason": "Web traffic estimated by imputer from known dataset distribution.",
    },
    "Page_Rank": {
        "friendly_name": "Search Engine PageRank",
        "category": "External Reputation",
        "phishing_reason": "Zero or negligible PageRank authority.",
        "legitimate_reason": "Positive PageRank authority from established web graph links.",
        "neutral_reason": "PageRank estimated by imputer from dataset distribution.",
    },
    "Google_Index": {
        "friendly_name": "Search Engine Indexing",
        "category": "External Reputation",
        "phishing_reason": "Website is not indexed in major search engine results.",
        "legitimate_reason": "Website is indexed and verified by search engines.",
        "neutral_reason": "Indexing status estimated by imputer from dataset distribution.",
    },
    "Links_pointing_to_page": {
        "friendly_name": "Inbound External Backlinks",
        "category": "External Reputation",
        "phishing_reason": "Zero or negligible inbound links pointing to the website.",
        "legitimate_reason": "Numerous inbound references from external domains.",
        "neutral_reason": "Backlink density estimated by imputer from dataset distribution.",
    },
    "Statistical_report": {
        "friendly_name": "Threat Intelligence Match",
        "category": "External Reputation",
        "phishing_reason": "Domain or IP matched known phishing blacklists or threat intelligence feeds.",
        "legitimate_reason": "No match found in known malicious host databases.",
        "neutral_reason": "Threat intelligence status clear.",
    },
}


class ModelExplainer:
    """
    Production-grade Explainable AI service using SHAP TreeExplainer.
    Identifies the top features influencing each prediction, computes genuine SHAP values,
    and returns human-friendly evidence-grounded explanations.
    """

    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        self.model = model
        self.feature_names = feature_names or CANONICAL_FEATURE_NAMES
        self._tree_explainer: Optional[shap.TreeExplainer] = None
        self._init_explainer()

    def _init_explainer(self) -> None:
        try:
            self._tree_explainer = shap.TreeExplainer(self.model)
            logging.info("Initialized SHAP TreeExplainer successfully.")
        except Exception as e:
            logging.warning(f"Could not initialize TreeExplainer directly: {e}. Fallback will be used.")
            self._tree_explainer = None

    def explain_instance(
        self,
        transformed_features: np.ndarray,
        raw_features: Dict[str, Any],
        predicted_class: int,
        probabilities: Dict[str, float],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Computes SHAP values for the transformed instance and returns the top_k
        contributing features with direction, importance, and plain-English explanations.
        """
        try:
            if transformed_features.ndim == 1:
                transformed_features = transformed_features.reshape(1, -1)

            if self._tree_explainer is not None:
                shap_obj = self._tree_explainer(transformed_features)
                raw_shap_values = shap_obj.values  # Shape: (1, n_features, n_classes) or (1, n_features)
                base_values = shap_obj.base_values

                if raw_shap_values.ndim == 3:
                    # Multi-class: pick the values for the predicted class
                    class_idx = int(predicted_class)
                    if class_idx < raw_shap_values.shape[2]:
                        feature_shap = raw_shap_values[0, :, class_idx]
                        base_val = float(base_values[0, class_idx]) if hasattr(base_values, "__getitem__") else float(base_values)
                    else:
                        feature_shap = raw_shap_values[0, :, 0]
                        base_val = 0.5
                else:
                    feature_shap = raw_shap_values[0, :]
                    base_val = float(base_values[0]) if hasattr(base_values, "__getitem__") else float(base_values)
                method_name = "SHAP (TreeExplainer)"
            else:
                # Heuristic fallback based on feature importances
                feature_importances = getattr(self.model, "feature_importances_", np.ones(len(self.feature_names)) / len(self.feature_names))
                # Direction from sign of transformed feature relative to mean
                feature_shap = feature_importances * (transformed_features[0] - 0.0)
                base_val = 0.5
                method_name = "Feature Importance Weighting (Fallback)"

            predicted_label_str = LABEL_MAPPING.get(int(predicted_class), "Phishing" if int(predicted_class) == 0 else "Legitimate")
            contributions: List[Dict[str, Any]] = []
            all_contributions_dict: Dict[str, float] = {}

            for idx, col in enumerate(self.feature_names):
                shap_val = float(feature_shap[idx])
                all_contributions_dict[col] = round(shap_val, 4)

                meta = FEATURE_METADATA.get(col, {
                    "friendly_name": col.replace("_", " ").title(),
                    "category": "General",
                    "phishing_reason": "Feature indicates suspicious pattern.",
                    "legitimate_reason": "Feature indicates normal pattern.",
                    "neutral_reason": "Feature signal is neutral or imputed.",
                })

                raw_val = raw_features.get(col)
                is_unavailable = raw_val is None or pd.isna(raw_val)

                # Determine direction:
                # In binary classification:
                # For class 0 (Phishing): positive SHAP pushes towards Phishing, negative pushes towards Legitimate.
                # For class 1 (Legitimate): positive SHAP pushes towards Legitimate, negative pushes towards Phishing.
                if int(predicted_class) == 0:  # Phishing
                    pushes_target = shap_val > 0
                    direction = "Phishing" if pushes_target else "Legitimate"
                else:  # Legitimate
                    pushes_target = shap_val > 0
                    direction = "Legitimate" if pushes_target else "Phishing"

                # Generate domain explanation
                if is_unavailable:
                    explanation = (
                        f"{meta['friendly_name']} was unavailable from network access and estimated via "
                        f"KNN imputation. It contributed a weight of {abs(shap_val):.3f} toward {direction}."
                    )
                    display_value = "Imputed (N/A)"
                elif raw_val == -1:
                    explanation = meta["phishing_reason"]
                    display_value = "Suspicious (-1)"
                elif raw_val == 1:
                    explanation = meta["legitimate_reason"]
                    display_value = "Normal (+1)"
                else:
                    explanation = meta["neutral_reason"]
                    display_value = "Neutral (0)"

                contributions.append({
                    "feature": col,
                    "feature_name": meta["friendly_name"],
                    "category": meta["category"],
                    "raw_value": None if is_unavailable else float(raw_val),
                    "display_value": display_value,
                    "shap_value": round(shap_val, 4),
                    "importance": round(abs(shap_val), 4),
                    "direction": direction,
                    "supports_prediction": direction == predicted_label_str,
                    "explanation": explanation,
                    "is_unavailable": is_unavailable,
                })

            # Sort by absolute importance descending and select top_k
            contributions.sort(key=lambda x: x["importance"], reverse=True)
            top_contributions = contributions[:top_k]

            return {
                "top_contributions": top_contributions,
                "all_contributions": all_contributions_dict,
                "base_value": round(float(base_val), 4),
                "method": method_name,
                "predicted_label": predicted_label_str,
                "confidence": probabilities.get(predicted_label_str.lower(), 1.0),
            }
        except Exception as e:
            logging.error(f"Error computing explainability: {e}")
            # Resilient fallback so prediction is NEVER aborted
            return {
                "top_contributions": [],
                "all_contributions": {},
                "base_value": 0.5,
                "method": "Unavailable (Error fallback)",
                "predicted_label": LABEL_MAPPING.get(int(predicted_class), "Unknown"),
                "confidence": 1.0,
            }
