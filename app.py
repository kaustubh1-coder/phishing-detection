import sys
import os
import json
import base64
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest, urlopen

import certifi
from bson import ObjectId

ca = certifi.where()

from dotenv import load_dotenv

load_dotenv()
mongo_db_url = os.getenv("MONGODB_URL_KEY")
import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from uvicorn import run as app_run
import pandas as pd

from networksecurity.constants.training_pipeline import (
    CANONICAL_FEATURE_NAMES,
    DATA_INGESTION_COLLECTION_NAME,
    DATA_INGESTION_DATABASE_NAME,
    LABEL_MAPPING,
    TARGET_COLUMN,
)
from networksecurity.pipeline.batch_prediction import PredictionPipeline
from networksecurity.utils.url_feature_extractor import extract_url_features


def create_mongo_client(uri: str | None):
    timeout_ms = int(os.getenv("MONGO_TIMEOUT_MS", "5000"))
    if not uri:
        return pymongo.MongoClient(serverSelectionTimeoutMS=timeout_ms)
    uri_lower = (uri or "").lower()
    uses_tls = (
        uri_lower.startswith("mongodb+srv://")
        or "tls=true" in uri_lower
        or "ssl=true" in uri_lower
    )
    if uses_tls:
        return pymongo.MongoClient(uri, tlsCAFile=ca, serverSelectionTimeoutMS=timeout_ms)
    return pymongo.MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)


client = create_mongo_client(mongo_db_url)



database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
TEMPLATE_DIR = BASE_DIR / "templates"
MODEL_PATH = BASE_DIR / "final_model" / "model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "final_model" / "preprocessor.pkl"
MODEL_METADATA_JSON_PATH = BASE_DIR / "final_model" / "model_metadata.json"
MODEL_COMPARISON_JSON_PATH = BASE_DIR / "final_model" / "model_comparison.json"
MODEL_COMPARISON_CSV_PATH = BASE_DIR / "final_model" / "model_comparison.csv"
PREDICTION_OUTPUT_PATH = BASE_DIR / "prediction_output" / "output.csv"
SAMPLE_DATA_PATH = BASE_DIR / "Network_Data" / "phisingData.csv"
DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER", "Sahilpatil009")
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME", "network-security-edi")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_TIMEOUT_SECONDS = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "5.0"))
URL_PREDICTION_COLLECTION_NAME = os.getenv(
    "URL_PREDICTION_COLLECTION_NAME", "url_prediction_history"
)
AUTH_USER_COLLECTION_NAME = os.getenv("AUTH_USER_COLLECTION_NAME", "users")
AUTH_SESSION_COLLECTION_NAME = os.getenv("AUTH_SESSION_COLLECTION_NAME", "user_sessions")
AUTH_SESSION_DAYS = int(os.getenv("AUTH_SESSION_DAYS", "7"))
url_prediction_collection = database[URL_PREDICTION_COLLECTION_NAME]
auth_user_collection = database[AUTH_USER_COLLECTION_NAME]
auth_session_collection = database[AUTH_SESSION_COLLECTION_NAME]

cors_origins_env = os.getenv("CORS_ORIGINS", "*")
if cors_origins_env.strip() == "*":
    origins = ["*"]
else:
    origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AuthSignupPayload(BaseModel):
    name: str
    email: str
    password: str


class AuthLoginPayload(BaseModel):
    email: str
    password: str


def render_template(template_name: str, **context: str) -> HTMLResponse:
    template = (TEMPLATE_DIR / template_name).read_text(encoding="utf-8")
    for key, value in context.items():
        template = template.replace("{{ " + key + " }}", str(value))
    return HTMLResponse(content=template)


def file_meta(path: Path) -> str:
    if not path.exists():
        return "Not found"
    size_kb = path.stat().st_size / 1024
    modified = datetime.fromtimestamp(path.stat().st_mtime).strftime(
        "%d %b %Y, %I:%M %p"
    )
    return f"{size_kb:,.0f} KB | {modified}"


def status_class(is_ready: bool) -> str:
    return "good" if is_ready else "warn"


def normalize_email(email: str) -> str:
    return email.strip().lower()


def hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        120_000,
    )
    encoded_digest = base64.b64encode(digest).decode("utf-8")
    return f"pbkdf2_sha256${salt}${encoded_digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, salt, expected_digest = stored_hash.split("$", 2)
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False
    candidate_digest = hash_password(password, salt).split("$", 2)[2]
    return hmac.compare_digest(candidate_digest, expected_digest)


def validate_auth_payload(email: str, password: str, name: str | None = None):
    if not email or "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    if name is not None and not name.strip():
        raise HTTPException(status_code=400, detail="Enter your name.")


def serialize_auth_user(document: dict) -> dict:
    created_at = document.get("createdAt")
    if isinstance(created_at, datetime):
        created_at = created_at.isoformat()
    return {
        "id": str(document.get("_id", "")),
        "name": document.get("name", ""),
        "email": document.get("email", ""),
        "createdAt": created_at or "",
    }


def create_auth_session(user_id: str) -> dict:
    token = secrets.token_urlsafe(32)
    created_at = datetime.now(timezone.utc)
    expires_at = created_at + timedelta(days=AUTH_SESSION_DAYS)
    auth_session_collection.insert_one(
        {
            "token": token,
            "userId": user_id,
            "createdAt": created_at,
            "expiresAt": expires_at,
        }
    )
    return {"token": token, "expiresAt": expires_at.isoformat()}


def extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        return None
    return authorization[len(prefix) :].strip()


def get_user_from_authorization(authorization: str | None) -> dict | None:
    token = extract_bearer_token(authorization)
    if not token:
        return None

    session = auth_session_collection.find_one({"token": token})
    if not session:
        return None

    expires_at = session.get("expiresAt")
    if isinstance(expires_at, datetime):
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at < datetime.now(timezone.utc):
            auth_session_collection.delete_one({"_id": session.get("_id")})
            return None

    try:
        user_id = ObjectId(session.get("userId"))
    except Exception:
        return None
    return auth_user_collection.find_one({"_id": user_id})


def require_authenticated_user(authorization: str | None) -> dict:
    user_document = get_user_from_authorization(authorization)
    if not user_document:
        raise HTTPException(status_code=401, detail="Sign in to continue.")
    return user_document


prediction_pipeline = PredictionPipeline(
    model_path=str(MODEL_PATH),
    preprocessor_path=str(PREPROCESSOR_PATH),
    metadata_path=str(MODEL_METADATA_JSON_PATH),
)


def prediction_label(value: int) -> str:
    return prediction_pipeline.get_label_mapping().get(
        int(value), "Phishing" if int(value) == 0 else "Legitimate"
    )


def prediction_class(value: int) -> str:
    return "good" if int(value) == 1 else "danger"


def feature_rows(features: dict) -> str:
    rows = []
    for name, value in features.items():
        if pd.isna(value) or value is None:
            signal = "Neutral / Imputed"
            cls = "warn"
            display_val = "N/A"
        elif value == -1:
            signal = "Suspicious"
            cls = "danger"
            display_val = str(int(value))
        elif value == 0:
            signal = "Neutral"
            cls = "warn"
            display_val = str(int(value))
        else:
            signal = "Normal"
            cls = "good"
            display_val = str(int(value))
        rows.append(
            "<tr>"
            f"<td>{escape(name)}</td>"
            f"<td>{display_val}</td>"
            f'<td><span class="badge {cls}">{signal}</span></td>'
            "</tr>"
        )
    return "".join(rows)


def contributions_rows(contributions: list) -> str:
    if not contributions:
        return '<tr><td colspan="4" class="muted">No feature contribution details available.</td></tr>'
    rows = []
    for c in contributions:
        fname = escape(str(c.get("feature_name", c.get("feature", ""))))
        disp_val = escape(str(c.get("display_value", c.get("raw_value", ""))))
        direction = escape(str(c.get("direction", "")))
        shap_val = c.get("shap_value", 0.0)
        exp = escape(str(c.get("explanation", "")))
        badge_cls = "good" if direction.lower() == "legitimate" else "danger"
        rows.append(
            "<tr>"
            f"<td><strong>{fname}</strong><br><span style='font-size:11px;color:var(--muted);'>{exp}</span></td>"
            f"<td>{disp_val}</td>"
            f"<td><code>{shap_val:+.4f}</code></td>"
            f'<td><span class="badge {badge_cls}">{direction}</span></td>'
            "</tr>"
        )
    return "".join(rows)


def local_prediction_summary(
    url: str,
    label: str,
    features: dict,
    metadata: dict | None = None,
    confidence: float | None = None,
    probabilities: dict | None = None,
    top_contributions: list | None = None,
) -> str:
    metadata = metadata or {}
    top_contributions = top_contributions or []
    conf_str = f" with {confidence * 100:.1f}% confidence" if confidence is not None else ""

    if top_contributions:
        top_reasons = []
        for tc in top_contributions[:3]:
            fname = tc.get("feature_name", tc.get("feature"))
            direction = tc.get("direction", "")
            shap_val = tc.get("shap_value", 0.0)
            sign = "+" if shap_val > 0 else ""
            top_reasons.append(f"{fname} ({direction}, impact {sign}{shap_val:.3f})")
        reason = f"Primary decision drivers identified by SHAP: {'; '.join(top_reasons)}."
    else:
        suspicious_features = [
            name for name, value in features.items() if not pd.isna(value) and value == -1
        ]
        neutral_features = [
            name for name, value in features.items() if pd.isna(value) or value == 0
        ]
        if suspicious_features:
            reason = f"Suspicious signals include {', '.join(suspicious_features[:5])}."
        elif neutral_features:
            reason = f"Most URL signals look normal, with uncertain or imputed values for {', '.join(neutral_features[:4])}."
        else:
            reason = "Most extracted URL signals look normal."

    unavail_note = ""
    warning = metadata.get("unavailable_warning") or metadata.get("unavailableWarning")
    if warning:
        unavail_note = f" Note: {warning}"

    return (
        f"The model predicted this URL as {label}{conf_str}. {reason}{unavail_note} "
        "Treat this as an ML-assisted check and manually verify the domain before sharing passwords or payment details."
    )


def explain_prediction_with_gemini(
    url: str,
    label: str,
    features: dict,
    metadata: dict,
    confidence: float | None = None,
    probabilities: dict | None = None,
    top_contributions: list | None = None,
) -> str:
    """
    Explains the already-predicted ML result using Gemini strictly grounded in verified evidence.
    Gemini does NOT determine or influence the ML prediction.
    If Gemini times out or fails, returns a fallback explanation safely.
    """
    fallback = local_prediction_summary(
        url=url,
        label=label,
        features=features,
        metadata=metadata,
        confidence=confidence,
        probabilities=probabilities,
        top_contributions=top_contributions,
    )
    if not GEMINI_API_KEY:
        return fallback

    conf_pct = f"{confidence * 100:.1f}%" if confidence is not None else "High"
    prob_p = probabilities.get("phishing", 0.0) if probabilities else 0.0
    prob_l = probabilities.get("legitimate", 0.0) if probabilities else 0.0

    contrib_lines = []
    if top_contributions:
        for idx, tc in enumerate(top_contributions[:5], 1):
            fname = tc.get("feature_name", tc.get("feature"))
            val_desc = tc.get("display_value", str(tc.get("raw_value")))
            exp = tc.get("explanation", "")
            shap_val = tc.get("shap_value", 0)
            direction = tc.get("direction", "")
            contrib_lines.append(
                f"{idx}. {fname} [{val_desc}]: {exp} (Impact: {shap_val:+.4f} towards {direction})"
            )
    contrib_text = "\n".join(contrib_lines) if contrib_lines else "Standard URL feature distribution."

    unavail_text = metadata.get("unavailable_warning") or "All essential signals verified."

    prompt = (
        "You are an Explainable AI assistant explaining an already-computed Machine Learning phishing prediction. "
        "Your explanation must be strictly grounded in the verified data below. "
        "DO NOT contradict the model's prediction or probability. "
        "Keep your response clear, concise, and under 110 words.\n\n"
        f"VERIFIED ML RESULTS:\n"
        f"- Target URL: {url}\n"
        f"- Model Prediction: {label}\n"
        f"- Genuine Confidence: {conf_pct} (Phishing Probability: {prob_p:.3f}, Legitimate Probability: {prob_l:.3f})\n"
        f"- Hostname: {metadata.get('hostname')}\n"
        f"- Page Fetch: {metadata.get('html_status')}\n"
        f"- Unavailable Features / Robustness: {unavail_text}\n"
        f"- Top Contributing Features (SHAP Analysis):\n{contrib_text}\n\n"
        "INSTRUCTIONS:\n"
        "1. State the model's prediction and confidence.\n"
        "2. Explain the top 2-3 feature contributions in plain English.\n"
        "3. If webpage signals were unavailable, clarify that missing signals were safely imputed without false alarm.\n"
        "4. Provide one practical safety advice for the user."
    )

    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 220},
    }
    request = UrlRequest(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY,
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=GEMINI_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode("utf-8"))
        parts = data["candidates"][0]["content"]["parts"]
        summary = " ".join(part.get("text", "") for part in parts).strip()
        if len(summary) < 60 or summary.lower().rstrip(". ").endswith(
            ("because", "include", "with")
        ):
            return fallback
        return summary
    except Exception as e:
        logging.warning(
            f"Gemini API explanation failed or timed out: {e}. Using fallback explanation."
        )
        return fallback


def predict_dataframe(input_df: pd.DataFrame) -> dict:
    """
    Delegates to the reusable PredictionPipeline, executing validation, logging,
    and single-pass transformation and prediction.
    """
    return prediction_pipeline.predict(input_df)


def build_url_prediction(url: str) -> dict:
    """
    Executes the canonical prediction pipeline for a single URL:
    URL -> Canonical feature extractor -> Canonical DataFrame -> Saved preprocessor
    -> Saved model -> Prediction -> Probability -> Structured Response.

    Gemini is only invoked after the ML prediction is complete, strictly for explanation.
    """
    canonical_result = prediction_pipeline.predict_url(url)
    features_dict = canonical_result["features"]
    url_metadata = canonical_result["metadata"]
    top_contributions = canonical_result.get("top_contributions", [])
    confidence = canonical_result["confidence"]
    probabilities = canonical_result["probabilities"]

    # Gemini explanation (strictly evidence-grounded with fallback)
    summary = explain_prediction_with_gemini(
        url=canonical_result["url"],
        label=canonical_result["prediction"],
        features=features_dict,
        metadata=url_metadata,
        confidence=confidence,
        probabilities=probabilities,
        top_contributions=top_contributions,
    )

    suspicious_count = sum(
        1 for value in features_dict.values() if value is not None and value == -1
    )
    neutral_count = sum(
        1 for value in features_dict.values() if value is None or value == 0
    )
    normal_count = sum(
        1 for value in features_dict.values() if value is not None and value == 1
    )
    suspicious_features = [
        name for name, value in features_dict.items() if value is not None and value == -1
    ]

    unavailable_warning = url_metadata.get("unavailable_warning")

    return {
        # Core structured API response (Requirement 10)
        "url": canonical_result["url"],
        "prediction": canonical_result["prediction"],
        "label": canonical_result["label"],
        "confidence": canonical_result["confidence"],
        "probabilities": canonical_result["probabilities"],
        "risk_level": canonical_result["risk_level"],
        "features": canonical_result["features"],
        # Explainable AI & missing-feature robustness fields
        "top_contributions": top_contributions,
        "topContributions": top_contributions,
        "explainability": canonical_result.get("explainability", {}),
        "unavailableWarning": unavailable_warning,
        "unavailable_warning": unavailable_warning,
        "missingFeaturesCount": url_metadata.get("missing_features_count", 0),
        "availableFeaturesCount": url_metadata.get("available_features_count", 30),
        # UI & integration metadata fields
        "finalUrl": url_metadata.get("final_url", canonical_result["url"]),
        "hostname": url_metadata.get("hostname", ""),
        "dnsStatus": url_metadata.get("dns_status", "Available"),
        "htmlStatus": url_metadata.get("html_status", "Available"),
        "statusClass": prediction_class(canonical_result["label"]),
        "summary": summary,
        "signals": {
            "suspicious": suspicious_count,
            "neutral": neutral_count,
            "normal": normal_count,
        },
        "suspiciousFeatures": suspicious_features,
    }


def save_url_prediction(result: dict, user: dict | None = None) -> dict:
    saved_at = datetime.now(timezone.utc)
    document = {
        "url": result["url"],
        "finalUrl": result.get("finalUrl", result["url"]),
        "hostname": result.get("hostname", ""),
        "dnsStatus": result.get("dnsStatus", "Available"),
        "htmlStatus": result.get("htmlStatus", "Available"),
        "prediction": result["prediction"],
        "label": result["label"],
        "statusClass": result.get("statusClass", "good" if result["label"] == 1 else "danger"),
        "confidence": result["confidence"],
        "probabilities": result.get("probabilities", {}),
        "risk_level": result.get("risk_level", "Low"),
        "summary": result.get("summary", ""),
        "signals": result.get("signals", {"suspicious": 0, "neutral": 0, "normal": 0}),
        "features": result.get("features", {}),
        "suspiciousFeatures": result.get("suspiciousFeatures", []),
        "topContributions": result.get("topContributions", []),
        "unavailableWarning": result.get("unavailableWarning"),
        "missingFeaturesCount": result.get("missingFeaturesCount", 0),
        "createdAt": saved_at,
    }
    if user:
        document["userId"] = str(user["_id"])
        document["userEmail"] = user.get("email", "")

    try:
        inserted = url_prediction_collection.insert_one(document)
        result["historyId"] = str(inserted.inserted_id)
    except Exception as e:
        logging.warning(f"Could not persist prediction to MongoDB: {e}")

    result["savedAt"] = saved_at.isoformat()
    return result


def serialize_url_prediction_history(document: dict) -> dict:
    created_at = document.get("createdAt")
    if isinstance(created_at, datetime):
        created_at = created_at.isoformat()

    label_val = document.get("label", 0)
    prediction_val = document.get("prediction", "Phishing")

    return {
        "historyId": str(document.get("_id", "")),
        "url": document.get("url", ""),
        "finalUrl": document.get("finalUrl", ""),
        "hostname": document.get("hostname", ""),
        "dnsStatus": document.get("dnsStatus", ""),
        "htmlStatus": document.get("htmlStatus", ""),
        "prediction": prediction_val,
        "label": label_val,
        "statusClass": document.get("statusClass", "danger"),
        "confidence": float(document.get("confidence", 0)),
        "probabilities": document.get("probabilities", {}),
        "risk_level": document.get("risk_level", "Low"),
        "summary": document.get("summary", ""),
        "signals": document.get(
            "signals", {"suspicious": 0, "neutral": 0, "normal": 0}
        ),
        "features": document.get("features", {}),
        "suspiciousFeatures": document.get("suspiciousFeatures", []),
        "createdAt": created_at or "",
    }


def get_url_prediction_history(limit: int = 10, user: dict | None = None) -> list:
    if not user:
        return []

    safe_limit = max(1, min(limit, 50))
    query = {"userId": str(user["_id"])}
    documents = (
        url_prediction_collection.find(query)
        .sort("createdAt", pymongo.DESCENDING)
        .limit(safe_limit)
    )
    return [serialize_url_prediction_history(document) for document in documents]


def build_model_comparison_payload() -> dict:
    if not MODEL_COMPARISON_JSON_PATH.exists():
        return {
            "ready": False,
            "generatedAt": "",
            "bestModelName": "",
            "bestModelScore": 0,
            "models": [],
            "message": "Run model training to generate a model comparison report.",
        }

    with open(MODEL_COMPARISON_JSON_PATH, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    best_model = next(
        (model for model in payload.get("models", []) if model.get("isBest")), None
    )
    payload["ready"] = True
    payload["bestModelName"] = payload.get("bestModelName") or (best_model or {}).get(
        "modelName", ""
    )
    payload["bestModelScore"] = payload.get("bestModelScore") or (best_model or {}).get(
        "testF1", 0
    )
    payload["reportMeta"] = file_meta(MODEL_COMPARISON_CSV_PATH)
    return payload


def build_status_payload() -> dict:
    model_ready = MODEL_PATH.exists() and PREPROCESSOR_PATH.exists()
    mongo_ready = bool(mongo_db_url)
    sample_ready = SAMPLE_DATA_PATH.exists()
    output_ready = PREDICTION_OUTPUT_PATH.exists()
    model_comparison_ready = MODEL_COMPARISON_JSON_PATH.exists()
    return {
        "model": {
            "status": "Ready" if model_ready else "Missing",
            "ready": model_ready,
            "meta": file_meta(MODEL_PATH),
        },
        "preprocessor": {
            "status": "Ready" if PREPROCESSOR_PATH.exists() else "Missing",
            "ready": PREPROCESSOR_PATH.exists(),
            "meta": file_meta(PREPROCESSOR_PATH),
        },
        "mongo": {
            "status": "Configured" if mongo_ready else "Missing",
            "ready": mongo_ready,
            "meta": mongo_db_url or "Set MONGODB_URL_KEY in .env",
        },
        "sampleData": {
            "status": "Available" if sample_ready else "Missing",
            "ready": sample_ready,
            "meta": file_meta(SAMPLE_DATA_PATH),
        },
        "output": {
            "status": "Available" if output_ready else "No predictions yet",
            "ready": output_ready,
            "meta": file_meta(PREDICTION_OUTPUT_PATH),
        },
        "modelComparison": {
            "status": "Available" if model_comparison_ready else "Not generated",
            "ready": model_comparison_ready,
            "meta": file_meta(MODEL_COMPARISON_JSON_PATH),
        },
        "gemini": {
            "status": "Configured" if GEMINI_API_KEY else "Missing",
            "ready": bool(GEMINI_API_KEY),
            "meta": GEMINI_MODEL if GEMINI_API_KEY else "Set GEMINI_API_KEY in .env",
        },
        "repo": {
            "name": f"{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}",
            "url": f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}",
        },
    }


@app.get("/", tags=["ui"])
async def index():
    status_payload = build_status_payload()

    return render_template(
        "index.html",
        MODEL_STATUS=status_payload["model"]["status"],
        MODEL_STATUS_CLASS=status_class(status_payload["model"]["ready"]),
        MODEL_META=escape(status_payload["model"]["meta"]),
        PREPROCESSOR_META=escape(status_payload["preprocessor"]["meta"]),
        MONGO_STATUS=status_payload["mongo"]["status"],
        MONGO_STATUS_CLASS=status_class(status_payload["mongo"]["ready"]),
        MONGO_META=escape(status_payload["mongo"]["meta"]),
        SAMPLE_STATUS=status_payload["sampleData"]["status"],
        SAMPLE_STATUS_CLASS=status_class(status_payload["sampleData"]["ready"]),
        SAMPLE_META=escape(status_payload["sampleData"]["meta"]),
        OUTPUT_STATUS=status_payload["output"]["status"],
        OUTPUT_STATUS_CLASS=status_class(status_payload["output"]["ready"]),
        OUTPUT_META=escape(status_payload["output"]["meta"]),
        DAGSHUB_REPO=escape(status_payload["repo"]["name"]),
        DAGSHUB_URL=escape(status_payload["repo"]["url"]),
        GEMINI_STATUS=status_payload["gemini"]["status"],
        GEMINI_STATUS_CLASS=status_class(status_payload["gemini"]["ready"]),
        GEMINI_META=escape(status_payload["gemini"]["meta"]),
    )


@app.get("/api/status")
async def api_status(authorization: str | None = Header(default=None)):
    require_authenticated_user(authorization)
    return build_status_payload()


@app.post("/api/auth/signup")
async def api_auth_signup(payload: AuthSignupPayload):
    email = normalize_email(payload.email)
    validate_auth_payload(email=email, password=payload.password, name=payload.name)

    if auth_user_collection.find_one({"email": email}):
        raise HTTPException(status_code=409, detail="An account already exists for this email.")

    created_at = datetime.now(timezone.utc)
    user_document = {
        "name": payload.name.strip(),
        "email": email,
        "passwordHash": hash_password(payload.password),
        "createdAt": created_at,
    }
    inserted = auth_user_collection.insert_one(user_document)
    user_document["_id"] = inserted.inserted_id
    session = create_auth_session(str(inserted.inserted_id))
    return {"user": serialize_auth_user(user_document), **session}


@app.post("/api/auth/login")
async def api_auth_login(payload: AuthLoginPayload):
    email = normalize_email(payload.email)
    validate_auth_payload(email=email, password=payload.password)

    user_document = auth_user_collection.find_one({"email": email})
    if not user_document or not verify_password(payload.password, user_document.get("passwordHash", "")):
        raise HTTPException(status_code=401, detail="Email or password is incorrect.")

    session = create_auth_session(str(user_document["_id"]))
    return {"user": serialize_auth_user(user_document), **session}


@app.get("/api/auth/me")
async def api_auth_me(authorization: str | None = Header(default=None)):
    user_document = require_authenticated_user(authorization)
    return {"user": serialize_auth_user(user_document)}


@app.post("/api/auth/logout")
async def api_auth_logout(authorization: str | None = Header(default=None)):
    token = extract_bearer_token(authorization)
    if token:
        auth_session_collection.delete_one({"token": token})
    return {"ok": True}


@app.get("/api/prediction-history")
async def api_prediction_history(limit: int = 10, authorization: str | None = Header(default=None)):
    try:
        user_document = require_authenticated_user(authorization)
        return {
            "items": get_url_prediction_history(limit, user_document),
            "requiresAuth": False,
        }
    except HTTPException:
        raise
    except Exception as e:
        error = NetworkSecurityException(e, sys)
        return {
            "error": str(error),
            "message": "Could not load URL prediction history from MongoDB.",
            "items": [],
        }


@app.get("/api/model-comparison")
async def api_model_comparison(authorization: str | None = Header(default=None)):
    try:
        require_authenticated_user(authorization)
        return build_model_comparison_payload()
    except HTTPException:
        raise
    except Exception as e:
        error = NetworkSecurityException(e, sys)
        return {
            "ready": False,
            "error": str(error),
            "message": "Could not load the model comparison report.",
            "models": [],
        }


@app.get("/api/model-robustness")
async def api_model_robustness():
    report_path = os.path.join("final_model", "robustness_evaluation.json")
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read robustness evaluation report: {e}")
    raise HTTPException(status_code=404, detail="Robustness evaluation report not found.")


@app.get("/train")
async def train_route():
    try:
        from networksecurity.pipeline.training_pipeline import TrainingPipeline

        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return render_template(
            "message.html",
            STATUS_CLASS="good",
            TITLE="Training completed",
            MESSAGE="The training pipeline finished successfully. New model artifacts are available for prediction.",
            DETAIL=escape(
                f"MLflow tracking repo: {DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}"
            ),
            PRIMARY_ACTION="/",
            PRIMARY_LABEL="Back to dashboard",
            SECONDARY_ACTION="/docs",
            SECONDARY_LABEL="Open API docs",
        )
    except Exception as e:
        error = NetworkSecurityException(e, sys)
        return render_template(
            "message.html",
            STATUS_CLASS="danger",
            TITLE="Training failed",
            MESSAGE="The training pipeline could not finish.",
            DETAIL=escape(str(error)),
            PRIMARY_ACTION="/",
            PRIMARY_LABEL="Back to dashboard",
            SECONDARY_ACTION="/docs",
            SECONDARY_LABEL="Open API docs",
        )


@app.post("/predict")
async def predict_route(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".csv"):
            raise ValueError("Uploaded file must be a CSV file.")
        df = pd.read_csv(file.file)
        input_df = df.drop(columns=[TARGET_COLUMN], errors="ignore")
        res = predict_dataframe(input_df)
        df["predicted_column"] = res["predictions"]
        label_map = prediction_pipeline.get_label_mapping()
        df["predicted_label"] = [
            label_map.get(int(p), "Phishing" if int(p) == 0 else "Legitimate")
            for p in res["predictions"]
        ]

        PREDICTION_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PREDICTION_OUTPUT_PATH, index=False)

        preview_df = df.head(200)
        table_html = preview_df.to_html(classes="data-table", index=False)
        predicted_counts = df["predicted_column"].value_counts().to_dict()
        phishing_count = int(predicted_counts.get(0, 0))
        legitimate_count = int(predicted_counts.get(1, 0))

        return render_template(
            "results.html",
            FILE_NAME=escape(file.filename or "Uploaded CSV"),
            ROW_COUNT=f"{len(df):,}",
            COLUMN_COUNT=f"{len(df.columns):,}",
            PHISHING_COUNT=f"{phishing_count:,}",
            LEGITIMATE_COUNT=f"{legitimate_count:,}",
            TABLE_HTML=table_html,
            PREVIEW_NOTE=escape(
                "Showing first 200 rows." if len(df) > 200 else "Showing all rows."
            ),
            OUTPUT_META=escape(file_meta(PREDICTION_OUTPUT_PATH)),
        )

    except ValueError as e:
        return render_template(
            "message.html",
            STATUS_CLASS="warn",
            TITLE="Invalid CSV schema",
            MESSAGE="The uploaded CSV does not match the canonical feature schema.",
            DETAIL=escape(str(e)),
            PRIMARY_ACTION="/",
            PRIMARY_LABEL="Back to dashboard",
            SECONDARY_ACTION="/docs",
            SECONDARY_LABEL="Open API docs",
        )
    except FileNotFoundError as e:
        return render_template(
            "message.html",
            STATUS_CLASS="danger",
            TITLE="Artifact missing",
            MESSAGE="The model or preprocessor artifact could not be found.",
            DETAIL=escape(str(e)),
            PRIMARY_ACTION="/",
            PRIMARY_LABEL="Back to dashboard",
            SECONDARY_ACTION="/docs",
            SECONDARY_LABEL="Open API docs",
        )
    except Exception as e:
        error = NetworkSecurityException(e, sys)
        return render_template(
            "message.html",
            STATUS_CLASS="danger",
            TITLE="Prediction failed",
            MESSAGE="The CSV could not be processed.",
            DETAIL=escape(str(error)),
            PRIMARY_ACTION="/",
            PRIMARY_LABEL="Back to dashboard",
            SECONDARY_ACTION="/docs",
            SECONDARY_LABEL="Open API docs",
        )


@app.post("/predict-url")
async def predict_url_route(url: str = Form(...)):
    try:
        result = build_url_prediction(url)
        result = save_url_prediction(result)
        features = result["features"]

        return render_template(
            "url_result.html",
            URL=escape(result["url"]),
            FINAL_URL=escape(result["finalUrl"]),
            HOSTNAME=escape(result["hostname"]),
            DNS_STATUS=escape(result["dnsStatus"]),
            HTML_STATUS=escape(result["htmlStatus"]),
            RESULT_LABEL=escape(str(result["prediction"])),
            RESULT_CLASS=result["statusClass"],
            RESULT_VALUE=str(result["label"]),
            CONFIDENCE_PCT=f"{result.get('confidence', 1.0) * 100:.1f}%",
            RISK_LEVEL=escape(result.get("risk_level", "Unknown")),
            RISK_CLASS="good" if result.get("risk_level") == "Low" else ("warn" if result.get("risk_level") == "Medium" else "danger"),
            GEMINI_SUMMARY=escape(result["summary"]),
            TOP_CONTRIBUTIONS_ROWS=contributions_rows(result.get("top_contributions", [])),
            UNAVAILABLE_WARNING=escape(result.get("unavailableWarning") or ""),
            HAS_UNAVAILABLE_WARNING="display:block;" if result.get("unavailableWarning") else "display:none;",
            FEATURE_ROWS=feature_rows(features),
        )
    except ValueError as e:
        return render_template(
            "message.html",
            STATUS_CLASS="warn",
            TITLE="Invalid URL",
            MESSAGE="The website URL could not be processed.",
            DETAIL=escape(str(e)),
            PRIMARY_ACTION="/",
            PRIMARY_LABEL="Back to dashboard",
            SECONDARY_ACTION="/docs",
            SECONDARY_LABEL="Open API docs",
        )
    except FileNotFoundError as e:
        return render_template(
            "message.html",
            STATUS_CLASS="danger",
            TITLE="Artifact missing",
            MESSAGE="The model or preprocessor artifact could not be found.",
            DETAIL=escape(str(e)),
            PRIMARY_ACTION="/",
            PRIMARY_LABEL="Back to dashboard",
            SECONDARY_ACTION="/docs",
            SECONDARY_LABEL="Open API docs",
        )
    except Exception as e:
        error = NetworkSecurityException(e, sys)
        return render_template(
            "message.html",
            STATUS_CLASS="danger",
            TITLE="URL prediction failed",
            MESSAGE="The website URL could not be converted into model features.",
            DETAIL=escape(str(error)),
            PRIMARY_ACTION="/",
            PRIMARY_LABEL="Back to dashboard",
            SECONDARY_ACTION="/docs",
            SECONDARY_LABEL="Open API docs",
        )


@app.post("/api/predict-url")
async def api_predict_url(url: str = Form(...), authorization: str | None = Header(default=None)):
    try:
        user_document = require_authenticated_user(authorization)
        result = build_url_prediction(url)
        return save_url_prediction(result, user_document)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        error = NetworkSecurityException(e, sys)
        logging.error(f"URL prediction error: {error}")
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/api/predict-csv")
async def api_predict_csv(file: UploadFile = File(...), authorization: str | None = Header(default=None)):
    try:
        require_authenticated_user(authorization)
        if not file.filename.endswith(".csv"):
            raise ValueError("Uploaded file must be a CSV file.")
        df = pd.read_csv(file.file)
        input_df = df.drop(columns=[TARGET_COLUMN], errors="ignore")
        res = predict_dataframe(input_df)
        df["predicted_column"] = res["predictions"]
        label_map = prediction_pipeline.get_label_mapping()
        df["predicted_label"] = [
            label_map.get(int(p), "Phishing" if int(p) == 0 else "Legitimate")
            for p in res["predictions"]
        ]

        PREDICTION_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PREDICTION_OUTPUT_PATH, index=False)

        predicted_counts = df["predicted_column"].value_counts().to_dict()
        preview = df.head(8).to_dict(orient="records")
        return {
            "fileName": file.filename or "Uploaded CSV",
            "rows": len(df),
            "columns": len(df.columns),
            "phishing": int(predicted_counts.get(0, 0)),
            "legitimate": int(predicted_counts.get(1, 0)),
            "outputMeta": file_meta(PREDICTION_OUTPUT_PATH),
            "preview": preview,
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        error = NetworkSecurityException(e, sys)
        logging.error(f"Batch prediction error: {error}")
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/download-output")
async def download_output():
    if not PREDICTION_OUTPUT_PATH.exists():
        return render_template(
            "message.html",
            STATUS_CLASS="warn",
            TITLE="No output file yet",
            MESSAGE="Run a prediction first to generate prediction_output/output.csv.",
            DETAIL="",
            PRIMARY_ACTION="/",
            PRIMARY_LABEL="Back to dashboard",
            SECONDARY_ACTION="/docs",
            SECONDARY_LABEL="Open API docs",
        )
    return FileResponse(
        PREDICTION_OUTPUT_PATH,
        media_type="text/csv",
        filename="network_security_predictions.csv",
    )


@app.get("/sample-data")
async def sample_data():
    return FileResponse(
        SAMPLE_DATA_PATH,
        media_type="text/csv",
        filename="phisingData.csv",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app_run(app, host="0.0.0.0", port=port)
