"""
app.py
------
Flask web application for the PhishGuard phishing email detector.

Routes
------
  GET  /            -> Paste-email analysis page
  POST /predict     -> Returns verdict + confidence + red flags + URL analysis
  GET  /metrics     -> Model performance metrics page
  GET  /about       -> About page explaining the project

Security improvements over v1
------------------------------
- Input length capped at MAX_EMAIL_CHARS (50 000) to prevent DoS via huge inputs.
- Email text is never reflected back into the page without Jinja2 auto-escaping
  (ensured by using {{ }} in templates, NOT | safe on user input).
- SECRET_KEY loaded from environment variable (falls back to a random key per process).
- Debug mode is disabled by default; only enabled when env var FLASK_DEBUG=1.
- Model artifacts are validated on load (checks expected attributes exist).
- Prediction errors are caught and shown as user-friendly messages, not 500s.
- Added /health endpoint for uptime monitoring and PythonAnywhere keep-alive.

Other changes from v1
---------------------
- predict_email returns richer explanation: per-flag descriptions, verdict icon,
  risk_level string for template use.
- Confidence thresholds slightly adjusted (0.75 / 0.50 / 0.20) based on real-world
  false-positive analysis.
- Added /about route for portfolio/demo use.
- MODEL_INFO keys accessed via .get() throughout to avoid KeyError on old artifacts.
"""

import json
import os
import re
import html
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
from scipy.sparse import csr_matrix, hstack

from feature_extractor import FEATURE_NAMES, extract_features, list_red_flags
from url_analyzer import URL_FEATURE_NAMES, analyze_text_urls, url_summary_features


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"

MAX_EMAIL_CHARS = 50_000          # Hard cap on input size
MIN_EMAIL_CHARS = 10              # Reject trivially short inputs
DEBUG_MODE = os.environ.get("FLASK_DEBUG", "0") == "1"

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(32))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _validate_model(model, vectorizer, scaler):
    """Raise ValueError if loaded artifacts look wrong."""
    if not hasattr(model, "predict_proba"):
        raise ValueError("model.pkl does not have predict_proba method")
    if not hasattr(vectorizer, "transform"):
        raise ValueError("vectorizer.pkl does not have transform method")
    if not hasattr(scaler, "transform"):
        raise ValueError("scaler.pkl does not have transform method")


def _load_artifacts():
    """Load the trained model, vectorizer, scaler, and metadata."""
    model = joblib.load(MODELS_DIR / "model.pkl")
    vectorizer = joblib.load(MODELS_DIR / "vectorizer.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    _validate_model(model, vectorizer, scaler)
    with open(MODELS_DIR / "model_info.json", "r", encoding="utf-8") as f:
        info = json.load(f)
    return model, vectorizer, scaler, info


try:
    MODEL, VECTORIZER, SCALER, MODEL_INFO = _load_artifacts()
    print(f"✅ Loaded model: {MODEL_INFO.get('best_model', 'unknown')}")
except FileNotFoundError:
    MODEL = VECTORIZER = SCALER = MODEL_INFO = None
    print("⚠ Model artifacts not found. Run `python train_model.py` first.")
except Exception as exc:
    MODEL = VECTORIZER = SCALER = MODEL_INFO = None
    print(f"⚠ Failed to load model artifacts: {exc}")


# ---------------------------------------------------------------------------
# Sample emails for the demo
# ---------------------------------------------------------------------------

SAMPLE_EMAILS = [
    {
        "title": "Obvious phishing — bank scare",
        "category": "phishing",
        "body": (
            "Dear Customer,\n\n"
            "Your Chase account has been LOCKED for security reasons. "
            "Click here IMMEDIATELY to verify your password and PIN: "
            "http://chase-unlock.security-alerts.org\n\n"
            "Failure to act within 24 hours will result in permanent suspension!!!"
        ),
    },
    {
        "title": "Subtle phishing — IT pretext",
        "category": "phishing",
        "body": (
            "Hi,\n\n"
            "This is IT Support. We're upgrading the email server tonight. "
            "Please confirm your username and password at the link below so "
            "we can preserve your mailbox: http://outlook-resetpw.cf/reset\n\n"
            "Thanks,\nIT Team"
        ),
    },
    {
        "title": "Phishing — prize scam",
        "category": "phishing",
        "body": (
            "CONGRATULATIONS!!!\n\n"
            "You have WON $1,000,000 in the Microsoft Lottery! "
            "To claim your prize, send your full name, address, and bank details "
            "to claims@microsoft-lottery-winner.tk IMMEDIATELY!\n\n"
            "This offer expires in 24 hours. ACT NOW!"
        ),
    },
    {
        "title": "Legitimate — team meeting",
        "category": "legitimate",
        "body": (
            "Hey team,\n\n"
            "Quick reminder: the design review is moved to Thursday 2pm in "
            "the main conference room. Calendar invite to follow. "
            "Let me know if there are any conflicts.\n\nThanks!"
        ),
    },
    {
        "title": "Legitimate — order confirmation",
        "category": "legitimate",
        "body": (
            "Your Amazon order #112-9283746-1029384 has shipped and is "
            "expected to arrive on Wednesday. You can track your package "
            "in the Amazon app or at amazon.com/orders."
        ),
    },
    {
        "title": "Legitimate — colleague request",
        "category": "legitimate",
        "body": (
            "Hi,\n\n"
            "Hope you are doing well. Could you send me the Q3 budget "
            "spreadsheet when you get a chance? I need it for the board "
            "meeting on Friday.\n\nThanks,\nMike"
        ),
    },
]


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _sanitize_input(text: str) -> str:
    """Strip null bytes and normalize whitespace. Does NOT strip HTML — Jinja handles escaping."""
    text = text.replace("\x00", "")              # Remove null bytes
    text = re.sub(r"\r\n|\r", "\n", text)        # Normalize line endings
    return text.strip()


def _build_feature_vector(text: str):
    """Construct the combined feature matrix the model expects."""
    tfidf = VECTORIZER.transform([text])
    feat_vec = extract_features(text).to_vector()
    url_feat_vec = [url_summary_features(text)[k] for k in URL_FEATURE_NAMES]
    numeric = np.array([feat_vec + url_feat_vec], dtype=float)
    numeric_scaled = SCALER.transform(numeric)
    if MODEL_INFO.get("uses_combined_features", True):
        return hstack([tfidf, csr_matrix(numeric_scaled)])
    return tfidf


def predict_email(text: str) -> dict:
    """
    Run the full prediction pipeline.
    Returns a JSON-serialisable result dict.
    """
    X = _build_feature_vector(text)
    proba = MODEL.predict_proba(X)[0]
    phishing_prob = float(proba[1])

    # Verdict classification with refined thresholds
    if phishing_prob >= 0.75:
        verdict = "Phishing — High Confidence"
        verdict_class = "danger"
        risk_level = "HIGH"
        verdict_icon = "🚨"
    elif phishing_prob >= 0.50:
        verdict = "Likely Phishing"
        verdict_class = "warning"
        risk_level = "MEDIUM"
        verdict_icon = "⚠️"
    elif phishing_prob >= 0.20:
        verdict = "Suspicious — Probably Legitimate"
        verdict_class = "caution"
        risk_level = "LOW"
        verdict_icon = "🔍"
    else:
        verdict = "Legitimate"
        verdict_class = "safe"
        risk_level = "NONE"
        verdict_icon = "✅"

    red_flags = list_red_flags(text)
    url_analyses = [a.to_dict() for a in analyze_text_urls(text)]
    features = extract_features(text).to_dict()
    url_features = url_summary_features(text)

    # Compute a combined risk score (0–100) for display
    url_score = min(url_features.get("max_url_score", 0) / 8, 1.0)
    flag_score = min(len(red_flags) / 6, 1.0)
    combined_risk = round((0.6 * phishing_prob + 0.25 * url_score + 0.15 * flag_score) * 100, 1)

    return {
        "label": int(phishing_prob >= 0.50),
        "phishing_prob": phishing_prob,
        "confidence_pct": round(phishing_prob * 100, 1),
        "legit_pct": round((1 - phishing_prob) * 100, 1),
        "verdict": verdict,
        "verdict_class": verdict_class,
        "risk_level": risk_level,
        "verdict_icon": verdict_icon,
        "combined_risk": combined_risk,
        "red_flags": red_flags,
        "urls": url_analyses,
        "features": features,
        "url_features": url_features,
        "model_used": MODEL_INFO.get("best_model", "unknown").replace("_", " ").title(),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    prefill_idx = request.args.get("sample", type=int)
    prefill_body = ""
    prefill_title = ""
    if prefill_idx is not None and 0 <= prefill_idx < len(SAMPLE_EMAILS):
        prefill_body = SAMPLE_EMAILS[prefill_idx]["body"]
        prefill_title = SAMPLE_EMAILS[prefill_idx]["title"]
    return render_template(
        "index.html",
        samples=SAMPLE_EMAILS,
        prefill_body=prefill_body,
        prefill_title=prefill_title,
        model_ready=MODEL is not None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    # Model not ready
    if MODEL is None:
        return render_template(
            "index.html",
            samples=SAMPLE_EMAILS,
            prefill_body="",
            prefill_title="",
            model_ready=False,
            error="Model not loaded. Run `python train_model.py` first.",
        )

    raw_text = request.form.get("email_text") or ""

    # --- Input validation ---
    if not raw_text.strip():
        return render_template(
            "index.html",
            samples=SAMPLE_EMAILS,
            prefill_body="",
            prefill_title="",
            model_ready=True,
            error="Please paste an email to analyse.",
        )

    if len(raw_text) < MIN_EMAIL_CHARS:
        return render_template(
            "index.html",
            samples=SAMPLE_EMAILS,
            prefill_body=raw_text,
            prefill_title="",
            model_ready=True,
            error=f"Input too short (minimum {MIN_EMAIL_CHARS} characters). Please paste a full email.",
        )

    if len(raw_text) > MAX_EMAIL_CHARS:
        return render_template(
            "index.html",
            samples=SAMPLE_EMAILS,
            prefill_body=raw_text[:200],
            prefill_title="",
            model_ready=True,
            error=f"Input too long (maximum {MAX_EMAIL_CHARS:,} characters). Please trim the email.",
        )

    email_text = _sanitize_input(raw_text)

    # --- Prediction (with graceful error handling) ---
    try:
        result = predict_email(email_text)
    except Exception as exc:
        app.logger.error("Prediction error: %s", exc, exc_info=True)
        return render_template(
            "index.html",
            samples=SAMPLE_EMAILS,
            prefill_body=email_text,
            prefill_title="",
            model_ready=True,
            error="An error occurred during analysis. Please try again.",
        )

    return render_template("result.html", email=email_text, result=result)


@app.route("/metrics")
def metrics():
    return render_template("metrics.html", info=MODEL_INFO)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/health")
def health():
    """Lightweight health-check endpoint for uptime monitoring."""
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL is not None,
        "model": MODEL_INFO.get("best_model") if MODEL_INFO else None,
    })


# ---------------------------------------------------------------------------
# Dev entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=DEBUG_MODE)
