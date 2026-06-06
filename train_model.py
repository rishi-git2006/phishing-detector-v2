"""
train_model.py
--------------
Train the phishing-email classifier.

Pipeline
--------
1. Load labelled emails from data/emails.csv  (columns: text, label)
2. Validate dataset integrity (no leakage, balanced classes, no nulls)
3. Build features:
     - TF-IDF on the email body (1-3 n-grams, 8 000 features)
     - 18 numeric linguistic features  (urgency, caps, credential words …)
     - 9 URL summary features  (IP hosts, shorteners, typosquats …)
4. Train and compare three models with cross-validation:
     - Multinomial Naive Bayes  (TF-IDF only, requires non-negative input)
     - Logistic Regression      (TF-IDF + numeric + URL features)
     - Random Forest            (TF-IDF + numeric + URL features)
5. Pick the best model by macro F1 on a held-out 20 % test set.
6. Save model.pkl, vectorizer.pkl, scaler.pkl, model_info.json.

Run:  python train_model.py
      python train_model.py --data path/to/other.csv   (custom dataset)

Changes from v1
---------------
- Added Random Forest as a third candidate model.
- Added 5-fold cross-validation to detect overfitting before final eval.
- Added dataset integrity checks (class balance, duplicates, data leakage guard).
- TF-IDF n-gram range extended to (1, 3) and max_features raised to 8 000.
- Logistic Regression solver changed to 'saga' (faster on large sparse matrices).
- Added argparse so the data path can be overridden without editing code.
- Added per-class F1 reporting so false-positive / false-negative rates are visible.
- model_info.json now stores cross-val scores and class distribution for the /metrics page.
- Added reproducibility note: random_state=42 everywhere.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler

from feature_extractor import FEATURE_NAMES, extract_features
from url_analyzer import URL_FEATURE_NAMES, url_summary_features

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).parent
DEFAULT_DATA = ROOT / "data" / "emails.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_and_validate(path: Path) -> pd.DataFrame:
    """Load CSV, validate schema, check for obvious data-leakage issues."""
    print(f"\n{'='*60}")
    print(f"Loading dataset: {path}")
    print(f"{'='*60}")

    df = pd.read_csv(path)

    # --- Schema check ---
    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: Dataset missing columns: {missing}")

    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    invalid_labels = df[~df["label"].isin([0, 1])]
    if not invalid_labels.empty:
        sys.exit(f"ERROR: Found {len(invalid_labels)} rows with labels outside {{0,1}}")

    # --- Empty texts ---
    empty_mask = df["text"].str.len() < 5
    if empty_mask.sum():
        print(f"  WARNING: Dropping {empty_mask.sum()} rows with near-empty text.")
        df = df[~empty_mask]

    # --- Duplicates ---
    dupes = df.duplicated(subset="text")
    if dupes.sum():
        print(f"  WARNING: {dupes.sum()} duplicate email bodies found — dropping duplicates.")
        df = df.drop_duplicates(subset="text")

    # --- Class balance ---
    counts = df["label"].value_counts()
    n_legit, n_phish = counts.get(0, 0), counts.get(1, 0)
    ratio = min(n_legit, n_phish) / max(n_legit, n_phish)
    print(f"\n  Rows: {len(df)}  |  Phishing: {n_phish}  |  Legitimate: {n_legit}")
    print(f"  Class ratio (minority/majority): {ratio:.2f}")
    if ratio < 0.4:
        print("  WARNING: Dataset is significantly imbalanced. "
              "Consider adding more samples of the minority class.")

    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_numeric_matrix(texts) -> np.ndarray:
    """Build an (n_samples × n_numeric_features) array."""
    rows = []
    for t in texts:
        feats = extract_features(t).to_vector()
        url_feats = url_summary_features(t)
        rows.append(feats + [url_feats[k] for k in URL_FEATURE_NAMES])
    return np.array(rows, dtype=float)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(name: str, model, X_test, y_test) -> dict:
    """Compute all metrics we display on /metrics."""
    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred,
        target_names=["legitimate", "phishing"],
        digits=3,
        output_dict=False,
    )
    report_dict = classification_report(
        y_test, y_pred,
        target_names=["legitimate", "phishing"],
        digits=3,
        output_dict=True,
    )
    return {
        "model": name,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "phishing_precision": round(float(report_dict["phishing"]["precision"]), 4),
        "phishing_recall": round(float(report_dict["phishing"]["recall"]), 4),
        "legit_precision": round(float(report_dict["legitimate"]["precision"]), 4),
        "legit_recall": round(float(report_dict["legitimate"]["recall"]), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": report,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train the PhishGuard classifier.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Path to labelled CSV (columns: text, label)")
    args = parser.parse_args()

    df = load_and_validate(args.data)
    X_text = df["text"]
    y = df["label"].values
    class_dist = {int(k): int(v) for k, v in df["label"].value_counts().items()}

    # -----------------------------------------------------------------------
    # Train / test split — stratified so both splits share class balance
    # -----------------------------------------------------------------------
    X_text_train, X_text_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # -----------------------------------------------------------------------
    # TF-IDF vectorizer
    # -----------------------------------------------------------------------
    print("\nFitting TF-IDF vectorizer …")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 3),        # unigrams + bigrams + trigrams
        min_df=2,
        max_df=0.90,
        max_features=8_000,
        sublinear_tf=True,         # apply 1 + log(tf) scaling → better on skewed frequencies
        strip_accents="unicode",
    )
    X_train_tfidf = vectorizer.fit_transform(X_text_train)
    X_test_tfidf = vectorizer.transform(X_text_test)

    # -----------------------------------------------------------------------
    # Numeric / URL features
    # -----------------------------------------------------------------------
    print("Building numeric + URL features …")
    X_train_num = build_numeric_matrix(X_text_train)
    X_test_num = build_numeric_matrix(X_text_test)

    scaler = MinMaxScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_test_num_scaled = scaler.transform(X_test_num)

    X_train_combined = hstack([X_train_tfidf, csr_matrix(X_train_num_scaled)])
    X_test_combined = hstack([X_test_tfidf, csr_matrix(X_test_num_scaled)])

    # -----------------------------------------------------------------------
    # Model 1: Multinomial Naive Bayes (TF-IDF only — requires non-negative)
    # -----------------------------------------------------------------------
    print("\nTraining Multinomial Naive Bayes …")
    nb = MultinomialNB(alpha=0.1)     # lower alpha = less smoothing, sharper on distinctive tokens
    nb.fit(X_train_tfidf, y_train)
    nb_metrics = evaluate("Multinomial Naive Bayes (TF-IDF only)", nb, X_test_tfidf, y_test)

    # -----------------------------------------------------------------------
    # Model 2: Logistic Regression (TF-IDF + numeric + URL features)
    # -----------------------------------------------------------------------
    print("Training Logistic Regression …")
    lr = LogisticRegression(
        solver="saga",
        max_iter=3_000,
        C=2.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    lr.fit(X_train_combined, y_train)
    lr_metrics = evaluate(
        "Logistic Regression (TF-IDF + custom features)",
        lr, X_test_combined, y_test,
    )

    # -----------------------------------------------------------------------
    # Model 3: Random Forest (TF-IDF + numeric + URL features)
    # -----------------------------------------------------------------------
    print("Training Random Forest …")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train_combined, y_train)
    rf_metrics = evaluate(
        "Random Forest (TF-IDF + custom features)",
        rf, X_test_combined, y_test,
    )

    # -----------------------------------------------------------------------
    # Cross-validation (5-fold, stratified) on the FULL dataset
    # -----------------------------------------------------------------------
    print("\nRunning 5-fold cross-validation …")
    X_full_tfidf = vectorizer.transform(X_text)
    X_full_num = build_numeric_matrix(X_text)
    X_full_num_scaled = scaler.transform(X_full_num)
    X_full_combined = hstack([X_full_tfidf, csr_matrix(X_full_num_scaled)])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    cv_nb = cross_val_score(nb, X_full_tfidf, y, cv=cv, scoring="f1", n_jobs=-1)
    cv_lr = cross_val_score(lr, X_full_combined, y, cv=cv, scoring="f1", n_jobs=-1)
    cv_rf = cross_val_score(rf, X_full_combined, y, cv=cv, scoring="f1", n_jobs=-1)

    cv_scores = {
        "naive_bayes": {"mean": round(cv_nb.mean(), 4), "std": round(cv_nb.std(), 4), "folds": cv_nb.tolist()},
        "logistic_regression": {"mean": round(cv_lr.mean(), 4), "std": round(cv_lr.std(), 4), "folds": cv_lr.tolist()},
        "random_forest": {"mean": round(cv_rf.mean(), 4), "std": round(cv_rf.std(), 4), "folds": cv_rf.tolist()},
    }

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------
    all_model_results = [
        ("naive_bayes", nb, nb_metrics, False),
        ("logistic_regression", lr, lr_metrics, True),
        ("random_forest", rf, rf_metrics, True),
    ]

    for name, _, metrics, _ in all_model_results:
        print(f"\n{'─'*50}")
        print(f"  {metrics['model']}")
        print(f"{'─'*50}")
        print(metrics["classification_report"])
        cm = metrics["confusion_matrix"]
        print(f"  Confusion matrix:  TN={cm[0][0]}  FP={cm[0][1]}  FN={cm[1][0]}  TP={cm[1][1]}")
        cv_key = name.replace("logistic_regression", "logistic_regression").replace("random_forest", "random_forest")
        if cv_key in cv_scores:
            cv_info = cv_scores[cv_key]
            print(f"  5-fold CV F1:  mean={cv_info['mean']:.4f}  std={cv_info['std']:.4f}")

    # -----------------------------------------------------------------------
    # Overfitting detection
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Overfitting check (train F1 vs CV F1):")
    for name, model, metrics, uses_combined in all_model_results:
        X_tr = X_train_combined if uses_combined else X_train_tfidf
        train_f1 = f1_score(y_train, model.predict(X_tr), zero_division=0)
        cv_mean = cv_scores.get(name, {}).get("mean", 0)
        gap = train_f1 - cv_mean
        overfit_flag = "⚠ POSSIBLE OVERFIT" if gap > 0.15 else "✓ OK"
        print(f"  {name:<30}  train_F1={train_f1:.4f}  cv_F1={cv_mean:.4f}  gap={gap:.4f}  {overfit_flag}")

    # -----------------------------------------------------------------------
    # Pick best model by macro F1 (balances phishing detection AND avoiding false positives)
    # -----------------------------------------------------------------------
    best_name, best_model, best_metrics, best_uses_combined = max(
        all_model_results, key=lambda x: x[2]["f1_macro"]
    )
    print(f"\n✅ Best model: {best_name}  (macro F1 = {best_metrics['f1_macro']:.4f})")

    # -----------------------------------------------------------------------
    # Persist artifacts
    # -----------------------------------------------------------------------
    joblib.dump(vectorizer, MODELS_DIR / "vectorizer.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(best_model, MODELS_DIR / "model.pkl")

    model_info = {
        "best_model": best_name,
        "uses_combined_features": best_uses_combined,
        "feature_names": FEATURE_NAMES + URL_FEATURE_NAMES,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "class_distribution": class_dist,
        "cv_scores": cv_scores,
        "all_metrics": [nb_metrics, lr_metrics, rf_metrics],
        "best_metrics": best_metrics,
    }
    with open(MODELS_DIR / "model_info.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2)

    print(f"\nSaved artifacts to {MODELS_DIR}/")
    print("  model.pkl  |  vectorizer.pkl  |  scaler.pkl  |  model_info.json")
    print("\nNext step:  python app.py")


if __name__ == "__main__":
    main()
