# PhishGuard v2 — Complete Change Log

Every change made during the v1 → v2 rewrite, and the reason it was necessary.

---

## 1. Dataset (`data/emails.csv`)

| # | Change | Reason |
|---|--------|--------|
| 1.1 | Expanded from ~50 to 213 labelled examples | More data = better generalisation. Under 100 samples causes severe overfitting. |
| 1.2 | Added 85 phishing emails covering 12 attack types (credential harvest, prize scam, CEO fraud, delivery scam, tech support, romance, inheritance, account suspend, crypto, tax, debt, gov impersonation) | Original dataset only covered 2–3 types, causing the model to miss novel phishing patterns. |
| 1.3 | Added 128 diverse legitimate emails (work, personal, confirmations, newsletters, HR, banking notifications) | Low legitimate diversity caused false positives on any transactional email. |
| 1.4 | Balanced class ratio to ~60/40 legit/phish | Severe imbalance (e.g. 80/20) biases models toward predicting the majority class and inflates accuracy while masking poor recall. |

---

## 2. `feature_extractor.py`

| # | Change | Reason |
|---|--------|--------|
| 2.1 | Added 8 new features: `html_tag_count`, `obfuscated_word_count`, `spelling_trick_count`, `spoofing_signal_count`, `excessive_punctuation`, and expanded existing counts | More features = richer signal for the classifier; new features target attack patterns the original missed. |
| 2.2 | Fixed `_count_phrases` to use word-boundary regex (`\b`) for single-word phrases | Original used `str.count()` which triggers on substrings (e.g. "exported" would hit "port"). This caused false positives. |
| 2.3 | Added `_LEGIT_ACRONYMS` set; `all_caps_ratio` now excludes `URL`, `PIN`, `SSN`, `CEO`, etc. | Legitimate business emails routinely use these acronyms in caps. Counting them skewed the ratio and caused false positives on professional emails. |
| 2.4 | Expanded `URGENT_WORDS` list from 8 → 29 phrases | Real phishing uses many urgency variants not in the original short list. |
| 2.5 | Expanded `CREDENTIAL_WORDS` from 6 → 25 phrases | Original missed common credential-harvesting phrases like "confirm your details", "date of birth", "security question". |
| 2.6 | Expanded `MONEY_WORDS` from 5 → 30 phrases | Original missed inheritance scams, grant fraud, processing-fee scams. |
| 2.7 | Expanded `THREAT_WORDS` from 4 → 20 phrases | Original missed legal-action threats like "summons", "lien", "prosecution". |
| 2.8 | Added `SPELLING_TRICKS` regex list (paypa1, g00gle, etc.) | Character-substitution obfuscation is a top phishing evasion technique; original had no detection for it. |
| 2.9 | Added `SPOOFING_PHRASES` list ("this is paypal", "i am the ceo", etc.) | Sender impersonation is one of the most effective social engineering vectors; entirely absent from v1. |
| 2.10 | Added `list_red_flags()` severity scaling based on hit count | v1 returned flat flag lists. Scaling severity (low/medium/high) by count gives the UI more nuanced information. |
| 2.11 | Used `@dataclass` with `to_vector()` and `to_dict()` | v1 returned a plain dict with inconsistent key ordering. A dataclass guarantees stable feature vector order, preventing silent model mismatch bugs. |
| 2.12 | Added `FEATURE_NAMES` constant aligned with `to_vector()` output | Allows `train_model.py` and `app.py` to reference features by name without hardcoding indices. |

---

## 3. `url_analyzer.py`

| # | Change | Reason |
|---|--------|--------|
| 3.1 | Added 4 new URL signals: `_is_deceptive_path`, `_has_excessive_subdomains`, `_uses_http`, `_has_suspicious_port` | v1 only checked IP, shortener, TLD, and typosquat. Real phishing URLs frequently use deceptive paths (/verify, /login, /secure) and plain HTTP. |
| 3.2 | URL suspicion score raised from 0–4 to 0–8 | Reflects the 4 additional detection signals. |
| 3.3 | Expanded `URL_SHORTENERS` from 8 → 28 services | Phishers use many shorteners not in the original list (rebrand.ly, cutt.ly, clck.ru, etc.). |
| 3.4 | Expanded `SUSPICIOUS_TLDS` from 10 → 40 TLDs | Phishing campaigns heavily exploit free-registration TLDs not in the original list. |
| 3.5 | Expanded `BRANDS` from 10 → 50 impersonation targets | Original missed coinbase, zelle, venmo, docusign, adobe — all heavily targeted in real campaigns. |
| 3.6 | Added `DECEPTIVE_PATH_PATTERNS` list (16 path segments) | Paths like `/verify`, `/secure`, `/confirm` are strong phishing signals entirely missed by v1. |
| 3.7 | Fixed typosquat check to scan all non-TLD labels, not just the first | v1 missed `paypal-login.evil.com` because "paypal" was in the third label. |
| 3.8 | Added Levenshtein-distance typosquat check (edit-distance ≤ 1) | v1 only used substring matching, missing close misspellings like `gooogle.com`. |
| 3.9 | `url_summary_features()` now returns 9 keys (was 6), aligned with `URL_FEATURE_NAMES` | Adding more URL signals to the ML feature vector improves model discrimination. |
| 3.10 | Added URL deduplication in `extract_urls()` | v1 could count the same URL multiple times if it appeared twice, inflating scores. |
| 3.11 | Added `UrlAnalysis` dataclass with `to_dict()` | Consistent serialisation for the result template; v1 returned raw dicts with missing keys. |

---

## 4. `train_model.py`

| # | Change | Reason |
|---|--------|--------|
| 4.1 | Added Random Forest as a third candidate model | Ensemble methods frequently outperform single classifiers; gives users a fairer model comparison. |
| 4.2 | Added 5-fold stratified cross-validation for all three models | v1 had no cross-validation; train/test split alone cannot reliably detect overfitting on a small dataset. |
| 4.3 | Added overfitting detection: prints (train F1 − CV F1) gap with warning if > 0.15 | Makes it easy to spot if any model is memorising the training data. |
| 4.4 | Added dataset integrity checks: schema, null values, duplicates, label validity, class balance warning | v1 loaded the CSV with no validation; a malformed dataset would cause cryptic errors downstream. |
| 4.5 | TF-IDF `ngram_range` extended from `(1,2)` to `(1,3)` | Trigrams capture phishing-specific 3-word phrases ("verify your account", "click here immediately") more precisely than bigrams alone. |
| 4.6 | TF-IDF `max_features` raised from 5 000 → 8 000 | Larger vocabulary reduces feature-loss on a diverse dataset. |
| 4.7 | Added `sublinear_tf=True` to TF-IDF | Applies 1 + log(tf) scaling; improves discrimination on high-frequency phishing keywords without letting them dominate. |
| 4.8 | Logistic Regression solver changed from `lbfgs` → `saga` | `saga` is faster on high-dimensional sparse matrices (TF-IDF output) and supports L1 regularisation. |
| 4.9 | Added `class_weight="balanced"` to LR and RF | Compensates for class imbalance automatically; reduces false negatives on the minority class. |
| 4.10 | Best model selected by macro F1 instead of accuracy | Accuracy is misleading on imbalanced data. Macro F1 equally weights both classes, penalising poor performance on phishing AND legitimate. |
| 4.11 | `model_info.json` now stores cross-val scores, per-class metrics, and confusion matrices | The `/metrics` page previously had no cross-val data and no per-class breakdown. |
| 4.12 | Added `argparse` for `--data` path override | v1 hardcoded the CSV path, making it impossible to train on a different dataset without editing the source. |
| 4.13 | `random_state=42` set consistently everywhere | Reproducibility: results are identical across runs, making debugging and comparisons valid. |
| 4.14 | Scaler changed from `StandardScaler` → `MinMaxScaler` for numeric features | `StandardScaler` can produce negative values incompatible with `MultinomialNB`. `MinMaxScaler` constrains output to [0, 1] for all models. |

---

## 5. `app.py`

| # | Change | Reason |
|---|--------|--------|
| 5.1 | Added `MAX_EMAIL_CHARS = 50 000` hard cap and `MIN_EMAIL_CHARS = 10` | v1 had no input length limit; a very large input could cause a DoS via slow TF-IDF transform. Short inputs would crash `predict_proba`. |
| 5.2 | Added `_sanitize_input()` — strips null bytes, normalises line endings | Null bytes crash some Python string operations; normalised line endings ensure consistent tokenisation. |
| 5.3 | Prediction wrapped in `try/except` with user-friendly error message | v1 returned a 500 page on any prediction error. |
| 5.4 | Model artifacts validated on load with `_validate_model()` | Catches a corrupt or mismatched pickle file at startup, not at first request. |
| 5.5 | `SECRET_KEY` loaded from environment variable | v1 had a hardcoded or missing secret key. Hardcoded keys are a security vulnerability. |
| 5.6 | `debug=True` replaced with `DEBUG_MODE` flag driven by `FLASK_DEBUG` env var | Running Flask with `debug=True` in production enables the interactive debugger, a critical security vulnerability. |
| 5.7 | Added `/health` endpoint returning JSON | Required for PythonAnywhere uptime monitoring and keep-alive pings. |
| 5.8 | Added `/about` route | Portfolio/demo use: explains the project to evaluators without reading source code. |
| 5.9 | Confidence thresholds adjusted: 0.75 / 0.50 / 0.20 | v1 thresholds were not documented or calibrated. New thresholds match the observed probability distributions. |
| 5.10 | Added `combined_risk` score (weighted blend of ML prob, URL score, flag count) | Gives the UI a single 0–100 display number that integrates all detection layers. |
| 5.11 | `SAMPLE_EMAILS` list moved into `app.py` with richer examples | v1 had no sample emails; demos required manually pasting text. |
| 5.12 | User input never passed through `| safe` filter in templates | Jinja2 auto-escaping is enabled by default; this is preserved. v1 had one instance of `| safe` on user-reflected content, which is an XSS vulnerability. |

---

## 6. Templates & CSS

| # | Change | Reason |
|---|--------|--------|
| 6.1 | Full dark-theme redesign with CSS custom properties | v1 had a basic light stylesheet with no design system. Custom properties make theming and maintenance easier. |
| 6.2 | Added `base.html` sticky header with navigation | v1 had no persistent navigation between pages. |
| 6.3 | Animated confidence bar (CSS + JS) | Provides visual feedback; makes the confidence percentage easier to grasp at a glance. |
| 6.4 | Red flags now grouped with severity badges (high/medium/low) | v1 showed a flat list with no severity distinction, making all flags look equally important. |
| 6.5 | URL analysis now shows per-URL score pill and tag chips | v1 showed a plain text list. Chip-based tags are faster to scan and highlight the most dangerous signals. |
| 6.6 | Feature grid uses colour-coded alert states | Highlights which specific features are elevated, directing attention efficiently. |
| 6.7 | Added combined risk score bar and risk level badge | Single at-a-glance risk indicator requested in the requirements. |
| 6.8 | Metrics page shows 5-fold CV scores with mini bar charts | Demonstrates model reliability beyond a single train/test split — important for college evaluation. |
| 6.9 | Metrics page shows per-class precision/recall (phishing vs legitimate) | False-positive rate (legitimate → phishing) and false-negative rate (phishing → legitimate) are key metrics; v1 only showed overall accuracy. |
| 6.10 | Added sample email grid on the index page with category badges | Makes demos frictionless — one click to load any example. |
| 6.11 | Added character counter on the textarea | Prevents user confusion when hitting the 50 000-char limit. |
| 6.12 | Full responsive layout for mobile / tablet | v1 had no responsive CSS; the layout broke below ~900px. |
| 6.13 | Added `about.html` page | Explains the tech stack, ML pipeline, limitations, and future work — essential for a portfolio/evaluation project. |
| 6.14 | Textarea uses `spellcheck="false"` and monospace font | Spell-check on email content adds UI noise; monospace makes raw email text readable. |

---

## 7. Project structure & deployment

| # | Change | Reason |
|---|--------|--------|
| 7.1 | Added `models/` directory for all trained artifacts | v1 saved `.pkl` files to the project root, cluttering it and making `.gitignore` harder. |
| 7.2 | Added `tests/test_core.py` with 40 unit tests | v1 had zero tests. Tests catch regressions when the code is modified and demonstrate engineering rigour to evaluators. |
| 7.3 | `requirements.txt` updated with version ranges | v1 had no version pins; `pip install` could silently install incompatible versions. |
| 7.4 | `wsgi.py` updated with clear setup instructions and `dotenv` support | v1 wsgi.py had a hardcoded path and no documentation. |
| 7.5 | Added `DEPLOY.md` with full step-by-step PythonAnywhere guide | v1 had a sparse deployment doc missing the static-files mapping and environment variable setup. |
| 7.6 | Added `.gitignore` | v1 would commit model `.pkl` files and `__pycache__` to version control. |
| 7.7 | Added `/health` endpoint and keep-alive instructions in `DEPLOY.md` | Free PythonAnywhere accounts sleep after inactivity; the health endpoint allows cron-based keep-alive. |
