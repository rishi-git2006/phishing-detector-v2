# PhishGuard — ML-Powered Phishing Email Detector

A machine learning system that detects phishing emails using 
TF-IDF + Logistic Regression, 18 linguistic features, and 9 URL signals.

## Tech Stack
Python · Flask · scikit-learn · TF-IDF · Logistic Regression · Naive Bayes · Random Forest

## Results
- **97.7% accuracy** on held-out test set
- **Macro F1: 0.975**
- Zero false positives on legitimate emails in test set
- 5-fold cross-validation confirms no overfitting

## Run locally
pip install -r requirements.txt
python train_model.py
python app.py

