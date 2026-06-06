"""
tests/test_core.py
------------------
Unit tests for feature_extractor, url_analyzer, and the Flask app.

Run:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_extractor import extract_features, list_red_flags
from url_analyzer import (
    analyze_url,
    analyze_text_urls,
    extract_urls,
    url_summary_features,
)


# ─────────────────────────────────────────────
#  feature_extractor tests
# ─────────────────────────────────────────────

class TestExtractFeatures:
    def test_empty_string(self):
        f = extract_features("")
        assert f.word_count == 0
        assert f.urgent_word_count == 0

    def test_whitespace_only(self):
        f = extract_features("   \n\t  ")
        assert f.char_length == 0  # stripped in extractor

    def test_urgent_phishing(self):
        text = "URGENT! Act now or your account will be suspended immediately!"
        f = extract_features(text)
        assert f.urgent_word_count >= 1
        assert f.threat_word_count >= 1
        assert f.exclamation_count >= 2

    def test_credential_request(self):
        text = "Please verify your password and credit card number to continue."
        f = extract_features(text)
        assert f.credential_word_count >= 2

    def test_money_bait(self):
        text = "Congratulations! You have won the lottery! Claim your prize now!"
        f = extract_features(text)
        assert f.money_word_count >= 2

    def test_generic_greeting(self):
        text = "Dear customer, your account has been suspended."
        f = extract_features(text)
        assert f.generic_greeting_count >= 1

    def test_legitimate_no_flags(self):
        text = "Hey, can we move the 3pm meeting to Thursday? Let me know if that works."
        f = extract_features(text)
        assert f.urgent_word_count == 0
        assert f.credential_word_count == 0
        assert f.threat_word_count == 0

    def test_all_caps_ratio(self):
        text = "THIS IS VERY IMPORTANT YOU MUST ACT NOW"
        f = extract_features(text)
        assert f.all_caps_ratio > 0.5
        assert f.all_caps_word_count >= 5

    def test_legit_acronyms_not_counted(self):
        text = "Please check the URL and enter your PIN at the ATM."
        f = extract_features(text)
        assert f.all_caps_word_count == 0  # URL, PIN, ATM are in the exclusion list

    def test_dollar_amount_detected(self):
        text = "Your account has been charged $499.99 without authorisation."
        f = extract_features(text)
        assert f.has_dollar_amount == 1

    def test_spelling_tricks(self):
        text = "Login to your paypa1 account here immediately."
        f = extract_features(text)
        assert f.spelling_trick_count >= 1

    def test_spoofing_signals(self):
        text = "This is PayPal security team. Your account needs verification."
        f = extract_features(text)
        assert f.spoofing_signal_count >= 1

    def test_to_vector_length(self):
        f = extract_features("Sample email text for testing.")
        assert len(f.to_vector()) == 18

    def test_to_dict_has_all_keys(self):
        from feature_extractor import FEATURE_NAMES
        f = extract_features("Test email")
        d = f.to_dict()
        for key in FEATURE_NAMES:
            assert key in d, f"Missing key: {key}"


class TestListRedFlags:
    def test_returns_list(self):
        flags = list_red_flags("Hello world")
        assert isinstance(flags, list)

    def test_phishing_email_has_flags(self):
        text = (
            "URGENT! Dear customer, verify your password NOW or your account "
            "will be SUSPENDED! Click here: http://paypa1.tk/verify"
        )
        flags = list_red_flags(text)
        assert len(flags) >= 3

    def test_legit_email_low_flags(self):
        text = "Hi, just confirming the meeting tomorrow at 3pm. See you then!"
        flags = list_red_flags(text)
        assert len(flags) <= 1

    def test_flag_has_required_keys(self):
        text = "URGENT! Verify your password immediately or face suspension!"
        flags = list_red_flags(text)
        for flag in flags:
            assert "label" in flag
            assert "severity" in flag
            assert "detail" in flag

    def test_severity_values_valid(self):
        text = "Dear customer, you have won a million dollars! Act now! Password required!"
        flags = list_red_flags(text)
        valid_severities = {"low", "medium", "high"}
        for flag in flags:
            assert flag["severity"] in valid_severities


# ─────────────────────────────────────────────
#  url_analyzer tests
# ─────────────────────────────────────────────

class TestExtractUrls:
    def test_no_urls(self):
        assert extract_urls("No links in this email at all.") == []

    def test_https_url(self):
        urls = extract_urls("Visit https://example.com for more info.")
        assert "https://example.com" in urls

    def test_http_url(self):
        urls = extract_urls("Click http://evil.tk/login to verify.")
        assert "http://evil.tk/login" in urls

    def test_multiple_urls(self):
        text = "See https://google.com and https://amazon.com"
        urls = extract_urls(text)
        assert len(urls) == 2

    def test_strips_trailing_punctuation(self):
        urls = extract_urls("Visit https://example.com.")
        assert "https://example.com" in urls

    def test_deduplication(self):
        text = "https://evil.tk/login and https://evil.tk/login again"
        urls = extract_urls(text)
        assert urls.count("https://evil.tk/login") == 1


class TestAnalyzeUrl:
    def test_ip_url(self):
        result = analyze_url("http://192.168.1.1/login")
        assert result.is_ip is True
        assert result.score >= 1

    def test_shortener(self):
        result = analyze_url("https://bit.ly/abc123")
        assert result.is_shortener is True

    def test_suspicious_tld_tk(self):
        result = analyze_url("http://paypal-verify.tk/login")
        assert result.suspicious_tld is True

    def test_suspicious_tld_xyz(self):
        result = analyze_url("https://amazon-prize.xyz/claim")
        assert result.suspicious_tld is True

    def test_typosquat_detection(self):
        result = analyze_url("http://paypa1-secure.com/verify")
        # typosquat OR deceptive path should be flagged
        assert result.typosquat_of is not None or result.deceptive_path

    def test_brand_in_subdomain(self):
        result = analyze_url("http://paypal-login.evil.com/account")
        assert result.typosquat_of == "paypal"

    def test_deceptive_path(self):
        result = analyze_url("https://secure-bank.com/verify/account")
        assert result.deceptive_path is True

    def test_excessive_subdomains(self):
        result = analyze_url("https://a.b.c.evil.com/login")
        assert result.excessive_subdomains is True

    def test_http_flagged(self):
        result = analyze_url("http://some-normal-site.com")
        assert result.uses_http is True

    def test_https_not_flagged_as_http(self):
        result = analyze_url("https://google.com")
        assert result.uses_http is False

    def test_clean_url_score_zero(self):
        result = analyze_url("https://google.com")
        assert result.score == 0

    def test_phishing_url_high_score(self):
        result = analyze_url("http://paypal-login.tk/verify?id=123")
        assert result.score >= 3

    def test_host_parsed(self):
        result = analyze_url("https://www.example.com/page")
        assert result.host == "www.example.com"


class TestUrlSummaryFeatures:
    def test_no_urls(self):
        feats = url_summary_features("No links here.")
        assert feats["url_count"] == 0
        assert feats["max_url_score"] == 0
        assert feats["any_suspicious_url"] == 0

    def test_phishing_text(self):
        text = "Click here: http://paypal-verify.tk/login to confirm your account."
        feats = url_summary_features(text)
        assert feats["url_count"] == 1
        assert feats["suspicious_tld_count"] >= 1
        assert feats["deceptive_path_count"] >= 1
        assert feats["any_suspicious_url"] == 1

    def test_summary_has_all_keys(self):
        from url_analyzer import URL_FEATURE_NAMES
        feats = url_summary_features("Test https://example.com")
        for key in URL_FEATURE_NAMES:
            assert key in feats, f"Missing key: {key}"

    def test_shortener_detected(self):
        text = "Click here: https://bit.ly/xyz"
        feats = url_summary_features(text)
        assert feats["shortener_count"] == 1

    def test_ip_url_detected(self):
        text = "Login at http://192.168.0.1/login"
        feats = url_summary_features(text)
        assert feats["ip_url_count"] == 1


# ─────────────────────────────────────────────
#  Flask app tests
# ─────────────────────────────────────────────

class TestFlaskApp:
    @pytest.fixture
    def client(self):
        import app as app_module
        app_module.app.config["TESTING"] = True
        app_module.app.config["WTF_CSRF_ENABLED"] = False
        with app_module.app.test_client() as c:
            yield c

    def test_index_loads(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"PhishGuard" in resp.data

    def test_metrics_loads(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_about_loads(self, client):
        resp = client.get("/about")
        assert resp.status_code == 200

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "status" in data
        assert data["status"] == "ok"

    def test_empty_submit_shows_error(self, client):
        resp = client.post("/predict", data={"email_text": ""})
        assert resp.status_code == 200
        assert b"Please paste" in resp.data or b"too short" in resp.data or b"error" in resp.data.lower()

    def test_too_short_input(self, client):
        resp = client.post("/predict", data={"email_text": "hi"})
        assert resp.status_code == 200

    def test_too_long_input(self, client):
        resp = client.post("/predict", data={"email_text": "a" * 60_000})
        assert resp.status_code == 200
        assert b"too long" in resp.data

    def test_predict_phishing(self, client):
        phishing = (
            "URGENT! Dear customer, verify your PayPal password NOW at "
            "http://paypa1-verify.tk/login or your account will be suspended!"
        )
        resp = client.post("/predict", data={"email_text": phishing})
        assert resp.status_code == 200
        assert b"result" in resp.data or b"verdict" in resp.data or b"Phishing" in resp.data

    def test_predict_legitimate(self, client):
        legit = (
            "Hey, just a reminder about the team lunch on Friday at noon. "
            "Let me know if you can make it. See you then!"
        )
        resp = client.post("/predict", data={"email_text": legit})
        assert resp.status_code == 200

    def test_null_byte_stripped(self, client):
        text = "Hello\x00 this is a test email with a null byte inside."
        resp = client.post("/predict", data={"email_text": text})
        # Should not crash — returns 200
        assert resp.status_code == 200

    def test_sample_prefill(self, client):
        resp = client.get("/?sample=0")
        assert resp.status_code == 200

    def test_xss_not_reflected(self, client):
        xss = '<script>alert("xss")</script> This is a test email body for XSS check.'
        resp = client.post("/predict", data={"email_text": xss})
        assert b'<script>alert("xss")</script>' not in resp.data
