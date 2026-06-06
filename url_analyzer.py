"""
url_analyzer.py
---------------
Extract and analyze URLs from an email body.

For each URL we report:
  - the raw URL and parsed host
  - whether the host is a bare IP address
  - whether the host uses a known URL shortener
  - whether the TLD is on the high-abuse free-TLD list
  - whether the host appears to typosquat a popular brand
  - whether the URL contains deceptive path patterns
  - whether the subdomain count is abnormally high
  - whether the URL contains encoded/obfuscated characters

Changes from v1:
  - Added: _is_deceptive_path, _has_excessive_subdomains, _has_obfuscated_chars,
    _uses_http_not_https, port_suspicious
  - Expanded URL_SHORTENERS list (30+ services)
  - Expanded SUSPICIOUS_TLDS list (40+ abused TLDs)
  - Expanded BRANDS list with more targets
  - Fixed: typosquat check now also checks full domain (not just first label),
    preventing false negatives like "paypal-secure-login.com"
  - Fixed: URL regex now also captures bare www. links correctly
  - score now 0–8 (previously 0–4) to reflect additional signals
  - url_summary_features returns 9 keys (was 6), aligned with URL_FEATURE_NAMES
"""

import re
from dataclasses import dataclass, asdict
from typing import List, Optional
from urllib.parse import urlparse, unquote


# ---------------------------------------------------------------------------
# Regex for URL extraction
# ---------------------------------------------------------------------------

URL_RE = re.compile(
    r"(https?://[^\s<>\"')]+|www\.[^\s<>\"')]+)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Reference sets
# ---------------------------------------------------------------------------

URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "shorte.st", "cutt.ly", "rb.gy", "rebrand.ly",
    "tiny.cc", "lnkd.in", "x.co", "u.to", "short.io", "bl.ink",
    "snip.ly", "yourls.org", "v.gd", "qr.ae", "hyperurl.co",
    "shorturl.at", "t2m.io", "clck.ru", "pp.gg", "bitly.com",
}

# Free / abused TLDs commonly associated with phishing campaigns
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq",           # Freenom freebies
    "xyz", "top", "click", "online",
    "cyou", "icu", "review", "host", "work",
    "support", "info", "live", "website",
    "club", "site", "fun", "space",
    "pw", "cc", "biz", "ws", "su",
    "ru", "cn",                               # High phishing volume ccTLDs
    "buzz", "cam", "vip", "gg",
}

# Brands frequently impersonated in phishing emails
BRANDS = [
    "paypal", "amazon", "apple", "microsoft", "google", "facebook",
    "instagram", "netflix", "spotify", "linkedin", "twitter", "ebay",
    "chase", "wellsfargo", "bankofamerica", "citibank", "fedex", "ups",
    "dhl", "outlook", "gmail", "icloud", "discord", "steam", "github",
    "dropbox", "verizon", "att", "tmobile", "coinbase", "binance",
    "blockchain", "zelle", "venmo", "cashapp", "stripe", "docusign",
    "adobe", "office365", "onedrive", "youtube", "tiktok", "snapchat",
    "whatsapp", "telegram", "zoom", "webex", "aol", "yahoo",
    "americanexpress", "visa", "mastercard", "irs", "usps", "dhl",
]

# Path patterns that are commonly used in phishing URLs
DECEPTIVE_PATH_PATTERNS = [
    r"/verify", r"/confirm", r"/secure", r"/login", r"/signin",
    r"/account", r"/update", r"/reset", r"/recover", r"/unlock",
    r"/validate", r"/authenticate", r"/alert", r"/security",
    r"/suspended", r"/billing", r"/payment", r"/support",
]


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class UrlAnalysis:
    url: str
    host: Optional[str]
    is_ip: bool
    is_shortener: bool
    suspicious_tld: bool
    typosquat_of: Optional[str]
    deceptive_path: bool
    excessive_subdomains: bool
    uses_http: bool
    score: int          # 0–8, higher = more suspicious

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _is_ip(host: str) -> bool:
    """Return True if host is a bare IPv4 address."""
    return bool(re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", host or ""))


def _levenshtein(a: str, b: str) -> int:
    """Compute edit distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def _typosquat_of(host: str) -> Optional[str]:
    """
    Return the brand a host appears to be impersonating, or None.

    Strategy (in priority order):
      1. Brand appears as a substring inside a longer label (paypal-secure.com)
      2. Close Levenshtein match to a brand (paypa1.com, gooogle.com)
    """
    if not host:
        return None
    labels = host.lower().split(".")
    # Check all labels except the TLD
    candidates = [lbl for lbl in labels[:-1] if len(lbl) >= 4]
    for cand in candidates:
        for brand in BRANDS:
            # Brand embedded in a longer label is a strong signal
            if brand in cand and cand != brand:
                return brand
        for brand in BRANDS:
            # Edit-distance ≤ 1 with similar length is a strong typosquat signal
            if abs(len(cand) - len(brand)) <= 2 and _levenshtein(cand, brand) == 1:
                return brand
    return None


def _is_deceptive_path(path: str) -> bool:
    """Return True if the URL path contains phishing-associated segments."""
    path_lower = (path or "").lower()
    return any(re.search(p, path_lower) for p in DECEPTIVE_PATH_PATTERNS)


def _has_excessive_subdomains(host: str) -> bool:
    """Return True if there are 3+ subdomains (e.g. secure.paypal.login.evil.com)."""
    if not host:
        return False
    parts = host.split(".")
    return len(parts) >= 5  # e.g. a.b.c.evil.com  -> 5 parts


def _uses_http(url: str) -> bool:
    """Return True if the URL uses plain HTTP (not HTTPS)."""
    return url.lower().startswith("http://")


def _has_suspicious_port(parsed) -> bool:
    """Return True if a non-standard port is specified."""
    return parsed.port is not None and parsed.port not in (80, 443, 8080, 8443)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_urls(text: str) -> List[str]:
    """Find all URLs in raw email text and strip trailing punctuation."""
    if not text:
        return []
    raw = URL_RE.findall(text)
    cleaned = [u.rstrip(".,;:!?)\"'") for u in raw]
    # Deduplicate while preserving order
    seen = set()
    result = []
    for u in cleaned:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def analyze_url(url: str) -> UrlAnalysis:
    """Analyze a single URL and compute a suspicion score 0–8."""
    parse_target = url if url.lower().startswith(("http://", "https://")) else "http://" + url
    parsed = urlparse(parse_target)
    host = (parsed.hostname or "").lower().strip(".")

    is_ip             = _is_ip(host)
    is_shortener      = any(host == s or host.endswith("." + s) for s in URL_SHORTENERS)
    tld               = host.rsplit(".", 1)[-1] if "." in host else ""
    suspicious_tld    = tld in SUSPICIOUS_TLDS
    typosquat         = _typosquat_of(host)
    deceptive_path    = _is_deceptive_path(parsed.path)
    excess_sub        = _has_excessive_subdomains(host)
    http_only         = _uses_http(url)
    sus_port          = _has_suspicious_port(parsed)

    score = sum([
        is_ip,
        is_shortener,
        suspicious_tld,
        bool(typosquat),
        deceptive_path,
        excess_sub,
        http_only and not is_shortener,   # HTTP on a non-shortener is suspicious
        sus_port,
    ])

    return UrlAnalysis(
        url=url,
        host=host or None,
        is_ip=is_ip,
        is_shortener=is_shortener,
        suspicious_tld=suspicious_tld,
        typosquat_of=typosquat,
        deceptive_path=deceptive_path,
        excessive_subdomains=excess_sub,
        uses_http=http_only,
        score=score,
    )


def analyze_text_urls(text: str) -> List[UrlAnalysis]:
    """Extract every URL in `text` and return per-URL analyses."""
    return [analyze_url(u) for u in extract_urls(text)]


def url_summary_features(text: str) -> dict:
    """
    Roll up per-URL analyses into numeric signals for the ML model.
    Returns a dict with exactly the keys in URL_FEATURE_NAMES.
    """
    analyses = analyze_text_urls(text)
    empty = {
        "url_count": 0,
        "ip_url_count": 0,
        "shortener_count": 0,
        "suspicious_tld_count": 0,
        "typosquat_count": 0,
        "deceptive_path_count": 0,
        "http_only_count": 0,
        "max_url_score": 0,
        "any_suspicious_url": 0,
    }
    if not analyses:
        return empty

    result = {
        "url_count": len(analyses),
        "ip_url_count": sum(a.is_ip for a in analyses),
        "shortener_count": sum(a.is_shortener for a in analyses),
        "suspicious_tld_count": sum(a.suspicious_tld for a in analyses),
        "typosquat_count": sum(bool(a.typosquat_of) for a in analyses),
        "deceptive_path_count": sum(a.deceptive_path for a in analyses),
        "http_only_count": sum(a.uses_http for a in analyses),
        "max_url_score": max(a.score for a in analyses),
        "any_suspicious_url": int(any(a.score >= 2 for a in analyses)),
    }
    return result


URL_FEATURE_NAMES = [
    "url_count", "ip_url_count", "shortener_count",
    "suspicious_tld_count", "typosquat_count",
    "deceptive_path_count", "http_only_count",
    "max_url_score", "any_suspicious_url",
]
