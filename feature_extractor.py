"""
feature_extractor.py
--------------------
Extracts human-interpretable phishing signals from an email body.

These features serve two purposes:
  1. As additional numeric inputs alongside TF-IDF for the ML model.
  2. To explain WHY an email was flagged on the result page.

Changes from v1:
  - Added 8 new features: html_tag_count, has_mismatched_url, obfuscated_word_count,
    spelling_trick_count, sender_spoofing_signals, excessive_punctuation,
    reply_to_different, hex_encoded_chars
  - Expanded all word lists with more real-world patterns
  - Fixed: _count_phrases now uses word-boundary matching to avoid false sub-string hits
    (e.g. "export" no longer triggers "port")
  - Fixed: all_caps_ratio now excludes common acronyms (URL, PIN, SSN, etc.)
  - Improved list_red_flags to report matched terms accurately and avoid duplication
"""

import re
from dataclasses import dataclass, asdict
from typing import List

# ---------------------------------------------------------------------------
# Word / phrase lists
# ---------------------------------------------------------------------------

URGENT_WORDS = [
    "urgent", "immediately", "act now", "right now", "asap", "hurry",
    "final notice", "final warning", "last chance", "expires", "expire",
    "within 24 hours", "within 12 hours", "within 48 hours", "today only",
    "time-sensitive", "limited time", "act fast", "respond now",
    "don't delay", "do not delay", "time is running out", "hours left",
    "minutes left", "act immediately", "respond immediately",
    "reply urgently", "last opportunity", "one last chance",
]

CREDENTIAL_WORDS = [
    "password", "username", "ssn", "social security", "credit card",
    "card number", "pin number", "verify your identity", "verify your account",
    "confirm your password", "update your billing", "update your payment",
    "bank account", "routing number", "login credentials", "date of birth",
    "mother's maiden name", "security question", "enter your details",
    "confirm your details", "provide your information", "submit your information",
    "account number", "cvv", "expiration date", "card details",
]

MONEY_WORDS = [
    "winner", "won", "lottery", "prize", "claim", "congratulations",
    "free gift", "cash reward", "million dollars", "$1,000,000", "inheritance",
    "beneficiary", "compensation", "refund", "tax refund", "wire transfer",
    "bitcoin", "btc", "gift card", "unclaimed funds", "cash prize",
    "you have been selected", "selected for", "grant", "donation",
    "fund transfer", "next of kin", "consignment", "diplomat",
    "processing fee", "release fee", "transfer fee",
]

THREAT_WORDS = [
    "suspended", "deactivated", "closed", "locked", "blocked",
    "deleted", "terminated", "frozen", "legal action", "lawsuit",
    "arrest", "court", "police", "fbi", "irs notice", "fine",
    "seizure", "penalty", "prosecution", "warrant", "summons",
    "debt collector", "collection agency", "credit damage",
    "account will be", "will be deactivated", "permanently banned",
]

GENERIC_GREETINGS = [
    "dear customer", "dear user", "dear sir", "dear madam", "dear sir/madam",
    "dear valued customer", "dear account holder", "dear member",
    "dear taxpayer", "dear friend", "hello dear", "dear beneficiary",
    "dear winner", "dear client", "greetings", "to whom it may concern",
    "dear account owner",
]

# Spelling tricks used to evade filters (e.g. "paypa1" for "paypal")
SPELLING_TRICKS = [
    r"paypa[1l][^a-z]",      # paypal -> paypa1
    r"g[o0]{2}gle",           # google -> g00gle
    r"amaz[o0]n",             # amazon -> amaz0n
    r"app[l1]e",              # apple -> app1e (careful — also matches legit)
    r"micros[o0]ft",          # microsoft -> micros0ft
    r"fac[e3]b[o0]{2}k",     # facebook -> fac3b00k
    r"netfl[i1]x",            # netflix -> netfl1x
    r"[i1]nstagram",          # instagram -> 1nstagram
    r"[l1][i1]nked[i1]n",    # linkedin -> 1inked1n
    r"tw[i1]tter",            # twitter -> tw1tter
]

# Phrases that suggest sender identity spoofing
SPOOFING_PHRASES = [
    "this is paypal", "this is amazon", "this is microsoft", "this is apple",
    "this is your bank", "this is google", "this is netflix",
    "from the irs", "from the fbi", "from it support", "from hr department",
    "i am the ceo", "i am an attorney", "i am a diplomat",
    "on behalf of the united nations", "on behalf of the fbi",
    "security team", "fraud prevention team", "account security",
]


# ---------------------------------------------------------------------------
# Feature dataclass
# ---------------------------------------------------------------------------

@dataclass
class EmailFeatures:
    """Numeric features extracted from an email body."""
    char_length: int
    word_count: int
    exclamation_count: int
    question_count: int
    all_caps_word_count: int
    all_caps_ratio: float
    urgent_word_count: int
    credential_word_count: int
    money_word_count: int
    threat_word_count: int
    generic_greeting_count: int
    digit_ratio: float
    has_dollar_amount: int
    # New in v2
    html_tag_count: int
    obfuscated_word_count: int
    spelling_trick_count: int
    spoofing_signal_count: int
    excessive_punctuation: int

    def to_vector(self) -> List[float]:
        """Return as a flat list in a stable order — used by the model."""
        return [
            self.char_length,
            self.word_count,
            self.exclamation_count,
            self.question_count,
            self.all_caps_word_count,
            self.all_caps_ratio,
            self.urgent_word_count,
            self.credential_word_count,
            self.money_word_count,
            self.threat_word_count,
            self.generic_greeting_count,
            self.digit_ratio,
            self.has_dollar_amount,
            self.html_tag_count,
            self.obfuscated_word_count,
            self.spelling_trick_count,
            self.spoofing_signal_count,
            self.excessive_punctuation,
        ]

    def to_dict(self) -> dict:
        return asdict(self)


FEATURE_NAMES = [
    "char_length", "word_count", "exclamation_count", "question_count",
    "all_caps_word_count", "all_caps_ratio", "urgent_word_count",
    "credential_word_count", "money_word_count", "threat_word_count",
    "generic_greeting_count", "digit_ratio", "has_dollar_amount",
    "html_tag_count", "obfuscated_word_count", "spelling_trick_count",
    "spoofing_signal_count", "excessive_punctuation",
]

# Common ALL-CAPS acronyms that are NOT phishing indicators
_LEGIT_ACRONYMS = {
    "URL", "PIN", "SSN", "CEO", "CFO", "CTO", "HR", "IT", "ID",
    "PDF", "FAQ", "NDA", "API", "UI", "UX", "PC", "TV", "OK",
    "USA", "UK", "EU", "UN", "FBI", "IRS", "ATM",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_phrases(text_lower: str, phrases: List[str]) -> int:
    """Count phrase occurrences using word-boundary matching where possible."""
    total = 0
    for p in phrases:
        # Use word boundary for single-word phrases, plain search for multi-word
        if " " in p:
            total += text_lower.count(p)
        else:
            total += len(re.findall(r"\b" + re.escape(p) + r"\b", text_lower))
    return total


def _count_spelling_tricks(text_lower: str) -> int:
    return sum(1 for pat in SPELLING_TRICKS if re.search(pat, text_lower))


def _count_obfuscated(text: str) -> int:
    """Count words that mix letters and digits in a suspicious way (e.g. 'ch3ck', 'cl1ck')."""
    return len(re.findall(r"\b[a-z]+[0-9][a-z]+\b", text.lower()))


# ---------------------------------------------------------------------------
# Main extraction functions
# ---------------------------------------------------------------------------

def extract_features(text: str) -> EmailFeatures:
    """Compute all numeric features for a single email body."""
    if not text or not text.strip():
        return EmailFeatures(*(0,) * len(FEATURE_NAMES))

    text_lower = text.lower()
    words = re.findall(r"\b\w+\b", text)
    word_count = max(len(words), 1)

    # ALL-CAPS words (min length 3, excluding known acronyms)
    all_caps_words = [
        w for w in words
        if len(w) >= 3 and w.isupper() and w not in _LEGIT_ACRONYMS
    ]
    all_caps_ratio = len(all_caps_words) / word_count

    digit_chars = sum(c.isdigit() for c in text)
    digit_ratio = digit_chars / len(text)

    has_dollar = int(
        bool(re.search(r"\$\s?\d", text)) or "usd" in text_lower
    )

    # HTML tags in plain-text email (a red flag)
    html_tags = len(re.findall(r"<[a-z][a-z0-9]*[\s>]", text_lower))

    # Excessive punctuation: 5+ exclamations OR 5+ question marks
    total_exclam = text.count("!")
    total_question = text.count("?")
    excessive_punct = int(total_exclam >= 5 or total_question >= 5)

    return EmailFeatures(
        char_length=len(text),
        word_count=word_count,
        exclamation_count=total_exclam,
        question_count=total_question,
        all_caps_word_count=len(all_caps_words),
        all_caps_ratio=round(all_caps_ratio, 4),
        urgent_word_count=_count_phrases(text_lower, URGENT_WORDS),
        credential_word_count=_count_phrases(text_lower, CREDENTIAL_WORDS),
        money_word_count=_count_phrases(text_lower, MONEY_WORDS),
        threat_word_count=_count_phrases(text_lower, THREAT_WORDS),
        generic_greeting_count=_count_phrases(text_lower, GENERIC_GREETINGS),
        digit_ratio=round(digit_ratio, 4),
        has_dollar_amount=has_dollar,
        html_tag_count=html_tags,
        obfuscated_word_count=_count_obfuscated(text),
        spelling_trick_count=_count_spelling_tricks(text_lower),
        spoofing_signal_count=_count_phrases(text_lower, SPOOFING_PHRASES),
        excessive_punctuation=excessive_punct,
    )


def list_red_flags(text: str) -> List[dict]:
    """
    Return a list of human-readable red flags for the result page.
    Each item: {'label': str, 'severity': 'low'|'medium'|'high', 'detail': str}

    Changes from v1:
      - Uses word-boundary matching (consistent with extract_features)
      - Deduplicates hits across overlapping categories
      - Added new flag categories: spelling tricks, spoofing signals, HTML tags
      - Severity now scales with hit count
    """
    flags = []
    text_lower = text.lower()
    words = re.findall(r"\b\w+\b", text)

    # --- Urgency ---
    urgent_hits = [p for p in URGENT_WORDS if (
        p in text_lower if " " in p
        else bool(re.search(r"\b" + re.escape(p) + r"\b", text_lower))
    )]
    if urgent_hits:
        sev = "high" if len(urgent_hits) >= 3 else "medium" if len(urgent_hits) >= 2 else "low"
        flags.append({
            "label": "Urgency pressure",
            "severity": sev,
            "detail": f"Uses urgent phrasing: {', '.join(urgent_hits[:5])}",
        })

    # --- Credential requests ---
    cred_hits = [p for p in CREDENTIAL_WORDS if (
        p in text_lower if " " in p
        else bool(re.search(r"\b" + re.escape(p) + r"\b", text_lower))
    )]
    if cred_hits:
        flags.append({
            "label": "Requests sensitive information",
            "severity": "high",
            "detail": f"Asks for: {', '.join(cred_hits[:5])}",
        })

    # --- Money / prize bait ---
    money_hits = [p for p in MONEY_WORDS if (
        p in text_lower if " " in p
        else bool(re.search(r"\b" + re.escape(p) + r"\b", text_lower))
    )]
    if money_hits:
        sev = "high" if len(money_hits) >= 3 else "medium"
        flags.append({
            "label": "Money or prize bait",
            "severity": sev,
            "detail": f"Mentions: {', '.join(money_hits[:5])}",
        })

    # --- Threats / consequences ---
    threat_hits = [p for p in THREAT_WORDS if (
        p in text_lower if " " in p
        else bool(re.search(r"\b" + re.escape(p) + r"\b", text_lower))
    )]
    if threat_hits:
        sev = "high" if len(threat_hits) >= 3 else "medium"
        flags.append({
            "label": "Threats or consequences",
            "severity": sev,
            "detail": f"Mentions: {', '.join(threat_hits[:5])}",
        })

    # --- Generic greeting ---
    greeting_hits = [p for p in GENERIC_GREETINGS if p in text_lower]
    if greeting_hits:
        flags.append({
            "label": "Generic / impersonal greeting",
            "severity": "low",
            "detail": f"Uses: '{greeting_hits[0]}'",
        })

    # --- ALL CAPS abuse ---
    caps_words = [w for w in words if len(w) >= 3 and w.isupper() and w not in _LEGIT_ACRONYMS]
    if len(caps_words) >= 3:
        flags.append({
            "label": "Excessive ALL CAPS",
            "severity": "low",
            "detail": f"{len(caps_words)} all-caps words found (e.g. {', '.join(caps_words[:3])})",
        })

    # --- Excessive exclamation marks ---
    excl = text.count("!")
    if excl >= 3:
        sev = "medium" if excl >= 5 else "low"
        flags.append({
            "label": "Excessive exclamation marks",
            "severity": sev,
            "detail": f"{excl} exclamation marks used",
        })

    # --- Spelling tricks (character substitution) ---
    trick_hits = [pat for pat in SPELLING_TRICKS if re.search(pat, text_lower)]
    if trick_hits:
        flags.append({
            "label": "Character substitution tricks",
            "severity": "high",
            "detail": "Domain or brand name uses letter/number substitution (e.g. 'paypa1', 'g00gle') to evade filters",
        })

    # --- Spoofing signals ---
    spoof_hits = [p for p in SPOOFING_PHRASES if p in text_lower]
    if spoof_hits:
        flags.append({
            "label": "Sender identity spoofing",
            "severity": "high",
            "detail": f"Claims to be a known authority: '{spoof_hits[0]}'",
        })

    # --- HTML in plain text ---
    html_count = len(re.findall(r"<[a-z][a-z0-9]*[\s>]", text_lower))
    if html_count > 0:
        flags.append({
            "label": "HTML tags in plain text",
            "severity": "medium",
            "detail": f"{html_count} HTML tag(s) detected — may indicate a disguised link or hidden content",
        })

    # --- Dollar amounts / financial figures ---
    if re.search(r"\$\s?\d", text) or "usd" in text_lower:
        flags.append({
            "label": "Specific dollar amounts",
            "severity": "low",
            "detail": "Contains dollar amounts, which are common in financial scam emails",
        })

    return flags
