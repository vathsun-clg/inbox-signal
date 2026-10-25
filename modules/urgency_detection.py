import re
from datetime import datetime
from typing import Dict, List

URGENCY_PATTERNS = [
    ("service outage", re.compile(r"\b(?:production|service|system|site|api)\b.{0,35}\b(?:down|outage|unavailable|offline)\b", re.I)),
    ("security incident", re.compile(r"\b(?:security incident|data breach|account compromised|suspected intrusion|ransomware)\b", re.I)),
    ("urgent request", re.compile(r"\b(?:urgent action required|urgent request|immediate action|as soon as possible|asap)\b", re.I)),
    ("time-sensitive approval", re.compile(r"\b(?:immediate approval|approve immediately|approval required today)\b", re.I)),
]
DEADLINE_PATTERNS = [
    re.compile(r"\b(?:deadline|due|respond|reply|submit|complete|approve|send).{0,35}\b(?:today|tomorrow|tonight)\b", re.I),
    re.compile(r"\b(?:by|before)\s+(?:today|tomorrow|tonight|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I),
    re.compile(r"\b(?:by|before)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b", re.I),
    re.compile(r"\b(?:on|by|before)\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?\b", re.I),
]


def _has_negated_urgent(text: str) -> bool:
    return bool(re.search(r"\b(?:not|never|no longer)\s+(?:urgent|an emergency|time[- ]sensitive)\b", text, re.I))


def detect_urgency(text: object, now: datetime | None = None) -> Dict[str, object]:
    value = "" if text is None else str(text)
    now = now or datetime.now()
    evidence: List[str] = []
    reasons: List[str] = []
    if _has_negated_urgent(value):
        reasons.append("The message explicitly negates urgency; that phrase was not treated as a positive signal.")
    for label, pattern in URGENCY_PATTERNS:
        match = pattern.search(value)
        if match and not (label == "urgent request" and _has_negated_urgent(value)):
            evidence.append(match.group(0).strip())
            reasons.append(label)
    deadline_phrases: List[str] = []
    for pattern in DEADLINE_PATTERNS:
        for match in pattern.finditer(value):
            phrase = match.group(0).strip()
            if phrase not in deadline_phrases:
                deadline_phrases.append(phrase)
    if any(re.search(r"\b(?:today|tonight)\b", phrase, re.I) for phrase in deadline_phrases):
        reasons.append("A same-day time constraint was detected.")
    elif any(re.search(r"\btomorrow\b", phrase, re.I) for phrase in deadline_phrases):
        reasons.append("A next-day time constraint was detected.")
    elif deadline_phrases:
        reasons.append("A possible deadline phrase was detected; confirm the exact date/time when ambiguous.")
    high_signal = bool(set(reasons) & {"service outage", "security incident", "time-sensitive approval"})
    same_day = any(re.search(r"\b(?:today|tonight)\b", p, re.I) for p in deadline_phrases)
    next_day = any(re.search(r"\btomorrow\b", p, re.I) for p in deadline_phrases)
    if high_signal or same_day:
        level = "High"
    elif evidence or next_day or deadline_phrases:
        level = "Medium"
    else:
        level = "Low"
    return {
        "urgency_level": level,
        "evidence": evidence,
        "reasons": reasons,
        "deadline_phrases": deadline_phrases,
        "deadline_count": len(deadline_phrases),
        "deadline_uncertain": any(re.search(r"\b(?:by|before)\s+\d{1,2}\b", p, re.I) for p in deadline_phrases),
        "analyzed_at": now.isoformat(timespec="seconds"),
    }
