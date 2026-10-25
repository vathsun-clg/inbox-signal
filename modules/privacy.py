import re
from typing import Dict

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d .()\-]{7,}\d)(?!\w)")


def mask_personal_data(text: object, enabled: bool = True) -> Dict[str, object]:
    value = "" if text is None else str(text)
    if not enabled:
        return {"text": value, "email_count": 0, "phone_count": 0, "masked": False}
    email_count = len(EMAIL_RE.findall(value))
    masked = EMAIL_RE.sub("[EMAIL]", value)
    phone_count = len(PHONE_RE.findall(masked))
    masked = PHONE_RE.sub("[PHONE]", masked)
    return {
        "text": masked,
        "email_count": email_count,
        "phone_count": phone_count,
        "masked": email_count > 0 or phone_count > 0,
    }
