import re
from typing import Dict, List


def normalize_whitespace(text: object) -> str:
    if text is None:
        return ""
    value = str(text).replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def remove_quoted_reply(text: str) -> tuple[str, bool]:
    lines = text.splitlines()
    kept: List[str] = []
    removed = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(">"):
            removed = True
            continue
        if re.match(r"^(On .{2,160} wrote:|From:.+|Sent:.+|-----Original Message-----)$", stripped, re.I):
            removed = True
            break
        kept.append(line)
    return "\n".join(kept).strip(), removed


def remove_signature(text: str) -> tuple[str, bool]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped in {"--", "__"} and index >= 1:
            return "\n".join(lines[:index]).strip(), True
        if index >= 1 and re.match(r"^(best|kind regards|regards|thanks|thank you|sincerely),?$", stripped, re.I):
            tail = lines[index:]
            if len(tail) <= 4:
                return "\n".join(lines[:index]).strip(), True
    return text.strip(), False


def preprocess_email(subject: object, body: object) -> Dict[str, object]:
    raw_subject = normalize_whitespace(subject)
    raw_body = normalize_whitespace(body)
    combined = "\n".join(part for part in (raw_subject, raw_body) if part).strip()
    warnings: List[str] = []
    if not raw_subject:
        warnings.append("Subject is empty.")
    if not raw_body:
        warnings.append("Body is empty.")
    if len(combined.split()) < 6:
        warnings.append("Message is unusually short; classification may be unreliable.")
    without_quotes, removed_quotes = remove_quoted_reply(combined)
    without_signature, removed_sig = remove_signature(without_quotes)
    processed = normalize_whitespace(without_signature)
    if removed_quotes:
        warnings.append("A quoted reply section was removed using conservative patterns.")
    if removed_sig:
        warnings.append("A likely signature section was removed; review the processed text if needed.")
    return {
        "subject": raw_subject,
        "body": raw_body,
        "original_text": combined,
        "processed_text": processed,
        "warnings": warnings,
        "removed_quoted_reply": removed_quotes,
        "removed_signature": removed_sig,
        "is_short": len(processed.split()) < 6,
    }
