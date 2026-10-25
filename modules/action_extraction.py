import re
from typing import Dict, List

ACTION_RULES = [
    ("Reply to the sender", re.compile(r"\b(?:please|kindly|can you|could you|would you).{0,80}\b(?:reply|respond|confirm)\b", re.I)),
    ("Approve the request", re.compile(r"\b(?:please|kindly|can you|could you|need you to).{0,60}\b(?:approve|sign off)\b", re.I)),
    ("Attend the meeting", re.compile(r"\b(?:join|attend|participate in).{0,70}\b(?:meeting|call|session)\b", re.I)),
    ("Review the document", re.compile(r"\b(?:please|kindly|can you|could you).{0,80}\b(?:review|check|read).{0,60}\b(?:document|file|proposal|report|attachment)\b", re.I)),
    ("Complete the task", re.compile(r"\b(?:please|kindly|can you|could you|need you to).{0,70}\b(?:complete|finish|update|prepare|fix)\b", re.I)),
    ("Investigate the reported issue", re.compile(r"\b(?:please|kindly|can you|could you).{0,70}\b(?:investigate|look into|diagnose|troubleshoot)\b", re.I)),
    ("Submit the requested information", re.compile(r"\b(?:please|kindly|can you|could you).{0,70}\b(?:submit|send|share|provide).{0,70}\b(?:information|details|data|files|documents)\b", re.I)),
]


def extract_actions(text: object, urgency: Dict[str, object] | None = None) -> List[Dict[str, object]]:
    value = "" if text is None else str(text)
    urgency = urgency or {}
    deadlines = urgency.get("deadline_phrases", [])
    actions: List[Dict[str, object]] = []
    seen = set()
    sentences = re.split(r"(?<=[.!?])\s+|\n+", value)
    for description, rule in ACTION_RULES:
        for sentence in sentences:
            match = rule.search(sentence)
            if not match or description in seen:
                continue
            evidence = sentence.strip()
            associated = deadlines[0] if deadlines else None
            actions.append({
                "description": description,
                "deadline": associated,
                "source_text": evidence[:280],
                "confidence": "Medium" if associated else "High",
                "status": "Pending",
            })
            seen.add(description)
    return actions
