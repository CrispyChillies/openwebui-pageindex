from __future__ import annotations

import re
from collections import OrderedDict

RARE_TOKEN_PATTERN = re.compile(r"\b(?:[A-Z]{2,}[A-Z0-9-]*\d+[A-Z0-9-]*|NDA|BLA|IND)\b")
WORD_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]{1,}")


def normalize_whitespace(value: str) -> str:
    return " ".join((value or "").split())


def extract_query_tokens(query: str) -> list[str]:
    seen: OrderedDict[str, None] = OrderedDict()
    for token in WORD_PATTERN.findall((query or "").lower()):
        if len(token) >= 3:
            seen[token] = None
    return list(seen.keys())


def extract_rare_tokens(query: str) -> list[str]:
    seen: OrderedDict[str, None] = OrderedDict()
    for token in RARE_TOKEN_PATTERN.findall((query or "").upper()):
        seen[token] = None
    return list(seen.keys())


def build_selector_query_variants(query: str) -> list[str]:
    """Build deterministic, meaning-preserving query variants for retrieval."""
    q = normalize_whitespace(query)
    if not q:
        return []

    rare = extract_rare_tokens(q)
    tokens = extract_query_tokens(q)

    compressed = " ".join(tokens[:24])
    clinical_focus_parts = ["sufficient clinical data", "support submission of an NDA"]
    if rare:
        clinical_focus_parts.append(" ".join(rare))
    clinical_focus = " ; ".join(clinical_focus_parts)

    regulatory_parts = ["agency agreement", "health authority response", "regulatory communication"]
    if rare:
        regulatory_parts.append("studies " + " ".join(rare))
    regulatory_focus = " ; ".join(regulatory_parts)

    document_style = q
    if rare:
        document_style = f"Question: {q} Evidence terms: {' '.join(rare)}"

    candidates = [q, compressed, clinical_focus, regulatory_focus, document_style]

    deduped: OrderedDict[str, None] = OrderedDict()
    for item in candidates:
        normalized = normalize_whitespace(item)
        if normalized:
            deduped[normalized] = None
    return list(deduped.keys())
