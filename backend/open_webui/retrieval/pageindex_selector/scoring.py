from __future__ import annotations

import math
from collections import Counter


def clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def lexical_overlap_score(query_tokens: list[str], haystack: str) -> float:
    if not query_tokens:
        return 0.0
    text = (haystack or "").lower()
    matched = 0
    for token in query_tokens:
        if token in text:
            matched += 1
    return clip01(matched / max(1, len(query_tokens)))


def rare_token_overlap_score(rare_tokens: list[str], haystacks: list[str]) -> float:
    if not rare_tokens:
        return 0.0
    text = "\n".join(haystacks).upper()
    matched = sum(1 for token in rare_tokens if token in text)
    return clip01(matched / max(1, len(rare_tokens)))


def semantic_chunk_score(chunk_scores: list[float]) -> float:
    if not chunk_scores:
        return 0.0
    top = sorted((clip01(s) for s in chunk_scores), reverse=True)[:5]
    weighted = 0.0
    denom = 0.0
    for i, score in enumerate(top):
        w = 1.0 / (1 + i * 0.5)
        weighted += score * w
        denom += w
    return clip01(weighted / max(denom, 1e-6))


def metadata_score(header_scores: list[float], title_desc_overlap: float) -> float:
    header = max((clip01(s) for s in header_scores), default=0.0)
    return clip01((header * 0.75) + (clip01(title_desc_overlap) * 0.25))


def evidence_coverage_score(node_ids: list[str], page_numbers: list[int], chunk_count: int) -> float:
    node_diversity = clip01(len(set(x for x in node_ids if x)) / 4.0)
    page_diversity = clip01(len(set(x for x in page_numbers if isinstance(x, int))) / 4.0)
    chunk_support = clip01(chunk_count / 6.0)
    return clip01((node_diversity * 0.35) + (page_diversity * 0.25) + (chunk_support * 0.40))


def filename_relevance_score(query_tokens: list[str], file_name: str) -> float:
    return lexical_overlap_score(query_tokens, file_name or "")


def combine_document_score(
    semantic_score: float,
    metadata_score_value: float,
    exact_overlap_score: float,
    coverage_score: float,
    filename_score: float,
) -> tuple[float, dict[str, float]]:
    breakdown = {
        "semantic_chunk_score": clip01(semantic_score),
        "metadata_score": clip01(metadata_score_value),
        "exact_token_overlap_score": clip01(exact_overlap_score),
        "evidence_coverage_score": clip01(coverage_score),
        "filename_score": clip01(filename_score),
    }
    final_score = (
        breakdown["semantic_chunk_score"] * 0.40
        + breakdown["metadata_score"] * 0.20
        + breakdown["exact_token_overlap_score"] * 0.25
        + breakdown["evidence_coverage_score"] * 0.10
        + breakdown["filename_score"] * 0.05
    )
    breakdown["final_score"] = round(clip01(final_score), 6)
    return breakdown["final_score"], breakdown


def rerank_adjustment(
    current_score: float,
    exact_overlap_score: float,
    metadata_score_value: float,
    best_chunk_score: float,
) -> float:
    strict = (
        clip01(current_score) * 0.55
        + clip01(exact_overlap_score) * 0.25
        + clip01(metadata_score_value) * 0.10
        + clip01(best_chunk_score) * 0.10
    )
    return clip01(strict)


def confidence_from_scores(scores: list[float]) -> str:
    if not scores:
        return "low"
    top = scores[0]
    second = scores[1] if len(scores) > 1 else 0.0
    margin = top - second
    if top >= 0.72 and margin >= 0.10:
        return "high"
    if top >= 0.55 and margin >= 0.05:
        return "medium"
    return "low"
