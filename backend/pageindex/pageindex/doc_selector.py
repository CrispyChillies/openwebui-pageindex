import json
import os
import re
from pathlib import Path


def _tokenize(text: str) -> set[str]:
    if not text:
        return set()
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def load_json_documents(json_dir: str) -> list[dict]:
    """Load PageIndex JSON files from a directory with lightweight metadata."""
    base = Path(json_dir)
    if not base.exists() or not base.is_dir():
        raise ValueError(f"Invalid json_dir: {json_dir}")

    docs: list[dict] = []
    for file_path in sorted(base.glob("*.json")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            # Skip malformed JSON in demo mode, keep behavior lightweight.
            continue

        doc_id = data.get("doc_id") or file_path.stem
        doc_title = data.get("doc_title") or data.get("doc_name") or file_path.stem
        doc_description = data.get("doc_description", "")

        docs.append(
            {
                "doc_id": doc_id,
                "doc_title": doc_title,
                "doc_description": doc_description,
                "path": str(file_path),
                "tree_data": data,
            }
        )

    return docs


def select_document_for_query(query: str, docs: list[dict], min_score: float = 0.08) -> dict:
    """
    Select the most relevant document using simple keyword overlap over doc_description.

    Returns:
      {
        'selected_doc': dict | None,
        'ranking': [{'doc_id':..., 'doc_title':..., 'score':...}],
        'reason': str,
        'uncertain': bool,
      }
    """
    if not docs:
        return {
            "selected_doc": None,
            "ranking": [],
            "reason": "No documents available.",
            "uncertain": True,
        }

    if len(docs) == 1:
        only = docs[0]
        return {
            "selected_doc": only,
            "ranking": [
                {
                    "doc_id": only.get("doc_id"),
                    "doc_title": only.get("doc_title"),
                    "score": 1.0,
                }
            ],
            "reason": "Only one document is available; selected directly.",
            "uncertain": False,
        }

    query_tokens = _tokenize(query)
    ranking = []

    for doc in docs:
        desc = doc.get("doc_description", "")
        desc_tokens = _tokenize(desc)
        if not desc_tokens or not query_tokens:
            score = 0.0
        else:
            overlap = len(query_tokens & desc_tokens)
            score = overlap / max(1, len(query_tokens))

        ranking.append(
            {
                "doc_id": doc.get("doc_id"),
                "doc_title": doc.get("doc_title"),
                "score": round(score, 4),
                "_doc": doc,
            }
        )

    ranking.sort(key=lambda x: x["score"], reverse=True)
    best = ranking[0]
    uncertain = best["score"] < min_score

    public_ranking = [
        {"doc_id": item["doc_id"], "doc_title": item["doc_title"], "score": item["score"]}
        for item in ranking
    ]

    if uncertain:
        reason = (
            f"Best description overlap score is low ({best['score']:.4f}). "
            "Selection may be uncertain."
        )
    else:
        reason = f"Selected highest description overlap score: {best['score']:.4f}."

    return {
        "selected_doc": best["_doc"],
        "ranking": public_ranking,
        "reason": reason,
        "uncertain": uncertain,
    }


def run_query_on_selected_document(query: str, selected_doc: dict, runner_fn):
    """Tiny adapter to execute existing single-doc QA flow on the selected document."""
    return runner_fn(query=query, tree_data=selected_doc.get("tree_data", {}), selected_doc=selected_doc)
