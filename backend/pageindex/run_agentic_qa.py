import argparse
import json
import logging
import os
import sys

import httpx
from dotenv import load_dotenv
from openai import OpenAI

from pageindex.agentic_qa import AgenticPageIndexQA
from pageindex.doc_selector import load_json_documents, select_document_for_query


def build_client(model: str) -> OpenAI:
    load_dotenv()
    if model.lower().startswith("qwen"):
        qwen_api_key = os.getenv("QWEN_API_KEY", "")
        qwen_base_url = os.getenv("QWEN_BASE_URL", "")

        if not qwen_api_key or not qwen_base_url:
            raise ValueError("QWEN_API_KEY and QWEN_BASE_URL must be set in .env for Qwen models")

        http_client = httpx.Client(verify=False)
        return OpenAI(api_key=qwen_api_key, base_url=qwen_base_url, http_client=http_client)

    openai_api_key = os.getenv("CHATGPT_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    openai_base_url = os.getenv("OPENAI_BASE_URL", "")
    if not openai_api_key:
        raise ValueError("CHATGPT_API_KEY or OPENAI_API_KEY must be set in .env for non-Qwen models")

    if openai_base_url:
        return OpenAI(api_key=openai_api_key, base_url=openai_base_url)
    return OpenAI(api_key=openai_api_key)


def format_citation_line(citation: dict) -> str:
    node_id = citation.get("node_id", "?")
    title = citation.get("title", "(untitled)")
    start = citation.get("start_index", "?")
    end = citation.get("end_index", "?")
    return f"- [{node_id}] {title} | pages {start}-{end}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic PageIndex QA over tree + on-demand full text")
    parser.add_argument("--json_path", type=str, default=None, help="Path to one PageIndex structure JSON")
    parser.add_argument("--json_dir", type=str, default=None, help="Directory of multiple PageIndex JSON files")
    parser.add_argument("--query", type=str, required=True, help="Question to ask")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-32B-Instruct", help="LLM model")
    parser.add_argument(
        "--adjacent-pages",
        type=int,
        default=0,
        help="When summaries are insufficient, include +/- this many adjacent pages around each selected node",
    )
    parser.add_argument(
        "--max-evidence-nodes",
        type=int,
        default=6,
        help="Maximum number of nodes to include in evidence synthesis",
    )
    parser.add_argument(
        "--source-path",
        type=str,
        default=None,
        help="Optional source document path override (otherwise loaded from JSON metadata)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("agentic_qa_cli")

    if not args.json_path and not args.json_dir:
        print("Error: provide either --json_path or --json_dir", file=sys.stderr)
        sys.exit(1)

    if args.json_path and args.json_dir:
        print("Error: provide only one of --json_path or --json_dir", file=sys.stderr)
        sys.exit(1)

    selected_doc_meta = None
    selection_debug = None

    if args.json_path:
        if not os.path.isfile(args.json_path):
            print(f"Error: JSON file not found: {args.json_path}", file=sys.stderr)
            sys.exit(1)
        with open(args.json_path, "r", encoding="utf-8") as f:
            tree_data = json.load(f)
        selected_doc_meta = {
            "doc_id": tree_data.get("doc_id") or os.path.splitext(os.path.basename(args.json_path))[0],
            "doc_title": tree_data.get("doc_title") or tree_data.get("doc_name") or os.path.basename(args.json_path),
            "doc_description": tree_data.get("doc_description", ""),
            "path": args.json_path,
        }
        selection_debug = {
            "reason": "Single document mode (--json_path).",
            "ranking": [
                {
                    "doc_id": selected_doc_meta["doc_id"],
                    "doc_title": selected_doc_meta["doc_title"],
                    "score": 1.0,
                }
            ],
            "uncertain": False,
        }
    else:
        try:
            docs = load_json_documents(args.json_dir)
        except Exception as exc:
            print(f"Error loading directory: {exc}", file=sys.stderr)
            sys.exit(1)

        if not docs:
            print(f"Error: no valid JSON files found in {args.json_dir}", file=sys.stderr)
            sys.exit(1)

        sel = select_document_for_query(args.query, docs)
        selected = sel.get("selected_doc")
        if not selected:
            print("Insufficient evidence: no document could be selected.")
            print("Selection reason:", sel.get("reason", ""))
            sys.exit(0)

        selected_doc_meta = {
            "doc_id": selected.get("doc_id"),
            "doc_title": selected.get("doc_title"),
            "doc_description": selected.get("doc_description", ""),
            "path": selected.get("path"),
        }
        selection_debug = {
            "reason": sel.get("reason", ""),
            "ranking": sel.get("ranking", []),
            "uncertain": sel.get("uncertain", False),
        }
        tree_data = selected.get("tree_data", {})

    try:
        client = build_client(args.model)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    agent = AgenticPageIndexQA(
        tree_data=tree_data,
        client=client,
        model=args.model,
        source_path=args.source_path,
        logger=logger,
    )

    result = agent.answer(
        query=args.query,
        adjacent_pages=max(0, args.adjacent_pages),
        max_evidence_nodes=max(1, args.max_evidence_nodes),
    )

    print("\n" + "=" * 70)
    print("AGENTIC PAGEINDEX QA RESULT")
    print("=" * 70)
    print("Selected document:")
    print(f"- id: {selected_doc_meta.get('doc_id')}")
    print(f"- title: {selected_doc_meta.get('doc_title')}")
    print(f"- path: {selected_doc_meta.get('path')}")
    if selected_doc_meta.get("doc_description"):
        print(f"- description: {selected_doc_meta.get('doc_description')}")
    print("Selection debug:")
    print(f"- reason: {selection_debug.get('reason', '')}")
    print(f"- uncertain: {selection_debug.get('uncertain', False)}")
    if selection_debug.get("ranking"):
        print("- ranking:")
        for row in selection_debug["ranking"]:
            print(f"  - {row.get('doc_id')} | {row.get('doc_title')} | score={row.get('score')}")
    print(f"Query: {args.query}")
    print(f"Evidence sufficient: {result.get('evidence_sufficient', 'no')}")
    print(f"Summary enough: {result.get('summary_enough', 'no')}")
    print(f"Used full text: {result.get('used_full_text', False)}")

    insuff_reason = result.get("insufficient_reason", "")
    if insuff_reason:
        print(f"Insufficient reason: {insuff_reason}")

    print("\nAnswer:")
    print(result.get("answer", ""))

    citations = result.get("citations", [])
    print("\nCitations:")
    if citations:
        for citation in citations:
            print(format_citation_line(citation))
    else:
        print("- (none)")


if __name__ == "__main__":
    main()
