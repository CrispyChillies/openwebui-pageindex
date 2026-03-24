import json
import logging
import os
from typing import Any

import PyPDF2
from openai import OpenAI

from pageindex.utils import extract_json
from tree_traversal import collect_all_node_ids, find_node_by_id, traverse


class PageIndexNodeTools:
    """Tool layer for node lookup, tree navigation, and full-text retrieval."""

    def __init__(self, tree_data: dict, source_path: str | None = None, logger: logging.Logger | None = None):
        self.tree_data = tree_data
        self.top_nodes: list[dict] = tree_data.get("structure", [])
        self.logger = logger or logging.getLogger(__name__)

        self.source_type = tree_data.get("source_type", "pdf")
        self.source_path = source_path or tree_data.get("source_path")
        self._pdf_pages: list[str] | None = None

        self._all_node_ids = collect_all_node_ids(self.top_nodes)

    def is_valid_node_id(self, node_id: str) -> bool:
        return node_id in self._all_node_ids

    def get_node(self, node_id: str) -> dict | None:
        if not self.is_valid_node_id(node_id):
            return None
        return find_node_by_id(self.top_nodes, node_id)

    def node_summary_lookup(self, node_id: str) -> dict[str, Any] | None:
        node = self.get_node(node_id)
        if not node:
            return None
        return {
            "node_id": node.get("node_id"),
            "title": node.get("title"),
            "summary": node.get("summary", ""),
            "start_index": node.get("start_index"),
            "end_index": node.get("end_index"),
        }

    def child_traversal(self, node_id: str | None = None) -> list[dict[str, Any]]:
        """Return children for a node, or top-level nodes if node_id is None."""
        if node_id is None:
            children = self.top_nodes
        else:
            parent = self.get_node(node_id)
            if not parent:
                return []
            children = parent.get("nodes", [])

        out = []
        for child in children:
            out.append(
                {
                    "node_id": child.get("node_id"),
                    "title": child.get("title"),
                    "summary": child.get("summary", ""),
                    "start_index": child.get("start_index"),
                    "end_index": child.get("end_index"),
                    "child_count": len(child.get("nodes", [])),
                }
            )
        return out

    def _load_pdf_pages(self) -> None:
        if self._pdf_pages is not None:
            return
        self._pdf_pages = []

        if not self.source_path or not os.path.isfile(self.source_path):
            self.logger.warning("Source PDF path is missing or invalid: %s", self.source_path)
            return

        try:
            reader = PyPDF2.PdfReader(self.source_path)
            for page in reader.pages:
                self._pdf_pages.append(page.extract_text() or "")
            self.logger.info("Loaded %d PDF pages for on-demand text retrieval", len(self._pdf_pages))
        except Exception as exc:
            self.logger.exception("Failed to load PDF pages: %s", exc)
            self._pdf_pages = []

    def _get_text_from_page_range(self, start_page: int, end_page: int) -> str:
        if self.source_type != "pdf":
            return ""

        self._load_pdf_pages()
        if not self._pdf_pages:
            return ""

        safe_start = max(1, start_page)
        safe_end = min(len(self._pdf_pages), end_page)
        if safe_end < safe_start:
            return ""

        return "\n".join(self._pdf_pages[safe_start - 1 : safe_end])

    def full_text_retrieval(self, node_id: str, adjacent_pages: int = 0) -> dict[str, Any] | None:
        """
        Retrieve full text for a node by node_id.
        Uses embedded node text if present and no adjacency is requested.
        Falls back to source document page-range extraction when possible.
        """
        node = self.get_node(node_id)
        if not node:
            return None

        start = node.get("start_index")
        end = node.get("end_index")
        if not isinstance(start, int) or not isinstance(end, int):
            return {
                "node_id": node.get("node_id"),
                "title": node.get("title"),
                "start_index": start,
                "end_index": end,
                "text": node.get("text", ""),
                "source": "node_text_only",
            }

        if adjacent_pages == 0 and node.get("text"):
            return {
                "node_id": node.get("node_id"),
                "title": node.get("title"),
                "start_index": start,
                "end_index": end,
                "text": node.get("text", ""),
                "source": "embedded_node_text",
            }

        expanded_start = max(1, start - adjacent_pages)
        expanded_end = end + adjacent_pages
        text = self._get_text_from_page_range(expanded_start, expanded_end)

        if not text and node.get("text"):
            return {
                "node_id": node.get("node_id"),
                "title": node.get("title"),
                "start_index": start,
                "end_index": end,
                "text": node.get("text", ""),
                "source": "embedded_node_text_fallback",
            }

        return {
            "node_id": node.get("node_id"),
            "title": node.get("title"),
            "start_index": start,
            "end_index": end,
            "expanded_start_index": expanded_start,
            "expanded_end_index": expanded_end,
            "text": text,
            "source": "source_document_pages",
        }

    def adjacent_context_retrieval(self, node_id: str, adjacent_pages: int = 1) -> dict[str, Any] | None:
        return self.full_text_retrieval(node_id=node_id, adjacent_pages=adjacent_pages)


class AgenticPageIndexQA:
    """Agent orchestration: tree-summary navigation first, then fetch full text on demand."""

    def __init__(
        self,
        tree_data: dict,
        client: OpenAI,
        model: str,
        source_path: str | None = None,
        logger: logging.Logger | None = None,
    ):
        self.tree_data = tree_data
        self.client = client
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
        self.tools = PageIndexNodeTools(tree_data=tree_data, source_path=source_path, logger=self.logger)

    def _chat_json(self, system_prompt: str, user_prompt: str, max_tokens: int = 2048) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        raw = (response.choices[0].message.content or "").strip()
        parsed = extract_json(raw)
        if isinstance(parsed, dict):
            return parsed
        return {}

    def _summary_sufficiency_check(self, query: str, retrieved_nodes: list[dict]) -> dict:
        node_briefs = []
        retrieved_id_set: set[str] = set()
        for node in retrieved_nodes:
            node_id = node.get("node_id")
            if isinstance(node_id, str):
                retrieved_id_set.add(node_id)
            node_briefs.append(
                {
                    "node_id": node_id,
                    "title": node.get("title"),
                    "start_index": node.get("start_index"),
                    "end_index": node.get("end_index"),
                    "summary": node.get("summary", ""),
                }
            )

        system_prompt = (
            "You decide whether summaries are sufficient to answer a query. "
            "Return strict JSON only."
        )
        user_prompt = (
            "Query:\n"
            f"{query}\n\n"
            "Retrieved node summaries:\n"
            f"{json.dumps(node_briefs, ensure_ascii=False)}\n\n"
            "Rules:\n"
            "- focus_node_ids must list the most relevant node_id values for final answering.\n"
            "- If any retrieved node is relevant, focus_node_ids must not be empty.\n"
            "- needs_full_text_node_ids must be a subset of focus_node_ids.\n"
            "- If summaries are insufficient for specific details, put those node IDs in needs_full_text_node_ids.\n\n"
            "Return JSON with this exact schema:\n"
            "{"
            '\"summary_enough\": \"yes or no\", '
            '\"reason\": \"short reason\", '
            '\"focus_node_ids\": [\"0001\", \"0002\"], '
            '\"needs_full_text_node_ids\": [\"0002\"]'
            "}"
        )
        result = self._chat_json(system_prompt, user_prompt)

        summary_enough_raw = str(result.get("summary_enough", "no")).strip().lower()
        focus_node_ids = result.get("focus_node_ids", [])
        needs_full_text_node_ids = result.get("needs_full_text_node_ids", [])
        if not isinstance(focus_node_ids, list):
            focus_node_ids = []
        if not isinstance(needs_full_text_node_ids, list):
            needs_full_text_node_ids = []

        valid_focus_ids = [
            nid
            for nid in focus_node_ids
            if isinstance(nid, str) and self.tools.is_valid_node_id(nid) and nid in retrieved_id_set
        ]
        valid_needs_ids = [
            nid
            for nid in needs_full_text_node_ids
            if isinstance(nid, str) and self.tools.is_valid_node_id(nid) and nid in retrieved_id_set
        ]

        if not valid_focus_ids:
            # Deterministic fallback to keep the pipeline robust when LLM omits focus IDs.
            valid_focus_ids = [
                node.get("node_id")
                for node in retrieved_nodes
                if isinstance(node.get("node_id"), str)
            ]

        valid_focus_set = set(valid_focus_ids)
        valid_needs_ids = [nid for nid in valid_needs_ids if nid in valid_focus_set]

        return {
            "summary_enough": "yes" if summary_enough_raw == "yes" else "no",
            "reason": result.get("reason", ""),
            "focus_node_ids": valid_focus_ids,
            "needs_full_text_node_ids": valid_needs_ids,
        }

    def _build_evidence_packets(
        self,
        retrieved_nodes: list[dict],
        summary_enough: bool,
        focus_node_ids: list[str],
        needs_full_text_node_ids: list[str],
        adjacent_pages: int,
        max_evidence_nodes: int,
    ) -> tuple[list[dict], bool, bool]:
        evidence = []
        used_full_text = False
        attempted_full_text = False

        retrieved_ids = [
            nid
            for nid in (n.get("node_id") for n in retrieved_nodes)
            if isinstance(nid, str)
        ]
        selected_ids = [nid for nid in (focus_node_ids or retrieved_ids) if isinstance(nid, str)]
        selected_ids = selected_ids[:max_evidence_nodes]
        selected_id_set = set(selected_ids)

        fetch_full_text_ids: set[str] = set()
        if not summary_enough:
            if needs_full_text_node_ids:
                fetch_full_text_ids = {
                    nid for nid in needs_full_text_node_ids if isinstance(nid, str) and nid in selected_id_set
                }
            else:
                # Conservative fallback: expand only focus nodes, never all retrieved nodes.
                fetch_full_text_ids = {
                    nid for nid in focus_node_ids if isinstance(nid, str) and nid in selected_id_set
                }
                if not fetch_full_text_ids and selected_ids:
                    # Last-resort guard for malformed sufficiency output.
                    fetch_full_text_ids = {selected_ids[0]}

        for node_id in selected_ids:
            summary_info = self.tools.node_summary_lookup(node_id)
            if not summary_info:
                continue

            packet = {
                "node_id": summary_info.get("node_id"),
                "title": summary_info.get("title"),
                "start_index": summary_info.get("start_index"),
                "end_index": summary_info.get("end_index"),
                "summary": summary_info.get("summary", ""),
            }

            if not summary_enough and node_id in fetch_full_text_ids:
                attempted_full_text = True
                text_info = self.tools.full_text_retrieval(node_id=node_id, adjacent_pages=adjacent_pages)
                if text_info and text_info.get("text"):
                    packet["text"] = text_info.get("text", "")
                    packet["text_source"] = text_info.get("source", "")
                    packet["expanded_start_index"] = text_info.get("expanded_start_index")
                    packet["expanded_end_index"] = text_info.get("expanded_end_index")
                    used_full_text = True

            evidence.append(packet)

        return evidence, used_full_text, attempted_full_text

    def _validate_citations(self, citations: list[Any], evidence_packets: list[dict]) -> list[dict[str, Any]]:
        evidence_by_id: dict[str, dict[str, Any]] = {}
        for packet in evidence_packets:
            node_id = packet.get("node_id")
            if isinstance(node_id, str):
                evidence_by_id[node_id] = packet

        validated: list[dict[str, Any]] = []
        for citation in citations:
            if not isinstance(citation, dict):
                continue

            node_id = citation.get("node_id")
            if not isinstance(node_id, str) or node_id not in evidence_by_id:
                continue

            evidence = evidence_by_id[node_id]
            start_index = evidence.get("start_index")
            end_index = evidence.get("end_index")
            if not isinstance(start_index, int) or not isinstance(end_index, int):
                continue

            title = citation.get("title")
            if not isinstance(title, str) or not title.strip():
                title = evidence.get("title") or "Untitled"

            cited_start = citation.get("start_index")
            cited_end = citation.get("end_index")
            if (
                not isinstance(cited_start, int)
                or not isinstance(cited_end, int)
                or cited_start > cited_end
                or cited_start != start_index
                or cited_end != end_index
            ):
                cited_start = start_index
                cited_end = end_index

            validated.append(
                {
                    "node_id": node_id,
                    "title": str(title),
                    "start_index": cited_start,
                    "end_index": cited_end,
                }
            )

        return validated

    def _grounded_answer(self, query: str, evidence_packets: list[dict]) -> dict:
        allowed_node_ids = [packet.get("node_id") for packet in evidence_packets if packet.get("node_id")]
        system_prompt = (
            "You are a rigorous QA assistant. Use only the provided evidence packets. "
            "Do not use outside knowledge or assumptions. "
            "If evidence is insufficient, state this clearly. "
            "Citations must only use node_id values from provided evidence packets. "
            "Return strict JSON only."
        )
        user_prompt = (
            "Query:\n"
            f"{query}\n\n"
            "Allowed citation node_ids:\n"
            f"{json.dumps(allowed_node_ids, ensure_ascii=False)}\n\n"
            "Evidence packets:\n"
            f"{json.dumps(evidence_packets, ensure_ascii=False)}\n\n"
            "Return JSON using this exact schema:\n"
            "{"
            '\"answer\": \"final grounded answer\", '
            '\"evidence_sufficient\": \"yes or no\", '
            '\"insufficient_reason\": \"empty string if yes\", '
            '\"citations\": ['
            '{\"node_id\":\"0001\",\"start_index\":1,\"end_index\":2,\"title\":\"Intro\"}'
            "]"
            "}"
        )
        return self._chat_json(system_prompt, user_prompt, max_tokens=2500)

    def answer(
        self,
        query: str,
        adjacent_pages: int = 0,
        max_evidence_nodes: int = 6,
    ) -> dict[str, Any]:
        self.logger.info("Starting agentic QA for query: %s", query)

        retrieved_nodes = traverse(
            tree_data=self.tree_data,
            query=query,
            client=self.client,
            model=self.model,
            verbose=False,
            log_progress=False,
        )

        if not retrieved_nodes:
            return {
                "answer": "Insufficient evidence: no relevant nodes were retrieved from the tree.",
                "evidence_sufficient": "no",
                "insufficient_reason": "No relevant nodes selected during tree traversal.",
                "citations": [],
                "retrieved_node_ids": [],
                "summary_enough": "no",
                "used_full_text": False,
            }

        sufficiency = self._summary_sufficiency_check(query=query, retrieved_nodes=retrieved_nodes)
        summary_enough = sufficiency.get("summary_enough") == "yes"

        self.logger.info(
            "Summary sufficiency decision: %s | reason: %s",
            sufficiency.get("summary_enough"),
            sufficiency.get("reason", ""),
        )

        evidence_packets, used_full_text, attempted_full_text = self._build_evidence_packets(
            retrieved_nodes=retrieved_nodes,
            summary_enough=summary_enough,
            focus_node_ids=sufficiency.get("focus_node_ids", []),
            needs_full_text_node_ids=sufficiency.get("needs_full_text_node_ids", []),
            adjacent_pages=adjacent_pages,
            max_evidence_nodes=max_evidence_nodes,
        )

        full_text_unavailable_reason = ""
        if not summary_enough and attempted_full_text and not used_full_text:
            full_text_unavailable_reason = (
                "Full-text expansion was requested but unavailable for selected nodes. "
                "The JSON appears to include summaries only (no node text), and source document pages "
                "could not be loaded from source_path."
            )

        if not evidence_packets:
            return {
                "answer": "Insufficient evidence: unable to collect evidence packets from retrieved nodes.",
                "evidence_sufficient": "no",
                "insufficient_reason": full_text_unavailable_reason or "No usable evidence packets found.",
                "citations": [],
                "retrieved_node_ids": [n.get("node_id") for n in retrieved_nodes if n.get("node_id")],
                "summary_enough": "yes" if summary_enough else "no",
                "used_full_text": used_full_text,
                "full_text_unavailable_reason": full_text_unavailable_reason,
            }

        grounded = self._grounded_answer(query=query, evidence_packets=evidence_packets)

        citations = grounded.get("citations", [])
        if not isinstance(citations, list):
            citations = []
        citations = self._validate_citations(citations, evidence_packets)

        answer = grounded.get("answer", "").strip()
        evidence_sufficient = grounded.get("evidence_sufficient", "no")
        insufficient_reason = grounded.get("insufficient_reason", "")

        if not answer:
            if full_text_unavailable_reason:
                answer = f"Insufficient evidence: {full_text_unavailable_reason}"
            else:
                answer = "Insufficient evidence: the model did not return a usable grounded answer."
            evidence_sufficient = "no"

        if evidence_sufficient != "yes" and not insufficient_reason and full_text_unavailable_reason:
            insufficient_reason = full_text_unavailable_reason

        return {
            "answer": answer,
            "evidence_sufficient": "yes" if evidence_sufficient == "yes" else "no",
            "insufficient_reason": insufficient_reason,
            "citations": citations,
            "retrieved_node_ids": [n.get("node_id") for n in retrieved_nodes if n.get("node_id")],
            "summary_enough": "yes" if summary_enough else "no",
            "used_full_text": used_full_text,
            "summary_sufficiency_reason": sufficiency.get("reason", ""),
            "full_text_unavailable_reason": full_text_unavailable_reason,
        }
