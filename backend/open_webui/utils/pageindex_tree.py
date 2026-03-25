from __future__ import annotations

from collections import defaultdict
from typing import Any


NodeDict = dict[str, Any]


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_root_nodes(tree_json: Any) -> list[NodeDict]:
    if isinstance(tree_json, list):
        return tree_json

    if isinstance(tree_json, dict):
        if isinstance(tree_json.get("structure"), list):
            return tree_json["structure"]
        if isinstance(tree_json.get("nodes"), list):
            return tree_json["nodes"]

    return []


def flatten_pageindex_tree(tree_json: Any) -> list[NodeDict]:
    root_nodes = _resolve_root_nodes(tree_json)
    flattened: list[NodeDict] = []

    def walk(nodes: list[NodeDict], parent_node_id: str | None, depth: int) -> None:
        for index, node in enumerate(nodes, start=1):
            if not isinstance(node, dict):
                continue

            raw_children = node.get("nodes")
            children: list[NodeDict] = raw_children if isinstance(raw_children, list) else []
            node_id = node.get("node_id")
            if not isinstance(node_id, str) or not node_id.strip():
                if parent_node_id:
                    node_id = f"{parent_node_id}.{index}"
                else:
                    node_id = str(index)

            flattened.append(
                {
                    "node_id": node_id,
                    "parent_node_id": parent_node_id,
                    "depth": depth,
                    "title": node.get("title"),
                    "summary": node.get("summary"),
                    "start_index": _to_int(node.get("start_index")),
                    "end_index": _to_int(node.get("end_index")),
                    "has_children": len(children) > 0,
                }
            )

            if children:
                walk(children, node_id, depth + 1)

    walk(root_nodes, parent_node_id=None, depth=0)
    return flattened


def reconstruct_pageindex_tree(flat_nodes: list[NodeDict]) -> list[NodeDict]:
    children_map: dict[str | None, list[NodeDict]] = defaultdict(list)

    for row in sorted(flat_nodes, key=lambda item: (item.get("depth", 0), item.get("node_id", ""))):
        node = {
            "node_id": row.get("node_id"),
            "title": row.get("title"),
            "summary": row.get("summary"),
            "start_index": row.get("start_index"),
            "end_index": row.get("end_index"),
            "nodes": [],
        }
        children_map[row.get("parent_node_id")].append(node)

    def attach(parent_id: str | None) -> list[NodeDict]:
        nodes = children_map.get(parent_id, [])
        for node in nodes:
            node["nodes"] = attach(node.get("node_id"))
        return nodes

    return attach(None)
