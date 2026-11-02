"""Tokenize fork trees with the model's chat template and per-token loss labels.

Each child is rendered together with the root (the template formats earlier assistant turns
differently once a later user message exists, so the root is cut out of the joint rendering, at the
block where the child starts). Every child of a tree must then share the exact same root text; trees
where they do not are dropped. Labels are the token ids inside assistant blocks (after
`<|im_start|>assistant\\n`, through `<|im_end|>`) and -100 everywhere else: always in the children,
and in the root when the tree has `train_root`.
"""

from __future__ import annotations

import json
import re

IGNORE = -100
BLOCK = "<|im_start|>"
ASSISTANT = re.compile(r"<\|im_start\|>assistant\n(.*?<\|im_end\|>)", re.S)


def _arguments(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        return {"raw": str(raw)}
    return parsed if isinstance(parsed, dict) else {"raw": parsed}


def _for_template(message: dict) -> dict:
    """Agent logs keep OpenAI wire format (arguments as a JSON string, content may be None); the
    Qwen3.5 template wants argument mappings and string content."""
    out = dict(message)
    out["content"] = out.get("content") or ""
    if out.get("tool_calls"):
        out["tool_calls"] = [
            {**call, "function": {**call["function"], "arguments": _arguments(call["function"].get("arguments"))}}
            for call in out["tool_calls"]
        ]
    return out


def _render(tok, messages: list[dict], tools: list[dict]) -> str:
    return tok.apply_chat_template([_for_template(m) for m in messages], tools=tools, tokenize=False)


def _cut(full: str, n_root_blocks: int) -> int:
    pos = -1
    for _ in range(n_root_blocks + 1):
        pos = full.index(BLOCK, pos + 1)
    return pos


def _labels(tok, text: str) -> tuple[list[int], list[int]]:
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    spans = [m.span(1) for m in ASSISTANT.finditer(text)]
    labels = []
    for tid, (a, b) in zip(enc["input_ids"], enc["offset_mapping"]):
        labels.append(tid if any(s <= a and b <= e for s, e in spans) else IGNORE)
    return enc["input_ids"], labels


def encode_tree(tree: dict, tok, max_len: int, min_child_tokens: int = 256, max_tree_tokens: int | None = None) -> dict | None:
    """Token ids for the shared root and each child (with labels), or None if the tree cannot be
    shared exactly, root plus any child exceeds `max_len`, or root plus all children exceed
    `max_tree_tokens`. Over-long trees are dropped whole, never cut, so no branch or join loses its
    closing turns."""
    tools, root = tree["tools"], tree["root"]
    n_root_blocks = _render(tok, root, tools).count(BLOCK)
    root_text, children = None, []
    for child in tree["children"]:
        full = _render(tok, root + child["messages"], tools)
        cut = _cut(full, n_root_blocks)
        if root_text is None:
            root_text = full[:cut]
        elif full[:cut] != root_text:
            return None
        ids, labels = _labels(tok, full[cut:])
        children.append({"kind": child["kind"], "ids": ids, "labels": labels})
    root_ids, root_labels = _labels(tok, root_text)
    if not tree.get("train_root"):
        root_labels = [IGNORE] * len(root_ids)
    room = max_len - len(root_ids)
    if room < min_child_tokens or any(len(c["ids"]) > room for c in children):
        return None  # dropped whole: a cut child would lose its closing commit / join turns
    children = [c for c in children if any(x != IGNORE for x in c["labels"])]
    if not children:
        return None
    if max_tree_tokens and len(root_ids) + sum(len(c["ids"]) for c in children) > max_tree_tokens:
        return None
    return {"root_ids": root_ids, "root_labels": root_labels, "children": children}


def flat_rows(encoded: dict) -> list[dict]:
    """Method A: one ordinary sample per child. The root is repeated in every row but its labels
    (if any) go into the join row only, so each token is trained once, as in method B."""
    root = encoded["root_ids"]
    root_labels = encoded.get("root_labels") or [IGNORE] * len(root)
    carrier = next((i for i, c in enumerate(encoded["children"]) if c["kind"] == "join"), 0)
    return [
        {"input_ids": root + c["ids"], "labels": (root_labels if i == carrier else [IGNORE] * len(root)) + c["labels"], "kind": c["kind"]}
        for i, c in enumerate(encoded["children"])
    ]
