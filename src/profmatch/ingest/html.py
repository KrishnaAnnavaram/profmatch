"""A small HTML tree (stdlib `html.parser`) with the few queries that the portal parser needs.

Tables are read by their header text, not by cell positions. A missing section gives an empty
result, never an exception.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

VOID = {"br", "img", "hr", "meta", "link", "input", "area", "base", "col", "source", "wbr"}


@dataclass
class Node:
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list["Node | str"] = field(default_factory=list)
    parent: "Node | None" = None

    def text(self) -> str:
        parts = []
        for c in self.children:
            parts.append(c if isinstance(c, str) else c.text())
        return re.sub(r"\s+", " ", " ".join(parts)).strip()

    def iter(self):
        yield self
        for c in self.children:
            if isinstance(c, Node):
                yield from c.iter()

    def find_all(self, tag: str) -> list["Node"]:
        return [n for n in self.iter() if n.tag == tag]

    def find(self, tag: str) -> "Node | None":
        found = self.find_all(tag)
        return found[0] if found else None


class _Builder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("document")
        self.cur = self.root

    def handle_starttag(self, tag, attrs):
        node = Node(tag, {k: v or "" for k, v in attrs}, parent=self.cur)
        self.cur.children.append(node)
        if tag not in VOID:
            self.cur = node

    def handle_endtag(self, tag):
        n = self.cur
        while n is not None and n.tag != tag:
            n = n.parent
        if n is not None and n.parent is not None:
            self.cur = n.parent

    def handle_data(self, data):
        if data.strip():
            self.cur.children.append(data)


def parse(html: str) -> Node:
    b = _Builder()
    b.feed(html or "")
    b.close()
    return b.root


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def tables(root: Node) -> list[list[list[str]]]:
    out = []
    for t in root.find_all("table"):
        rows = []
        for tr in t.find_all("tr"):
            cells = [c.text() for c in tr.children if isinstance(c, Node) and c.tag in ("td", "th")]
            if cells:
                rows.append(cells)
        if rows:
            out.append(rows)
    return out


def table_records(root: Node, aliases: dict[str, list[str]], required: list[str]) -> list[dict[str, str]]:
    """Rows of the first table whose header row has all `required` fields (matched by alias)."""
    for rows in tables(root):
        header = [_norm(h) for h in rows[0]]
        index = {}
        for key, names in aliases.items():
            for name in names:
                if _norm(name) in header:
                    index[key] = header.index(_norm(name))
                    break
        if all(k in index for k in required):
            recs = []
            for r in rows[1:]:
                recs.append({k: (r[i] if i < len(r) else "") for k, i in index.items()})
            return recs
    return []


def section_items(root: Node, heading_words: list[str]) -> list[str]:
    """List items (or comma-separated text) that follow a heading with one of `heading_words`."""
    wanted = [_norm(w) for w in heading_words]
    nodes = list(root.iter())
    for i, n in enumerate(nodes):
        if n.tag in ("h1", "h2", "h3", "h4", "h5", "h6") and any(w in _norm(n.text()) for w in wanted):
            for m in nodes[i + 1:]:
                if m.tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
                    break
                if m.tag in ("ul", "ol"):
                    return [li.text() for li in m.find_all("li") if li.text()]
                if m.tag == "p" and m.text():
                    return [x.strip() for x in m.text().split(",") if x.strip()]
            return []
    return []


def prefixed_value(root: Node, prefix: str) -> str:
    """Text after `prefix` in the first element whose own text starts with it (for example 'Title:')."""
    for n in root.iter():
        own = " ".join(c for c in n.children if isinstance(c, str)).strip()
        if own.lower().startswith(prefix.lower()):
            return own[len(prefix):].strip()
    return ""


def links(root: Node, contains: str) -> list[str]:
    seen, out = set(), []
    for a in root.find_all("a"):
        href = a.attrs.get("href", "")
        if contains in href and href not in seen:
            seen.add(href)
            out.append(href)
    return out
