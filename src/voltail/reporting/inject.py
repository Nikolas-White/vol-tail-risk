"""Replace auto-generated blocks inside hand-written markdown files.

A block looks like::

    <!-- AUTO:key_results -->
    ...anything; replaced on every `make report`...
    <!-- /AUTO:key_results -->

Only text *between* the markers is touched, so prose around it stays hand-written
while every number inside it comes from the pipeline.
"""

from __future__ import annotations

import re
from pathlib import Path

_BLOCK = re.compile(
    r"(<!-- AUTO:(?P<key>[a-z0-9_]+) -->)(?P<body>.*?)(<!-- /AUTO:(?P=key) -->)", re.S
)


def inject_blocks(text: str, blocks: dict[str, str]) -> tuple[str, list[str]]:
    """Return ``text`` with known blocks replaced, plus the list of keys replaced."""
    replaced: list[str] = []

    def _sub(m: re.Match[str]) -> str:
        key = m.group("key")
        if key not in blocks:
            return m.group(0)
        replaced.append(key)
        return f"{m.group(1)}\n{blocks[key].strip()}\n{m.group(4)}"

    return _BLOCK.sub(_sub, text), replaced


def inject_file(path: Path, blocks: dict[str, str]) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    new, replaced = inject_blocks(text, blocks)
    if new != text:
        path.write_text(new, encoding="utf-8")
    return replaced


def block_keys(text: str) -> list[str]:
    return [m.group("key") for m in _BLOCK.finditer(text)]
