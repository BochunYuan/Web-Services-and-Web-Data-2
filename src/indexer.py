"""Inverted index creation and persistence."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Iterable

try:
    from .crawler import PageData
except ImportError:  # pragma: no cover - allows direct execution from src/.
    from crawler import PageData

WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def tokenize(text: str) -> list[str]:
    """Extract searchable, case-insensitive tokens from text."""

    return [match.group(0).lower() for match in WORD_PATTERN.finditer(text)]


class InvertedIndex:
    """Stores where each word appears across crawled pages."""

    def __init__(
        self,
        index: dict[str, dict[str, dict[str, int | list[int]]]] | None = None,
        pages: dict[str, dict[str, str | int]] | None = None,
    ) -> None:
        self.index = index or {}
        self.pages = pages or {}

    @classmethod
    def build(cls, pages: Iterable[PageData]) -> "InvertedIndex":
        instance = cls()
        for page in pages:
            instance.add_page(page)
        return instance

    @classmethod
    def load(cls, path: str | Path) -> "InvertedIndex":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(index=payload.get("index", {}), pages=payload.get("pages", {}))

    def add_page(self, page: PageData) -> None:
        tokens = tokenize(page.text)
        self.pages[page.url] = {
            "title": page.title,
            "word_count": len(tokens),
        }

        for position, token in enumerate(tokens):
            postings = self.index.setdefault(token, {})
            page_stats = postings.setdefault(page.url, {"count": 0, "positions": []})
            page_stats["count"] += 1
            page_stats["positions"].append(position)

    def get_postings(self, term: str) -> dict[str, dict[str, int | list[int]]]:
        tokens = tokenize(term)
        if not tokens:
            return {}
        return self.index.get(tokens[0], {})

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def to_dict(self) -> dict[str, dict[str, object]]:
        return {
            "index": self.index,
            "pages": self.pages,
        }

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def term_count(self) -> int:
        return len(self.index)

