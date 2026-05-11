"""Search operations over the inverted index using TF-IDF ranking."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

try:
    from .indexer import InvertedIndex, tokenize
except ImportError:  # pragma: no cover - allows direct execution from src/.
    from indexer import InvertedIndex, tokenize


@dataclass(slots=True)
class SearchResult:
    """Represents a ranked search result for a query."""

    url: str
    title: str
    score: float
    phrase_matches: int
    term_frequencies: dict[str, int]
    tfidf_scores: dict[str, float] = field(default_factory=dict)


class SearchEngine:
    """Provides print and find behaviour on top of the inverted index."""

    def __init__(self, index: InvertedIndex) -> None:
        """Initialise the search engine with a loaded inverted index."""
        self.index = index

    def print_term(self, term: str) -> dict[str, dict[str, int | list[int]]]:
        """Return the raw postings list for a single search term."""
        return self.index.get_postings(term)

    def find(self, query: str) -> list[SearchResult]:
        """Search the index using TF-IDF scoring with phrase match bonus."""
        terms = tokenize(query)
        if not terms:
            raise ValueError("Query must contain at least one searchable word.")

        candidate_pages: set[str] | None = None
        for term in terms:
            postings = self.index.index.get(term, {})
            urls = set(postings)
            candidate_pages = urls if candidate_pages is None else candidate_pages & urls

        if not candidate_pages:
            return []

        total_docs = self.index.page_count

        results: list[SearchResult] = []
        for url in candidate_pages:
            term_frequencies: dict[str, int] = {}
            tfidf_scores: dict[str, float] = {}
            tfidf_total = 0.0

            for term in dict.fromkeys(terms):
                count = int(self.index.index[term][url]["count"])
                term_frequencies[term] = count

                word_count = int(self.index.pages[url].get("word_count", 1))
                tf = count / word_count

                doc_freq = len(self.index.index.get(term, {}))
                idf = math.log(1 + total_docs / (1 + doc_freq))

                tfidf = tf * idf
                tfidf_scores[term] = round(tfidf, 6)
                tfidf_total += tfidf

            phrase_matches = self._count_phrase_matches(url, terms)
            phrase_bonus = phrase_matches * 0.5 * len(terms)
            score = round(tfidf_total + phrase_bonus, 6)

            page_title = str(self.index.pages.get(url, {}).get("title", url))
            results.append(
                SearchResult(
                    url=url,
                    title=page_title,
                    score=score,
                    phrase_matches=phrase_matches,
                    term_frequencies=term_frequencies,
                    tfidf_scores=tfidf_scores,
                )
            )

        results.sort(key=lambda result: (-result.score, -result.phrase_matches, result.url))
        return results

    def _count_phrase_matches(self, url: str, terms: list[str]) -> int:
        """Count how many times the query terms appear consecutively on a page."""
        if len(terms) < 2:
            return 0

        position_sets = [
            set(self.index.index[term][url]["positions"])  # type: ignore[index]
            for term in terms
        ]

        phrase_matches = 0
        for start_position in position_sets[0]:
            if all((start_position + offset) in positions for offset, positions in enumerate(position_sets[1:], start=1)):
                phrase_matches += 1
        return phrase_matches

