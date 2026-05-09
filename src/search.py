"""Search engine with print and find commands.

Provides query processing over an InvertedIndex, supporting single-word
and multi-word searches with TF-IDF ranked results.
"""

import logging
import re

from src.indexer import InvertedIndex, tokenize

logger = logging.getLogger(__name__)


class SearchResult:
    """A single search result with its relevance score.

    Attributes:
        url: The page URL.
        score: Aggregate TF-IDF score for all query terms.
        title: The page title.
        term_details: Per-term frequency breakdown for this page.
    """

    def __init__(self, url: str, score: float, title: str = ""):
        self.url = url
        self.score = score
        self.title = title
        self.term_details: dict[str, int] = {}
        self.snippet: str = ""
        self.matched_phrases: list[str] = []

    def __repr__(self) -> str:
        return f"SearchResult(url={self.url!r}, score={self.score:.4f})"


class SearchEngine:
    """Query processor for the inverted index.

    Args:
        index: An InvertedIndex instance (must be built or loaded first).
    """

    def __init__(self, index: InvertedIndex):
        self.index = index

    def _parse_query(self, query: str) -> tuple[list[list[str]], list[str]]:
        """Parse a raw query into quoted phrases and standalone terms."""
        phrase_texts = re.findall(r'"([^"]+)"', query)
        phrases = []
        for phrase_text in phrase_texts:
            phrase_terms = tokenize(phrase_text)
            if phrase_terms:
                phrases.append(phrase_terms)

        query_without_phrases = re.sub(r'"[^"]+"', " ", query)
        terms = tokenize(query_without_phrases)
        return phrases, terms

    def _find_match_start(
        self,
        url: str,
        phrases: list[list[str]],
        terms: list[str],
    ) -> tuple[int | None, int]:
        """Find the best starting position for a result snippet."""
        for phrase_terms in phrases:
            phrase_positions = self.index.get_phrase_positions(url, phrase_terms)
            if phrase_positions:
                return phrase_positions[0], len(phrase_terms)

        if len(terms) > 1:
            full_query_positions = self.index.get_phrase_positions(url, terms)
            if full_query_positions:
                return full_query_positions[0], len(terms)

        best_position = None
        for term in terms:
            posting = self.index.get_posting(term)
            if posting and url in posting:
                position = posting[url].positions[0]
                if best_position is None or position < best_position:
                    best_position = position

        if best_position is None:
            return None, 1
        return best_position, 1

    def print_word(self, word: str) -> str:
        """Return the formatted inverted index entry for a single word.

        Delegates to InvertedIndex.format_posting for display.

        Args:
            word: The word to look up (case-insensitive).

        Returns:
            Human-readable string of the word's index entry.
        """
        if not word or not word.strip():
            return "Please provide a word to look up."
        return self.index.format_posting(word.strip())

    def find(self, query: str) -> list[SearchResult]:
        """Find all pages matching a query phrase.

        For single-word queries, returns all pages containing that word.
        For multi-word queries, returns pages containing ALL query terms
        (boolean AND). Quoted phrases such as "good friends" must match
        exactly. Results are ranked by aggregate TF-IDF score, with a
        phrase bonus when the full unquoted query appears contiguously.

        Args:
            query: One or more search terms separated by spaces.

        Returns:
            List of SearchResult objects sorted by score (descending).
            Returns an empty list if no pages match.
        """
        phrases, standalone_terms = self._parse_query(query)
        query_terms = tokenize(query)
        if not query_terms:
            return []

        required_doc_sets: list[set[str]] = []

        for phrase_terms in phrases:
            phrase_docs = self.index.get_documents_for_phrase(phrase_terms)
            if not phrase_docs:
                logger.info(f"Phrase not in index: {' '.join(phrase_terms)}")
                return []
            required_doc_sets.append(phrase_docs)

        for term in standalone_terms:
            term_docs = self.index.get_documents_for_word(term)
            if not term_docs:
                logger.info(f"Term not in index: {term}")
                return []
            required_doc_sets.append(term_docs)

        if not required_doc_sets:
            return []

        matching_urls = set.intersection(*required_doc_sets)
        results = []
        for url in matching_urls:
            score = sum(self.index.get_tfidf(term, url) for term in query_terms)

            result = SearchResult(
                url=url,
                score=score,
                title=self.index.doc_titles.get(url, ""),
            )
            for term in query_terms:
                posting = self.index.get_posting(term)
                if posting and url in posting:
                    result.term_details[term] = posting[url].frequency

            if len(query_terms) > 1:
                full_query_positions = self.index.get_phrase_positions(url, query_terms)
                if full_query_positions:
                    phrase_bonus = max(
                        1.0,
                        sum(self.index.idf.get(term, 0.0) for term in query_terms)
                        / len(query_terms),
                    )
                    result.score += phrase_bonus
                    result.matched_phrases.append(" ".join(query_terms))

            for phrase_terms in phrases:
                phrase_text = " ".join(phrase_terms)
                if phrase_text not in result.matched_phrases:
                    result.matched_phrases.append(phrase_text)

            start_position, match_length = self._find_match_start(
                url=url,
                phrases=phrases,
                terms=query_terms,
            )
            if start_position is not None:
                result.snippet = self.index.get_snippet(
                    url=url,
                    start_position=start_position,
                    match_length=match_length,
                )

            results.append(result)

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def format_results(self, query: str, results: list[SearchResult]) -> str:
        """Format search results as a human-readable string.

        Args:
            query: The original query string.
            results: List of SearchResult objects.

        Returns:
            Formatted string showing ranked results with scores.
        """
        if not results:
            terms = tokenize(query)
            if not terms:
                return "Please provide a search query."
            return f'No results found for "{query}".'

        lines = [f'Results for "{query}" ({len(results)} page(s) found):\n']

        for rank, result in enumerate(results, start=1):
            lines.append(f"  {rank}. {result.url}")
            if result.title:
                lines.append(f"     Title: {result.title}")
            lines.append(f"     Score: {result.score:.6f}")

            if result.term_details:
                detail_parts = [
                    f"{t}={f}" for t, f in result.term_details.items()
                ]
                lines.append(f"     Term frequencies: {', '.join(detail_parts)}")
            if result.matched_phrases:
                phrase_summary = ", ".join(f'"{phrase}"' for phrase in result.matched_phrases)
                lines.append(f"     Phrase match: {phrase_summary}")
            if result.snippet:
                lines.append(f"     Snippet: {result.snippet}")

        return "\n".join(lines)
