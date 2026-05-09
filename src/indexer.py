"""Inverted index builder with TF-IDF scoring.

Processes crawled page content into a searchable inverted index that
stores word frequency, positions, and TF-IDF weights for each word
across all documents.
"""

import json
import math
import re
import logging
import os
from typing import Any

from src.crawler import CrawlResult

logger = logging.getLogger(__name__)

DEFAULT_INDEX_PATH = os.path.join("data", "index.json")


def tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens, stripping punctuation.

    Keeps only alphabetic characters and apostrophes within words.
    All tokens are converted to lowercase for case-insensitive matching.

    Args:
        text: Raw text string to tokenize.

    Returns:
        List of lowercase word tokens in order of appearance.
    """
    tokens = re.findall(r"[a-zA-Z']+", text)
    return [t.strip("'").lower() for t in tokens if t.strip("'")]


class PostingEntry:
    """Index entry for a single word in a single document.

    Attributes:
        frequency: Number of times the word appears in the document.
        positions: List of word positions (0-based) where the word occurs.
        tf: Term frequency (frequency / total words in document).
    """

    def __init__(self) -> None:
        self.frequency: int = 0
        self.positions: list[int] = []
        self.tf: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "frequency": self.frequency,
            "positions": self.positions,
            "tf": round(self.tf, 6),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PostingEntry":
        entry = cls()
        entry.frequency = data["frequency"]
        entry.positions = data["positions"]
        entry.tf = data.get("tf", 0.0)
        return entry


class InvertedIndex:
    """Inverted index mapping words to their occurrences across documents.

    Structure:
        index[word][url] = PostingEntry(frequency, positions, tf)

    Also stores document metadata (word counts) and precomputed IDF values
    for TF-IDF scoring.
    """

    def __init__(self) -> None:
        self.index: dict[str, dict[str, PostingEntry]] = {}
        self.doc_lengths: dict[str, int] = {}
        self.doc_titles: dict[str, str] = {}
        self.doc_tokens: dict[str, list[str]] = {}
        self.idf: dict[str, float] = {}
        self._total_docs: int = 0

    @property
    def total_docs(self) -> int:
        return self._total_docs

    @property
    def total_words(self) -> int:
        return len(self.index)

    def build(self, pages: list[CrawlResult]) -> None:
        """Build the inverted index from a list of crawled pages.

        For each page, tokenizes the text, records word positions and
        frequencies, computes TF for each word-document pair, then
        computes IDF across all documents.

        Args:
            pages: List of CrawlResult objects from the crawler.
        """
        self.index.clear()
        self.doc_lengths.clear()
        self.doc_titles.clear()
        self.doc_tokens.clear()
        self.idf.clear()
        self._total_docs = len(pages)

        logger.info(f"Building index from {self._total_docs} pages...")

        for page in pages:
            tokens = tokenize(page.text)
            self.doc_lengths[page.url] = len(tokens)
            self.doc_titles[page.url] = page.title
            self.doc_tokens[page.url] = tokens

            for position, word in enumerate(tokens):
                if word not in self.index:
                    self.index[word] = {}
                if page.url not in self.index[word]:
                    self.index[word][page.url] = PostingEntry()

                entry = self.index[word][page.url]
                entry.frequency += 1
                entry.positions.append(position)

        self._compute_tf()
        self._compute_idf()

        logger.info(
            f"Index built: {self.total_words} unique words "
            f"across {self._total_docs} pages"
        )

    def _compute_tf(self) -> None:
        """Compute term frequency for every word-document pair.

        TF(t, d) = count(t in d) / total_words(d)
        """
        for word, postings in self.index.items():
            for url, entry in postings.items():
                doc_len = self.doc_lengths.get(url, 1)
                entry.tf = entry.frequency / doc_len if doc_len > 0 else 0.0

    def _compute_idf(self) -> None:
        """Compute inverse document frequency for every word.

        IDF(t) = log(N / df(t))
        where N = total documents, df(t) = documents containing term t.
        Uses log base e. Adds 1 to denominator to avoid division by zero
        is unnecessary here since every word has at least one document.
        """
        for word, postings in self.index.items():
            df = len(postings)
            self.idf[word] = math.log(self._total_docs / df) if df > 0 else 0.0

    def get_tfidf(self, word: str, url: str) -> float:
        """Compute TF-IDF score for a word in a specific document.

        Args:
            word: The search term (lowercase).
            url: The document URL.

        Returns:
            TF-IDF score, or 0.0 if the word is not in the document.
        """
        word = word.lower()
        if word not in self.index or url not in self.index[word]:
            return 0.0
        return self.index[word][url].tf * self.idf.get(word, 0.0)

    def get_posting(self, word: str) -> dict[str, PostingEntry] | None:
        """Retrieve the posting list for a word.

        Args:
            word: The search term (case-insensitive).

        Returns:
            Dictionary mapping URLs to PostingEntry objects, or None
            if the word is not in the index.
        """
        return self.index.get(word.lower())

    def get_documents_for_word(self, word: str) -> set[str]:
        """Return the set of document URLs that contain a given word."""
        posting = self.get_posting(word)
        if posting is None:
            return set()
        return set(posting.keys())

    def get_phrase_positions(self, url: str, terms: list[str]) -> list[int]:
        """Return the starting positions of an exact phrase in a document.

        Args:
            url: Document URL to inspect.
            terms: Tokenized phrase terms in order.

        Returns:
            Sorted list of starting positions where the exact phrase occurs.
        """
        normalized_terms = [term.lower() for term in terms if term.strip()]
        if not normalized_terms:
            return []

        postings = []
        for term in normalized_terms:
            posting = self.get_posting(term)
            if posting is None or url not in posting:
                return []
            postings.append(posting[url])

        candidate_positions = set(postings[0].positions)
        for offset, entry in enumerate(postings[1:], start=1):
            shifted = {position - offset for position in entry.positions}
            candidate_positions &= shifted
            if not candidate_positions:
                return []

        return sorted(candidate_positions)

    def get_documents_for_phrase(self, terms: list[str]) -> set[str]:
        """Return documents containing an exact multi-word phrase."""
        normalized_terms = [term.lower() for term in terms if term.strip()]
        if not normalized_terms:
            return set()

        doc_sets = [self.get_documents_for_word(term) for term in normalized_terms]
        if any(not docs for docs in doc_sets):
            return set()

        candidate_urls = set.intersection(*doc_sets)
        return {
            url for url in candidate_urls
            if self.get_phrase_positions(url, normalized_terms)
        }

    def get_snippet(
        self,
        url: str,
        start_position: int,
        match_length: int = 1,
        window: int = 5,
    ) -> str:
        """Build a short snippet around a matched position.

        Args:
            url: Document URL to extract a snippet from.
            start_position: Start index of the matched term or phrase.
            match_length: Number of consecutive matched tokens.
            window: Number of context tokens to show on each side.

        Returns:
            A human-readable snippet with the matched span highlighted.
        """
        tokens = self.doc_tokens.get(url, [])
        if not tokens or start_position < 0 or start_position >= len(tokens):
            return ""

        start = max(0, start_position - window)
        end = min(len(tokens), start_position + match_length + window)

        leading = "... " if start > 0 else ""
        trailing = " ..." if end < len(tokens) else ""
        snippet_tokens = tokens[start:end]

        highlight_start = start_position - start
        highlight_end = highlight_start + match_length
        highlighted_tokens = []
        for index, token in enumerate(snippet_tokens):
            if highlight_start <= index < highlight_end:
                highlighted_tokens.append(f"[{token}]")
            else:
                highlighted_tokens.append(token)

        return leading + " ".join(highlighted_tokens) + trailing

    def save(self, filepath: str = DEFAULT_INDEX_PATH) -> None:
        """Serialize the index to a JSON file.

        Args:
            filepath: Path where the index file will be saved.
        """
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        data = {
            "total_docs": self._total_docs,
            "doc_lengths": self.doc_lengths,
            "doc_titles": self.doc_titles,
            "doc_tokens": self.doc_tokens,
            "idf": {w: round(v, 6) for w, v in self.idf.items()},
            "index": {
                word: {
                    url: entry.to_dict()
                    for url, entry in postings.items()
                }
                for word, postings in self.index.items()
            },
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        file_size = os.path.getsize(filepath)
        logger.info(f"Index saved to {filepath} ({file_size / 1024:.1f} KB)")

    def load(self, filepath: str = DEFAULT_INDEX_PATH) -> None:
        """Deserialize the index from a JSON file.

        Args:
            filepath: Path to the previously saved index file.

        Raises:
            FileNotFoundError: If the index file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._total_docs = data["total_docs"]
        self.doc_lengths = data["doc_lengths"]
        self.doc_titles = data.get("doc_titles", {})
        self.doc_tokens = data.get("doc_tokens", {})
        self.idf = {w: float(v) for w, v in data["idf"].items()}

        self.index = {}
        for word, postings in data["index"].items():
            self.index[word] = {
                url: PostingEntry.from_dict(entry_data)
                for url, entry_data in postings.items()
            }

        logger.info(
            f"Index loaded from {filepath}: {self.total_words} words, "
            f"{self._total_docs} pages"
        )

    def format_posting(self, word: str) -> str:
        """Format the posting list for a word as a human-readable string.

        Used by the `print` command to display index entries.

        Args:
            word: The word to look up (case-insensitive).

        Returns:
            Formatted string showing all documents and statistics for the word.
        """
        word = word.lower()
        posting = self.get_posting(word)

        if posting is None:
            return f'Word "{word}" not found in index.'

        lines = [f'Inverted index for "{word}" (IDF: {self.idf.get(word, 0):.4f}):']
        sorted_postings = sorted(
            posting.items(),
            key=lambda item: item[1].frequency,
            reverse=True,
        )

        for url, entry in sorted_postings:
            tfidf = self.get_tfidf(word, url)
            lines.append(f"  Page: {url}")
            lines.append(f"    Frequency: {entry.frequency}")
            lines.append(f"    Positions: {entry.positions}")
            lines.append(f"    TF: {entry.tf:.6f}")
            lines.append(f"    TF-IDF: {tfidf:.6f}")

        return "\n".join(lines)
