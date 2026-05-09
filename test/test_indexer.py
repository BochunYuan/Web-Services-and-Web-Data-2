"""Tests for the inverted index module."""

import json
import math
import os
import tempfile

import pytest

from src.crawler import CrawlResult
from src.indexer import InvertedIndex, PostingEntry, tokenize


# ---------------------------------------------------------------------------
# tokenize()
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_basic(self):
        assert tokenize("hello world") == ["hello", "world"]

    def test_case_insensitive(self):
        assert tokenize("Hello WORLD") == ["hello", "world"]

    def test_strips_punctuation(self):
        tokens = tokenize("Hello, world! How are you?")
        assert tokens == ["hello", "world", "how", "are", "you"]

    def test_keeps_apostrophes(self):
        tokens = tokenize("it's don't can't")
        assert tokens == ["it's", "don't", "can't"]

    def test_strips_leading_trailing_apostrophes(self):
        tokens = tokenize("'hello' 'world'")
        assert tokens == ["hello", "world"]

    def test_empty_string(self):
        assert tokenize("") == []

    def test_numbers_ignored(self):
        tokens = tokenize("page 42 has 3 items")
        assert tokens == ["page", "has", "items"]

    def test_mixed_punctuation(self):
        tokens = tokenize("well-known semi;colon dash-board")
        assert tokens == ["well", "known", "semi", "colon", "dash", "board"]

    def test_only_punctuation(self):
        assert tokenize("!@#$%^&*()") == []

    def test_whitespace_only(self):
        assert tokenize("   \t\n  ") == []


# ---------------------------------------------------------------------------
# PostingEntry
# ---------------------------------------------------------------------------

class TestPostingEntry:
    def test_defaults(self):
        e = PostingEntry()
        assert e.frequency == 0
        assert e.positions == []
        assert e.tf == 0.0

    def test_to_dict(self):
        e = PostingEntry()
        e.frequency = 3
        e.positions = [0, 5, 10]
        e.tf = 0.15
        d = e.to_dict()
        assert d["frequency"] == 3
        assert d["positions"] == [0, 5, 10]
        assert d["tf"] == 0.15

    def test_from_dict(self):
        data = {"frequency": 2, "positions": [1, 7], "tf": 0.25}
        e = PostingEntry.from_dict(data)
        assert e.frequency == 2
        assert e.positions == [1, 7]
        assert e.tf == 0.25

    def test_from_dict_missing_tf(self):
        data = {"frequency": 1, "positions": [0]}
        e = PostingEntry.from_dict(data)
        assert e.tf == 0.0

    def test_round_trip(self):
        e = PostingEntry()
        e.frequency = 5
        e.positions = [2, 4, 6, 8, 10]
        e.tf = 0.123456
        restored = PostingEntry.from_dict(e.to_dict())
        assert restored.frequency == e.frequency
        assert restored.positions == e.positions
        assert abs(restored.tf - e.tf) < 1e-5


# ---------------------------------------------------------------------------
# InvertedIndex.build
# ---------------------------------------------------------------------------

SAMPLE_PAGES = [
    CrawlResult("http://a.com/1", "The cat sat on the mat", "Page A"),
    CrawlResult("http://a.com/2", "The dog chased the cat", "Page B"),
    CrawlResult("http://a.com/3", "A bird sang in the tree", "Page C"),
]


class TestInvertedIndexBuild:
    def setup_method(self):
        self.idx = InvertedIndex()
        self.idx.build(SAMPLE_PAGES)

    def test_total_docs(self):
        assert self.idx.total_docs == 3

    def test_total_words(self):
        assert self.idx.total_words > 0

    def test_doc_lengths(self):
        assert self.idx.doc_lengths["http://a.com/1"] == 6
        assert self.idx.doc_lengths["http://a.com/2"] == 5
        assert self.idx.doc_lengths["http://a.com/3"] == 6

    def test_doc_titles(self):
        assert self.idx.doc_titles["http://a.com/1"] == "Page A"

    def test_doc_tokens(self):
        assert self.idx.doc_tokens["http://a.com/1"] == [
            "the", "cat", "sat", "on", "the", "mat"
        ]

    def test_word_frequency(self):
        posting = self.idx.get_posting("the")
        assert posting is not None
        assert posting["http://a.com/1"].frequency == 2

    def test_word_positions(self):
        posting = self.idx.get_posting("the")
        assert posting["http://a.com/1"].positions == [0, 4]

    def test_case_insensitive_indexing(self):
        pages = [CrawlResult("http://x.com", "Hello HELLO hello")]
        idx = InvertedIndex()
        idx.build(pages)
        posting = idx.get_posting("hello")
        assert posting is not None
        assert posting["http://x.com"].frequency == 3

    def test_rebuild_clears_old_data(self):
        old_words = self.idx.total_words
        new_pages = [CrawlResult("http://b.com/1", "completely new content")]
        self.idx.build(new_pages)
        assert self.idx.total_docs == 1
        assert "the" not in self.idx.index

    def test_empty_pages(self):
        idx = InvertedIndex()
        idx.build([])
        assert idx.total_docs == 0
        assert idx.total_words == 0

    def test_single_page(self):
        idx = InvertedIndex()
        idx.build([CrawlResult("http://x.com", "solo")])
        assert idx.total_docs == 1
        assert idx.get_posting("solo") is not None


# ---------------------------------------------------------------------------
# TF and IDF computation
# ---------------------------------------------------------------------------

class TestTfIdf:
    def setup_method(self):
        self.idx = InvertedIndex()
        self.idx.build(SAMPLE_PAGES)

    def test_tf_values(self):
        posting = self.idx.get_posting("the")
        # "the" appears 2 times in doc1 (6 words) → TF = 2/6
        assert abs(posting["http://a.com/1"].tf - 2 / 6) < 1e-6

    def test_idf_common_word(self):
        # "the" appears in all 3 docs → IDF = log(3/3) = 0
        assert abs(self.idx.idf["the"] - math.log(3 / 3)) < 1e-6

    def test_idf_rare_word(self):
        # "dog" appears in 1 doc → IDF = log(3/1)
        assert abs(self.idx.idf["dog"] - math.log(3 / 1)) < 1e-6

    def test_get_tfidf_existing(self):
        score = self.idx.get_tfidf("cat", "http://a.com/1")
        expected = (1 / 6) * math.log(3 / 2)
        assert abs(score - expected) < 1e-6

    def test_get_tfidf_missing_word(self):
        assert self.idx.get_tfidf("nonexistent", "http://a.com/1") == 0.0

    def test_get_tfidf_missing_url(self):
        assert self.idx.get_tfidf("cat", "http://missing.com") == 0.0

    def test_get_tfidf_case_insensitive(self):
        score = self.idx.get_tfidf("CAT", "http://a.com/1")
        assert score > 0


# ---------------------------------------------------------------------------
# get_posting / get_documents_for_word
# ---------------------------------------------------------------------------

class TestPostingRetrieval:
    def setup_method(self):
        self.idx = InvertedIndex()
        self.idx.build(SAMPLE_PAGES)

    def test_get_posting_existing(self):
        posting = self.idx.get_posting("cat")
        assert posting is not None
        assert len(posting) == 2

    def test_get_posting_nonexistent(self):
        assert self.idx.get_posting("xyz") is None

    def test_get_posting_case_insensitive(self):
        assert self.idx.get_posting("CAT") is not None

    def test_get_documents_for_word(self):
        docs = self.idx.get_documents_for_word("cat")
        assert docs == {"http://a.com/1", "http://a.com/2"}

    def test_get_documents_for_nonexistent(self):
        assert self.idx.get_documents_for_word("xyz") == set()


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_round_trip(self):
        idx = InvertedIndex()
        idx.build(SAMPLE_PAGES)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_index.json")
            idx.save(path)

            assert os.path.exists(path)

            idx2 = InvertedIndex()
            idx2.load(path)

            assert idx2.total_docs == idx.total_docs
            assert idx2.total_words == idx.total_words
            assert idx2.doc_lengths == idx.doc_lengths
            assert idx2.doc_titles == idx.doc_titles
            assert idx2.doc_tokens == idx.doc_tokens

            for word in idx.index:
                assert word in idx2.index
                for url in idx.index[word]:
                    orig = idx.index[word][url]
                    loaded = idx2.index[word][url]
                    assert orig.frequency == loaded.frequency
                    assert orig.positions == loaded.positions
                    assert abs(orig.tf - loaded.tf) < 1e-5

    def test_load_file_not_found(self):
        idx = InvertedIndex()
        with pytest.raises(FileNotFoundError):
            idx.load("/nonexistent/path/index.json")

    def test_load_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {{{")
            tmp_path = f.name
        try:
            idx = InvertedIndex()
            with pytest.raises(json.JSONDecodeError):
                idx.load(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_save_creates_directories(self):
        idx = InvertedIndex()
        idx.build(SAMPLE_PAGES)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "index.json")
            idx.save(path)
            assert os.path.exists(path)

    def test_save_to_current_directory_file(self):
        idx = InvertedIndex()
        idx.build(SAMPLE_PAGES)

        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                idx.save("index.json")
                assert os.path.exists("index.json")
            finally:
                os.chdir(cwd)


# ---------------------------------------------------------------------------
# format_posting
# ---------------------------------------------------------------------------

class TestFormatPosting:
    def setup_method(self):
        self.idx = InvertedIndex()
        self.idx.build(SAMPLE_PAGES)

    def test_existing_word(self):
        output = self.idx.format_posting("cat")
        assert 'Inverted index for "cat"' in output
        assert "Frequency:" in output
        assert "Positions:" in output
        assert "TF:" in output
        assert "TF-IDF:" in output

    def test_nonexistent_word(self):
        output = self.idx.format_posting("xyz")
        assert "not found" in output

    def test_case_insensitive(self):
        output = self.idx.format_posting("CAT")
        assert 'Inverted index for "cat"' in output

    def test_sorted_by_frequency(self):
        pages = [
            CrawlResult("http://x.com/1", "apple apple apple"),
            CrawlResult("http://x.com/2", "apple"),
        ]
        idx = InvertedIndex()
        idx.build(pages)
        output = idx.format_posting("apple")
        lines = output.split("\n")
        page_lines = [l for l in lines if "Page:" in l]
        assert "x.com/1" in page_lines[0]
        assert "x.com/2" in page_lines[1]


# ---------------------------------------------------------------------------
# Phrase positions and snippets
# ---------------------------------------------------------------------------

class TestPhraseAndSnippetHelpers:
    def setup_method(self):
        self.idx = InvertedIndex()
        self.idx.build([
            CrawlResult(
                "http://x.com/1",
                "good friends make good memories with good friends",
                "Page 1",
            ),
            CrawlResult(
                "http://x.com/2",
                "good people can become trusted friends",
                "Page 2",
            ),
        ])

    def test_get_phrase_positions(self):
        positions = self.idx.get_phrase_positions("http://x.com/1", ["good", "friends"])
        assert positions == [0, 6]

    def test_get_phrase_positions_missing_phrase(self):
        positions = self.idx.get_phrase_positions("http://x.com/2", ["good", "friends"])
        assert positions == []

    def test_get_phrase_positions_empty_terms(self):
        assert self.idx.get_phrase_positions("http://x.com/1", []) == []

    def test_get_phrase_positions_missing_word(self):
        assert self.idx.get_phrase_positions("http://x.com/1", ["good", "unknown"]) == []

    def test_get_documents_for_phrase(self):
        docs = self.idx.get_documents_for_phrase(["good", "friends"])
        assert docs == {"http://x.com/1"}

    def test_get_documents_for_phrase_empty_terms(self):
        assert self.idx.get_documents_for_phrase([]) == set()

    def test_get_documents_for_phrase_missing_word(self):
        assert self.idx.get_documents_for_phrase(["unknown", "friends"]) == set()

    def test_get_snippet_highlights_match(self):
        snippet = self.idx.get_snippet(
            "http://x.com/1",
            start_position=0,
            match_length=2,
        )
        assert "[good]" in snippet
        assert "[friends]" in snippet

    def test_get_snippet_invalid_position(self):
        assert self.idx.get_snippet("http://x.com/1", start_position=99) == ""
