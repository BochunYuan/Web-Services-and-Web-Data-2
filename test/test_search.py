"""Tests for the search module."""

import math

import pytest

from src.crawler import CrawlResult
from src.indexer import InvertedIndex
from src.search import SearchEngine, SearchResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PAGES = [
    CrawlResult("http://e.com/1", "Good friends make good days", "Page 1"),
    CrawlResult("http://e.com/2", "Friends are the family you choose. Good friends are rare.", "Page 2"),
    CrawlResult("http://e.com/3", "A beautiful day in the neighborhood", "Page 3"),
    CrawlResult("http://e.com/4", "Stars shine bright in the night sky", "Page 4"),
]


@pytest.fixture
def engine():
    idx = InvertedIndex()
    idx.build(PAGES)
    return SearchEngine(idx)


@pytest.fixture
def index():
    idx = InvertedIndex()
    idx.build(PAGES)
    return idx


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------

class TestSearchResult:
    def test_attributes(self):
        sr = SearchResult("http://x.com", 0.5, "Title")
        assert sr.url == "http://x.com"
        assert sr.score == 0.5
        assert sr.title == "Title"
        assert sr.term_details == {}

    def test_default_title(self):
        sr = SearchResult("http://x.com", 0.1)
        assert sr.title == ""

    def test_repr(self):
        sr = SearchResult("http://x.com", 1.2345)
        r = repr(sr)
        assert "x.com" in r
        assert "1.2345" in r


# ---------------------------------------------------------------------------
# SearchEngine.print_word
# ---------------------------------------------------------------------------

class TestPrintWord:
    def test_existing_word(self, engine):
        output = engine.print_word("good")
        assert 'Inverted index for "good"' in output
        assert "Frequency:" in output

    def test_nonexistent_word(self, engine):
        output = engine.print_word("zzzzz")
        assert "not found" in output

    def test_empty_string(self, engine):
        output = engine.print_word("")
        assert "Please provide a word" in output

    def test_whitespace_only(self, engine):
        output = engine.print_word("   ")
        assert "Please provide a word" in output

    def test_case_insensitive(self, engine):
        output = engine.print_word("GOOD")
        assert 'Inverted index for "good"' in output


# ---------------------------------------------------------------------------
# SearchEngine.find — single word
# ---------------------------------------------------------------------------

class TestFindSingleWord:
    def test_word_in_multiple_pages(self, engine):
        results = engine.find("friends")
        assert len(results) == 2
        urls = {r.url for r in results}
        assert "http://e.com/1" in urls
        assert "http://e.com/2" in urls

    def test_word_in_one_page(self, engine):
        results = engine.find("neighborhood")
        assert len(results) == 1
        assert results[0].url == "http://e.com/3"

    def test_nonexistent_word(self, engine):
        results = engine.find("zzzzz")
        assert results == []

    def test_case_insensitive(self, engine):
        r1 = engine.find("good")
        r2 = engine.find("GOOD")
        assert len(r1) == len(r2)
        assert {r.url for r in r1} == {r.url for r in r2}


# ---------------------------------------------------------------------------
# SearchEngine.find — multi-word
# ---------------------------------------------------------------------------

class TestFindMultiWord:
    def test_both_words_present(self, engine):
        results = engine.find("good friends")
        urls = {r.url for r in results}
        assert "http://e.com/1" in urls
        assert "http://e.com/2" in urls

    def test_no_common_page(self, engine):
        results = engine.find("neighborhood stars")
        assert results == []

    def test_partial_missing_term(self, engine):
        results = engine.find("good zzzzz")
        assert results == []

    def test_all_terms_missing(self, engine):
        results = engine.find("xxxx yyyy")
        assert results == []

    def test_quoted_phrase_requires_exact_match(self, engine):
        results = engine.find('"family you"')
        assert len(results) == 1
        assert results[0].url == "http://e.com/2"

    def test_quoted_phrase_missing_returns_no_results(self, engine):
        assert engine.find('"friends good"') == []

    def test_phrase_and_term_query(self, engine):
        results = engine.find('"family you" choose')
        assert len(results) == 1
        assert results[0].url == "http://e.com/2"
        assert "family you" in results[0].matched_phrases


# ---------------------------------------------------------------------------
# SearchEngine.find — edge cases
# ---------------------------------------------------------------------------

class TestFindEdgeCases:
    def test_empty_query(self, engine):
        assert engine.find("") == []

    def test_punctuation_only(self, engine):
        assert engine.find("!!! ???") == []

    def test_numbers_only(self, engine):
        assert engine.find("123 456") == []

    def test_duplicate_terms(self, engine):
        results = engine.find("good good")
        assert len(results) > 0

    def test_internal_match_start_without_term_hit(self, engine):
        start, match_length = engine._find_match_start(
            url="http://e.com/1",
            phrases=[],
            terms=["missing"],
        )
        assert start is None
        assert match_length == 1


# ---------------------------------------------------------------------------
# TF-IDF ranking order
# ---------------------------------------------------------------------------

class TestRanking:
    def test_higher_frequency_ranks_first(self, engine):
        results = engine.find("good")
        assert len(results) >= 2
        assert results[0].score >= results[1].score

    def test_scores_are_positive(self, engine):
        results = engine.find("friends")
        for r in results:
            assert r.score >= 0

    def test_term_details_populated(self, engine):
        results = engine.find("good friends")
        for r in results:
            assert "good" in r.term_details
            assert "friends" in r.term_details
            assert r.term_details["good"] >= 1
            assert r.term_details["friends"] >= 1

    def test_title_populated(self, engine):
        results = engine.find("neighborhood")
        assert results[0].title == "Page 3"

    def test_exact_phrase_ranks_first(self):
        idx = InvertedIndex()
        idx.build([
            CrawlResult("http://phrase.com/1", "good friends make life better", "Phrase"),
            CrawlResult("http://phrase.com/2", "good people trust loyal friends", "Non Phrase"),
        ])
        engine = SearchEngine(idx)
        results = engine.find("good friends")
        assert results[0].url == "http://phrase.com/1"
        assert "good friends" in results[0].matched_phrases
        assert results[0].snippet


# ---------------------------------------------------------------------------
# SearchEngine.format_results
# ---------------------------------------------------------------------------

class TestFormatResults:
    def test_with_results(self, engine):
        results = engine.find("good")
        output = engine.format_results("good", results)
        assert 'Results for "good"' in output
        assert "page(s) found" in output
        assert "Score:" in output

    def test_no_results(self, engine):
        output = engine.format_results("zzzzz", [])
        assert "No results found" in output

    def test_empty_query(self, engine):
        output = engine.format_results("", [])
        assert "Please provide a search query" in output

    def test_shows_rank_numbers(self, engine):
        results = engine.find("good")
        output = engine.format_results("good", results)
        assert "1." in output

    def test_shows_title(self, engine):
        results = engine.find("neighborhood")
        output = engine.format_results("neighborhood", results)
        assert "Title: Page 3" in output

    def test_shows_term_frequencies(self, engine):
        results = engine.find("good friends")
        output = engine.format_results("good friends", results)
        assert "Term frequencies:" in output

    def test_result_without_title(self, engine):
        sr = SearchResult("http://x.com", 0.5, "")
        output = engine.format_results("test", [sr])
        assert "Title:" not in output

    def test_shows_phrase_match_and_snippet(self, engine):
        results = engine.find("good friends")
        output = engine.format_results("good friends", results)
        assert "Phrase match:" in output
        assert "Snippet:" in output
