"""Tests for search behaviour, TF-IDF ranking, and the CLI entry point."""

from __future__ import annotations

import math

import pytest

from src.crawler import PageData
from src.indexer import InvertedIndex
from src.main import main
from src.search import SearchEngine, SearchResult


def build_sample_index() -> InvertedIndex:
    return InvertedIndex.build(
        [
            PageData(
                url="https://quotes.toscrape.com/",
                title="Home",
                text="Good friends good books and a sleepy conscience",
                links=[],
            ),
            PageData(
                url="https://quotes.toscrape.com/page/2/",
                title="Page Two",
                text="Friends build good habits",
                links=[],
            ),
            PageData(
                url="https://quotes.toscrape.com/page/3/",
                title="Page Three",
                text="Books open worlds",
                links=[],
            ),
        ]
    )


# ---------- Single-word queries ----------

def test_find_returns_results_for_single_word_query() -> None:
    engine = SearchEngine(build_sample_index())

    results = engine.find("good")

    assert len(results) == 2
    urls = {r.url for r in results}
    assert "https://quotes.toscrape.com/" in urls
    assert "https://quotes.toscrape.com/page/2/" in urls
    assert all(r.score > 0 for r in results)


def test_find_single_word_returns_correct_frequencies() -> None:
    engine = SearchEngine(build_sample_index())

    results = engine.find("good")

    by_url = {r.url: r for r in results}
    assert by_url["https://quotes.toscrape.com/"].term_frequencies["good"] == 2
    assert by_url["https://quotes.toscrape.com/page/2/"].term_frequencies["good"] == 1


def test_find_unique_word_returns_single_result() -> None:
    engine = SearchEngine(build_sample_index())

    results = engine.find("sleepy")

    assert len(results) == 1
    assert results[0].url == "https://quotes.toscrape.com/"


# ---------- Multi-word queries ----------

def test_find_supports_multi_word_queries_with_phrase_bonus() -> None:
    engine = SearchEngine(build_sample_index())

    results = engine.find("good friends")

    assert [result.url for result in results] == [
        "https://quotes.toscrape.com/",
        "https://quotes.toscrape.com/page/2/",
    ]
    assert results[0].phrase_matches == 1
    assert results[0].score > results[1].score


def test_multi_word_query_requires_all_terms() -> None:
    engine = SearchEngine(build_sample_index())

    results = engine.find("good worlds")

    assert results == []


def test_find_is_case_insensitive() -> None:
    engine = SearchEngine(build_sample_index())

    upper = engine.find("GOOD")
    lower = engine.find("good")
    mixed = engine.find("GoOd")

    assert len(upper) == len(lower) == len(mixed)
    assert {r.url for r in upper} == {r.url for r in lower} == {r.url for r in mixed}


# ---------- Edge cases ----------

def test_find_rejects_empty_queries() -> None:
    engine = SearchEngine(build_sample_index())

    with pytest.raises(ValueError, match="at least one searchable word"):
        engine.find("   ")


def test_find_rejects_punctuation_only_query() -> None:
    engine = SearchEngine(build_sample_index())

    with pytest.raises(ValueError, match="at least one searchable word"):
        engine.find("!!! @@@")


def test_find_returns_empty_for_nonexistent_word() -> None:
    engine = SearchEngine(build_sample_index())

    results = engine.find("xyznonexistent")

    assert results == []


def test_print_term_returns_postings() -> None:
    engine = SearchEngine(build_sample_index())

    postings = engine.print_term("good")

    assert "https://quotes.toscrape.com/" in postings
    assert postings["https://quotes.toscrape.com/"]["count"] == 2


def test_print_term_returns_empty_for_missing_word() -> None:
    engine = SearchEngine(build_sample_index())

    assert engine.print_term("xyznonexistent") == {}


# ---------- TF-IDF scoring ----------

def test_tfidf_scores_are_positive() -> None:
    engine = SearchEngine(build_sample_index())

    results = engine.find("good")

    for r in results:
        assert r.tfidf_scores["good"] > 0


def test_tfidf_higher_for_rarer_terms() -> None:
    index = InvertedIndex.build([
        PageData(url="https://a.com/", title="A", text="rare common common common", links=[]),
        PageData(url="https://b.com/", title="B", text="common common common common", links=[]),
    ])
    engine = SearchEngine(index)

    results = engine.find("rare")

    assert len(results) == 1
    assert results[0].tfidf_scores["rare"] > 0


def test_tfidf_formula_correctness() -> None:
    index = InvertedIndex.build([
        PageData(url="https://a.com/", title="A", text="hello world hello", links=[]),
        PageData(url="https://b.com/", title="B", text="world only", links=[]),
    ])
    engine = SearchEngine(index)

    results = engine.find("hello")

    assert len(results) == 1
    r = results[0]
    tf = 2 / 3
    idf = math.log(1 + 2 / (1 + 1))
    expected = round(tf * idf, 6)
    assert r.tfidf_scores["hello"] == expected


def test_phrase_match_boosts_score() -> None:
    index = InvertedIndex.build([
        PageData(url="https://a.com/", title="A", text="good friends are the best friends", links=[]),
        PageData(url="https://b.com/", title="B", text="friends are good and good", links=[]),
    ])
    engine = SearchEngine(index)

    results = engine.find("good friends")

    by_url = {r.url: r for r in results}
    assert by_url["https://a.com/"].phrase_matches == 1
    assert by_url["https://b.com/"].phrase_matches == 0
    assert by_url["https://a.com/"].score > by_url["https://b.com/"].score


def test_results_sorted_by_score_descending() -> None:
    index = InvertedIndex.build([
        PageData(url="https://a.com/", title="A", text="word word word word", links=[]),
        PageData(url="https://b.com/", title="B", text="word other stuff here", links=[]),
        PageData(url="https://c.com/", title="C", text="word word other stuff", links=[]),
    ])
    engine = SearchEngine(index)

    results = engine.find("word")

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_result_contains_tfidf_scores_field() -> None:
    engine = SearchEngine(build_sample_index())

    results = engine.find("books")

    assert len(results) > 0
    for r in results:
        assert isinstance(r.tfidf_scores, dict)
        assert "books" in r.tfidf_scores


# ---------- Phrase matching ----------

def test_no_phrase_match_for_single_term() -> None:
    engine = SearchEngine(build_sample_index())

    results = engine.find("good")

    for r in results:
        assert r.phrase_matches == 0


def test_phrase_match_consecutive_positions() -> None:
    index = InvertedIndex.build([
        PageData(url="https://a.com/", title="A", text="the quick brown fox", links=[]),
    ])
    engine = SearchEngine(index)

    results = engine.find("quick brown")

    assert results[0].phrase_matches == 1


def test_no_phrase_match_when_terms_not_adjacent() -> None:
    index = InvertedIndex.build([
        PageData(url="https://a.com/", title="A", text="quick xxx brown", links=[]),
    ])
    engine = SearchEngine(index)

    results = engine.find("quick brown")

    assert results[0].phrase_matches == 0


# ---------- CLI commands ----------

def test_cli_load_print_and_find_commands(tmp_path, capsys) -> None:
    index = build_sample_index()
    index_file = tmp_path / "index.json"
    index.save(index_file)

    assert main(["load", "--index-file", str(index_file)]) == 0
    load_output = capsys.readouterr()
    assert "Loaded index with" in load_output.out

    assert main(["print", "friends", "--index-file", str(index_file)]) == 0
    print_output = capsys.readouterr()
    assert "Index entry for 'friends':" in print_output.out

    assert main(["find", "good", "friends", "--index-file", str(index_file)]) == 0
    find_output = capsys.readouterr()
    assert "Results for query" in find_output.out


def test_cli_build_command_creates_index_file(tmp_path, monkeypatch, capsys) -> None:
    class StubCrawler:
        def __init__(self, base_url: str, politeness_delay: float, timeout: float) -> None:
            self.base_url = base_url
            self.politeness_delay = politeness_delay
            self.timeout = timeout
            self.errors = {}

        def crawl(self):
            return [
                PageData(
                    url="https://quotes.toscrape.com/",
                    title="Home",
                    text="Curiosity builds knowledge",
                    links=[],
                )
            ]

    monkeypatch.setattr("src.main.WebCrawler", StubCrawler)
    index_file = tmp_path / "built-index.json"

    assert main(["build", "--index-file", str(index_file), "--delay", "6"]) == 0
    output = capsys.readouterr()
    assert "Built index with" in output.out
    assert index_file.exists()


def test_cli_find_handles_missing_query(tmp_path, capsys) -> None:
    index = build_sample_index()
    index_file = tmp_path / "index.json"
    index.save(index_file)

    assert main(["find", "--index-file", str(index_file)]) == 1
    error_output = capsys.readouterr()
    assert "at least one searchable word" in error_output.err


def test_cli_print_shows_no_entry_message(tmp_path, capsys) -> None:
    index = build_sample_index()
    index_file = tmp_path / "index.json"
    index.save(index_file)

    assert main(["print", "xyznonexistent", "--index-file", str(index_file)]) == 0
    output = capsys.readouterr()
    assert "No index entry found" in output.out


def test_cli_print_rejects_empty_term(tmp_path, capsys) -> None:
    index = build_sample_index()
    index_file = tmp_path / "index.json"
    index.save(index_file)

    assert main(["print", "!!!", "--index-file", str(index_file)]) == 1
    output = capsys.readouterr()
    assert "searchable word" in output.err


def test_cli_find_no_results(tmp_path, capsys) -> None:
    index = build_sample_index()
    index_file = tmp_path / "index.json"
    index.save(index_file)

    assert main(["find", "xyznonexistent", "--index-file", str(index_file)]) == 0
    output = capsys.readouterr()
    assert "No pages found" in output.out


def test_cli_find_shows_tfidf_in_output(tmp_path, capsys) -> None:
    index = build_sample_index()
    index_file = tmp_path / "index.json"
    index.save(index_file)

    assert main(["find", "good", "--index-file", str(index_file)]) == 0
    output = capsys.readouterr()
    assert "tfidf=" in output.out


def test_cli_missing_index_file(tmp_path, capsys) -> None:
    result = main(["load", "--index-file", str(tmp_path / "missing.json")])

    assert result == 1
    output = capsys.readouterr()
    assert "Index file not found" in output.err


def test_cli_build_no_pages(tmp_path, monkeypatch, capsys) -> None:
    class EmptyCrawler:
        def __init__(self, base_url: str, politeness_delay: float, timeout: float) -> None:
            self.errors = {}

        def crawl(self):
            return []

    monkeypatch.setattr("src.main.WebCrawler", EmptyCrawler)
    index_file = tmp_path / "empty-index.json"

    assert main(["build", "--index-file", str(index_file)]) == 1
    output = capsys.readouterr()
    assert "No pages were crawled" in output.err


# ---------- Performance / stress ----------

def test_large_index_search_performance() -> None:
    pages = [
        PageData(
            url=f"https://example.com/page/{i}/",
            title=f"Page {i}",
            text=f"common word page{i} unique{i} " + " ".join(f"filler{j}" for j in range(50)),
            links=[],
        )
        for i in range(100)
    ]
    index = InvertedIndex.build(pages)
    engine = SearchEngine(index)

    results = engine.find("common")

    assert len(results) == 100
    assert all(r.score > 0 for r in results)


def test_large_index_build_and_term_count() -> None:
    pages = [
        PageData(
            url=f"https://example.com/{i}/",
            title=f"P{i}",
            text=f"word{i} shared",
            links=[],
        )
        for i in range(50)
    ]
    index = InvertedIndex.build(pages)

    assert index.page_count == 50
    assert index.term_count == 51
    assert len(index.get_postings("shared")) == 50
