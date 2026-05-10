"""Tests for the inverted index module."""

from __future__ import annotations

import json

import pytest

from src.crawler import PageData
from src.indexer import InvertedIndex, tokenize


# ---------- Tokenizer ----------

def test_tokenize_normalises_case_and_punctuation() -> None:
    assert tokenize("Good friends, GOOD habits.") == [
        "good",
        "friends",
        "good",
        "habits",
    ]


def test_tokenize_empty_string() -> None:
    assert tokenize("") == []


def test_tokenize_whitespace_only() -> None:
    assert tokenize("   \t\n  ") == []


def test_tokenize_preserves_numbers() -> None:
    assert tokenize("chapter 42 section 7") == ["chapter", "42", "section", "7"]


def test_tokenize_handles_contractions() -> None:
    assert tokenize("don't won't it's") == ["don't", "won't", "it's"]


def test_tokenize_strips_special_characters() -> None:
    result = tokenize("hello! @world# $python% ^code&")
    assert result == ["hello", "world", "python", "code"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("HELLO", ["hello"]),
        ("hello", ["hello"]),
        ("HeLLo WoRLD", ["hello", "world"]),
    ],
    ids=["all-upper", "all-lower", "mixed-case"],
)
def test_tokenize_case_insensitive(text: str, expected: list[str]) -> None:
    assert tokenize(text) == expected


# ---------- Index building ----------

def test_inverted_index_tracks_counts_and_positions() -> None:
    pages = [
        PageData(
            url="https://quotes.toscrape.com/",
            title="Home",
            text="Good friends show good humour",
            links=[],
        ),
        PageData(
            url="https://quotes.toscrape.com/page/2/",
            title="Page Two",
            text="Friends value honesty",
            links=[],
        ),
    ]

    index = InvertedIndex.build(pages)

    assert index.page_count == 2
    assert index.get_postings("good") == {
        "https://quotes.toscrape.com/": {"count": 2, "positions": [0, 3]}
    }
    assert index.get_postings("friends") == {
        "https://quotes.toscrape.com/": {"count": 1, "positions": [1]},
        "https://quotes.toscrape.com/page/2/": {"count": 1, "positions": [0]},
    }


def test_build_from_empty_page_list() -> None:
    index = InvertedIndex.build([])

    assert index.page_count == 0
    assert index.term_count == 0


def test_build_from_single_page() -> None:
    page = PageData(url="https://example.com/", title="Test", text="hello world", links=[])
    index = InvertedIndex.build([page])

    assert index.page_count == 1
    assert index.term_count == 2
    assert index.get_postings("hello") == {
        "https://example.com/": {"count": 1, "positions": [0]}
    }


def test_build_from_page_with_empty_text() -> None:
    page = PageData(url="https://example.com/", title="Empty", text="", links=[])
    index = InvertedIndex.build([page])

    assert index.page_count == 1
    assert index.term_count == 0


def test_page_stores_word_count() -> None:
    page = PageData(
        url="https://example.com/",
        title="Test",
        text="one two three four five",
        links=[],
    )
    index = InvertedIndex.build([page])

    assert index.pages["https://example.com/"]["word_count"] == 5


def test_index_handles_repeated_words() -> None:
    page = PageData(
        url="https://example.com/",
        title="Repeat",
        text="hello hello hello",
        links=[],
    )
    index = InvertedIndex.build([page])

    postings = index.get_postings("hello")
    assert postings["https://example.com/"]["count"] == 3
    assert postings["https://example.com/"]["positions"] == [0, 1, 2]


def test_get_postings_returns_empty_for_missing_term() -> None:
    index = InvertedIndex.build([
        PageData(url="https://example.com/", title="T", text="hello", links=[])
    ])

    assert index.get_postings("nonexistent") == {}


def test_get_postings_handles_non_word_input() -> None:
    index = InvertedIndex.build([
        PageData(url="https://example.com/", title="T", text="hello", links=[])
    ])

    assert index.get_postings("!!!") == {}


# ---------- Save / Load ----------

def test_index_can_be_saved_and_loaded(tmp_path) -> None:
    index = InvertedIndex.build(
        [
            PageData(
                url="https://quotes.toscrape.com/",
                title="Home",
                text="Stay curious",
                links=[],
            )
        ]
    )
    index_file = tmp_path / "index.json"

    index.save(index_file)
    loaded = InvertedIndex.load(index_file)

    assert loaded.pages == index.pages
    assert loaded.index == index.index


def test_save_creates_parent_directories(tmp_path) -> None:
    index = InvertedIndex.build([
        PageData(url="https://example.com/", title="T", text="hello", links=[])
    ])
    deep_path = tmp_path / "a" / "b" / "c" / "index.json"

    index.save(deep_path)

    assert deep_path.exists()
    loaded = InvertedIndex.load(deep_path)
    assert loaded.term_count == 1


def test_saved_index_is_valid_json(tmp_path) -> None:
    index = InvertedIndex.build([
        PageData(url="https://example.com/", title="T", text="hello world", links=[])
    ])
    index_file = tmp_path / "index.json"
    index.save(index_file)

    data = json.loads(index_file.read_text(encoding="utf-8"))
    assert "index" in data
    assert "pages" in data


def test_load_nonexistent_file_raises_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        InvertedIndex.load(tmp_path / "missing.json")


def test_roundtrip_preserves_all_data(tmp_path) -> None:
    pages = [
        PageData(url=f"https://example.com/{i}/", title=f"Page {i}", text=f"word{i} common", links=[])
        for i in range(5)
    ]
    index = InvertedIndex.build(pages)
    index_file = tmp_path / "index.json"

    index.save(index_file)
    loaded = InvertedIndex.load(index_file)

    assert loaded.page_count == index.page_count
    assert loaded.term_count == index.term_count
    for term in index.index:
        assert loaded.get_postings(term) == index.get_postings(term)


# ---------- Properties ----------

def test_page_count_and_term_count() -> None:
    pages = [
        PageData(url="https://example.com/1/", title="A", text="alpha beta", links=[]),
        PageData(url="https://example.com/2/", title="B", text="beta gamma", links=[]),
        PageData(url="https://example.com/3/", title="C", text="gamma delta", links=[]),
    ]
    index = InvertedIndex.build(pages)

    assert index.page_count == 3
    assert index.term_count == 4


def test_to_dict_structure() -> None:
    index = InvertedIndex.build([
        PageData(url="https://example.com/", title="T", text="hello", links=[])
    ])

    d = index.to_dict()
    assert set(d.keys()) == {"index", "pages"}
    assert "hello" in d["index"]
    assert "https://example.com/" in d["pages"]
