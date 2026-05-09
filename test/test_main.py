"""Tests for the CLI entry point."""

import os
import tempfile
from unittest.mock import patch, MagicMock
from io import StringIO

import pytest

from src.crawler import CrawlResult
from src.indexer import InvertedIndex
from src.main import run_cli, _handle_build, _handle_load, _handle_print, _handle_find
from src.search import SearchEngine


# ---------------------------------------------------------------------------
# Helper: build a test index on disk
# ---------------------------------------------------------------------------

def _create_test_index(path):
    pages = [
        CrawlResult("http://t.com/1", "Good friends make good days", "Page 1"),
        CrawlResult("http://t.com/2", "A beautiful day in the neighborhood", "Page 2"),
    ]
    idx = InvertedIndex()
    idx.build(pages)
    idx.save(path)
    return idx


# ---------------------------------------------------------------------------
# run_cli — quit / exit
# ---------------------------------------------------------------------------

class TestCliQuit:
    @patch("builtins.input", side_effect=["quit"])
    def test_quit(self, mock_input, capsys):
        run_cli()
        assert "Goodbye!" in capsys.readouterr().out

    @patch("builtins.input", side_effect=["exit"])
    def test_exit(self, mock_input, capsys):
        run_cli()
        assert "Goodbye!" in capsys.readouterr().out

    @patch("builtins.input", side_effect=EOFError)
    def test_eof(self, mock_input, capsys):
        run_cli()
        assert "Goodbye!" in capsys.readouterr().out

    @patch("builtins.input", side_effect=KeyboardInterrupt)
    def test_keyboard_interrupt(self, mock_input, capsys):
        run_cli()
        assert "Goodbye!" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# run_cli — help and unknown
# ---------------------------------------------------------------------------

class TestCliHelp:
    @patch("builtins.input", side_effect=["help", "quit"])
    def test_help(self, mock_input, capsys):
        run_cli()
        out = capsys.readouterr().out
        assert "Available commands:" in out

    @patch("builtins.input", side_effect=["badcmd", "quit"])
    def test_unknown_command(self, mock_input, capsys):
        run_cli()
        out = capsys.readouterr().out
        assert "Unknown command" in out

    @patch("builtins.input", side_effect=["", "quit"])
    def test_empty_input_ignored(self, mock_input, capsys):
        run_cli()
        out = capsys.readouterr().out
        assert "Goodbye!" in out


# ---------------------------------------------------------------------------
# run_cli — print/find before load
# ---------------------------------------------------------------------------

class TestCliNotLoaded:
    @patch("builtins.input", side_effect=["print hello", "quit"])
    def test_print_before_load(self, mock_input, capsys):
        run_cli()
        out = capsys.readouterr().out
        assert "Index not loaded" in out

    @patch("builtins.input", side_effect=["find hello", "quit"])
    def test_find_before_load(self, mock_input, capsys):
        run_cli()
        out = capsys.readouterr().out
        assert "Index not loaded" in out


# ---------------------------------------------------------------------------
# run_cli — load command
# ---------------------------------------------------------------------------

class TestCliLoad:
    @patch("src.main.DEFAULT_INDEX_PATH", "nonexistent_path/index.json")
    @patch("builtins.input", side_effect=["load", "quit"])
    def test_load_missing_file(self, mock_input, capsys):
        run_cli()
        out = capsys.readouterr().out
        assert "not found" in out

    @patch("builtins.input")
    def test_load_success(self, mock_input, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "index.json")
            _create_test_index(path)

            with patch("src.main.DEFAULT_INDEX_PATH", path):
                mock_input.side_effect = ["load", "print good", "quit"]
                run_cli()

            out = capsys.readouterr().out
            assert "Index loaded" in out
            assert "Inverted index" in out


# ---------------------------------------------------------------------------
# run_cli — build command
# ---------------------------------------------------------------------------

class TestCliBuild:
    @patch("src.main.Crawler")
    @patch("builtins.input", side_effect=["build", "quit"])
    def test_build(self, mock_input, MockCrawler, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "index.json")
            with patch("src.main.DEFAULT_INDEX_PATH", path), \
                 patch("src.indexer.DEFAULT_INDEX_PATH", path):
                mock_crawler = MagicMock()
                mock_crawler.crawl.return_value = [
                    CrawlResult("http://t.com/1", "hello world", "T"),
                ]
                MockCrawler.return_value = mock_crawler
                run_cli()

            out = capsys.readouterr().out
            assert "Crawling" in out
            assert "Crawled 1 pages" in out
            assert "Index built" in out


# ---------------------------------------------------------------------------
# run_cli — full session: load → print → find
# ---------------------------------------------------------------------------

class TestCliFullSession:
    @patch("builtins.input")
    def test_load_print_find(self, mock_input, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "index.json")
            _create_test_index(path)

            with patch("src.main.DEFAULT_INDEX_PATH", path):
                mock_input.side_effect = [
                    "load",
                    "print good",
                    "find good",
                    "find neighborhood",
                    "find zzzzz",
                    "find",
                    "quit",
                ]
                run_cli()

            out = capsys.readouterr().out
            assert "Index loaded" in out
            assert "Inverted index" in out
            assert "Results for" in out
            assert "No results found" in out
            assert "Please provide a search query" in out


# ---------------------------------------------------------------------------
# _handle_load — generic exception
# ---------------------------------------------------------------------------

class TestHandleLoadError:
    def test_generic_error(self, capsys):
        idx = InvertedIndex()
        engine = SearchEngine(idx)
        with patch.object(idx, "load", side_effect=RuntimeError("broken")):
            result = _handle_load(idx, engine)
            assert result is False
        out = capsys.readouterr().out
        assert "Error loading index" in out
