"""Command-line interface for the coursework search engine."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    from .crawler import DEFAULT_TARGET_URL, WebCrawler
    from .indexer import InvertedIndex, tokenize
    from .search import SearchEngine
except ImportError:  # pragma: no cover - allows direct execution from src/.
    from crawler import DEFAULT_TARGET_URL, WebCrawler
    from indexer import InvertedIndex, tokenize
    from search import SearchEngine

DEFAULT_INDEX_FILE = Path("data/index.json")


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser with build, load, print, and find subcommands."""
    parser = argparse.ArgumentParser(description="Search engine coursework tool.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Crawl the website and build the index.")
    build_parser.add_argument("--url", default=DEFAULT_TARGET_URL, help="Website to crawl.")
    build_parser.add_argument(
        "--index-file",
        default=str(DEFAULT_INDEX_FILE),
        help="Where to store the compiled index file.",
    )
    build_parser.add_argument(
        "--delay",
        type=float,
        default=6.0,
        help="Seconds to wait between successive HTTP requests.",
    )
    build_parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout for each request in seconds.",
    )
    build_parser.set_defaults(handler=handle_build)

    load_parser = subparsers.add_parser("load", help="Load a previously built index from disk.")
    load_parser.add_argument(
        "--index-file",
        default=str(DEFAULT_INDEX_FILE),
        help="Path to the compiled index file.",
    )
    load_parser.set_defaults(handler=handle_load)

    print_parser = subparsers.add_parser("print", help="Print the postings list for one word.")
    print_parser.add_argument("term", nargs="?", help="Word to inspect in the inverted index.")
    print_parser.add_argument(
        "--index-file",
        default=str(DEFAULT_INDEX_FILE),
        help="Path to the compiled index file.",
    )
    print_parser.set_defaults(handler=handle_print)

    find_parser = subparsers.add_parser("find", help="Find pages containing the query terms.")
    find_parser.add_argument("query", nargs="*", help="One or more words to search for.")
    find_parser.add_argument(
        "--index-file",
        default=str(DEFAULT_INDEX_FILE),
        help="Path to the compiled index file.",
    )
    find_parser.set_defaults(handler=handle_find)

    return parser


def handle_build(args: argparse.Namespace) -> int:
    """Crawl the target website, build the inverted index, and save it to disk."""
    crawler = WebCrawler(
        base_url=args.url,
        politeness_delay=args.delay,
        timeout=args.timeout,
    )
    pages = crawler.crawl()
    if not pages:
        print("No pages were crawled, so no index file was created.", file=sys.stderr)
        return 1

    index = InvertedIndex.build(pages)
    index.save(args.index_file)

    print(
        f"Built index with {index.term_count} terms across {index.page_count} pages "
        f"and saved it to {args.index_file}."
    )
    if crawler.errors:
        print(f"Skipped {len(crawler.errors)} pages because of HTTP errors.")
    return 0


def handle_load(args: argparse.Namespace) -> int:
    """Load a previously built index from disk and print a summary."""
    index = InvertedIndex.load(args.index_file)
    print(
        f"Loaded index with {index.term_count} terms across {index.page_count} pages "
        f"from {args.index_file}."
    )
    return 0


def handle_print(args: argparse.Namespace) -> int:
    """Print the inverted index entry for a single word."""
    if not args.term or not tokenize(args.term):
        print("Please provide a searchable word for the print command.", file=sys.stderr)
        return 1

    index = InvertedIndex.load(args.index_file)
    search_engine = SearchEngine(index)
    term = tokenize(args.term)[0]
    postings = search_engine.print_term(args.term)

    if not postings:
        print(f"No index entry found for '{term}'.")
        return 0

    print(f"Index entry for '{term}':")
    for url in sorted(postings):
        page_info = index.pages.get(url, {})
        stats = postings[url]
        print(
            f"- {url} | title={page_info.get('title', url)} | "
            f"count={stats['count']} | positions={stats['positions']}"
        )
    return 0


def handle_find(args: argparse.Namespace) -> int:
    """Search the index for query terms and display TF-IDF ranked results."""
    query_text = " ".join(args.query)
    index = InvertedIndex.load(args.index_file)
    search_engine = SearchEngine(index)

    try:
        results = search_engine.find(query_text)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not results:
        print(f"No pages found for query: {query_text!r}")
        return 0

    print(f"Results for query: {query_text!r}")
    for result in results:
        term_summary = ", ".join(
            f"{term}={count}" for term, count in result.term_frequencies.items()
        )
        tfidf_summary = ", ".join(
            f"{term}={score:.4f}" for term, score in result.tfidf_scores.items()
        )
        extras = f"tfidf=[{tfidf_summary}]"
        if result.phrase_matches:
            extras = f"{extras}, phrase_matches={result.phrase_matches}"
        print(
            f"- {result.url} | title={result.title} | score={result.score:.4f} | {term_summary} | {extras}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the appropriate command handler."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return args.handler(args)
    except FileNotFoundError:
        print(
            f"Index file not found: {args.index_file}. Run the build command first.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

