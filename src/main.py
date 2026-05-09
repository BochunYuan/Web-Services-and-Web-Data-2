"""CLI entry point for the search engine tool.

Provides an interactive shell supporting build, load, print, and find
commands for crawling, indexing, and searching quotes.toscrape.com.
"""

import sys
import logging
import time

from src.crawler import Crawler, BASE_URL
from src.indexer import InvertedIndex, DEFAULT_INDEX_PATH
from src.search import SearchEngine

logger = logging.getLogger(__name__)

HELP_TEXT = """
Available commands:
  build             Crawl the website and build the inverted index
  load              Load a previously saved index from file
  print <word>      Show the inverted index entry for a word
  find <query>      Find pages containing the search term(s)
  help              Show this help message
  quit / exit       Exit the search tool
""".strip()


def run_cli() -> None:
    """Run the interactive command-line shell."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    index = InvertedIndex()
    engine = SearchEngine(index)
    index_ready = False

    print("=" * 56)
    print("  Search Engine Tool — quotes.toscrape.com")
    print("=" * 56)
    print(f'Type "help" for available commands.\n')

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        parts = user_input.split(maxsplit=1)
        command = parts[0].lower()
        argument = parts[1] if len(parts) > 1 else ""

        if command in ("quit", "exit"):
            print("Goodbye!")
            break

        elif command == "help":
            print(HELP_TEXT)

        elif command == "build":
            _handle_build(index, engine)
            index_ready = True

        elif command == "load":
            success = _handle_load(index, engine)
            if success:
                index_ready = True

        elif command == "print":
            if not index_ready:
                print('Index not loaded. Run "build" or "load" first.')
                continue
            _handle_print(engine, argument)

        elif command == "find":
            if not index_ready:
                print('Index not loaded. Run "build" or "load" first.')
                continue
            _handle_find(engine, argument)

        else:
            print(f'Unknown command: "{command}". Type "help" for usage.')


def _handle_build(index: InvertedIndex, engine: SearchEngine) -> None:
    """Execute the build command: crawl, index, and save."""
    print(f"Crawling {BASE_URL} ...")
    start = time.time()

    crawler = Crawler()
    pages = crawler.crawl()

    elapsed_crawl = time.time() - start
    print(f"Crawled {len(pages)} pages in {elapsed_crawl:.0f} seconds.\n")

    print("Building inverted index...")
    index.build(pages)
    print(
        f"Index built: {index.total_words} unique words "
        f"across {index.total_docs} pages.\n"
    )

    index.save(DEFAULT_INDEX_PATH)
    print(f"Index saved to {DEFAULT_INDEX_PATH}")


def _handle_load(index: InvertedIndex, engine: SearchEngine) -> bool:
    """Execute the load command: read index from file."""
    try:
        index.load(DEFAULT_INDEX_PATH)
        print(
            f"Index loaded: {index.total_words} unique words, "
            f"{index.total_docs} pages."
        )
        return True
    except FileNotFoundError:
        print(
            f'Index file not found at "{DEFAULT_INDEX_PATH}". '
            f'Run "build" first to create it.'
        )
        return False
    except Exception as e:
        print(f"Error loading index: {e}")
        return False


def _handle_print(engine: SearchEngine, argument: str) -> None:
    """Execute the print command: display index entry for a word."""
    print(engine.print_word(argument))


def _handle_find(engine: SearchEngine, argument: str) -> None:
    """Execute the find command: search for pages matching a query."""
    results = engine.find(argument)
    print(engine.format_results(argument, results))


if __name__ == "__main__":
    run_cli()
