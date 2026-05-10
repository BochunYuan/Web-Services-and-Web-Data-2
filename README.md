# Search Engine Tool Coursework

## Project Overview

This project implements a small command-line search engine for the coursework website target `https://quotes.toscrape.com/`. It crawls the site, builds an inverted index containing word frequencies and positions, saves the index to disk, and allows users to inspect or search the stored index.

## Project Structure

```text
src/
  crawler.py
  indexer.py
  search.py
  main.py
tests/
  test_crawler.py
  test_indexer.py
  test_search.py
data/
  .gitkeep
requirements.txt
README.md
```

## Installation

1. Create and activate a virtual environment if desired.
2. Install the dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Usage

All commands are run through the CLI entry point:

```bash
python3 -m src.main <command> [options]
```

### `build`

Crawls the target website, respects the politeness delay, builds the inverted index, and stores it in `data/index.json` by default.

```bash
python3 -m src.main build
```

Optional arguments:

```bash
python3 -m src.main build --index-file data/custom-index.json --delay 6 --timeout 10
```

### `load`

Loads an existing compiled index file and reports a short summary.

```bash
python3 -m src.main load
```

### `print`

Prints the postings list for a specific word, including frequencies and positions.

```bash
python3 -m src.main print nonsense
```

### `find`

Searches for one or more words and returns pages that contain all query terms.

```bash
python3 -m src.main find indifference
python3 -m src.main find good friends
```

## Testing

Run the test suite with:

```bash
pytest
```

The tests cover:

- crawler link traversal and politeness behaviour
- inverted index tokenisation, counts, and persistence
- single-word and multi-word search behaviour
- command-line `load`, `print`, and `find` workflows

## Implementation Notes

- Search is case-insensitive.
- The inverted index stores word counts and token positions for each page.
- Multi-word search uses AND semantics and adds a small ranking bonus for exact phrase matches.
- The `build` command defaults to a 6-second politeness delay, matching the coursework brief.
