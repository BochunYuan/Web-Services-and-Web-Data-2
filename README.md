# Search Engine Tool Coursework

## Project Overview

This project implements a command-line search engine targeting `https://quotes.toscrape.com/`. It crawls every reachable page on the site, builds an inverted index storing word frequencies and positions, persists the index to disk as JSON, and lets users inspect or query the index with TF-IDF ranked results.

## Architecture

```
┌──────────┐     ┌──────────┐     ┌──────────────┐
│  Crawler │────>│ Indexer  │────>│ SearchEngine │
│  (BFS)   │     │ (build)  │     │ (TF-IDF)     │
└──────────┘     └────┬─────┘     └──────┬───────┘
                      │ save/load        │ find/print
                      v                  v
                 index.json          CLI output
```

- **Crawler** (`crawler.py`): Performs a breadth-first traversal starting from the base URL. Each page is fetched via `requests`, parsed with `BeautifulSoup`, and all internal links are queued for crawling. A configurable politeness delay (default 6 seconds) is enforced between successive HTTP requests. External links, fragment-only anchors, and non-HTTP schemes are filtered out. URLs are normalised (lowercased, fragments stripped) to avoid duplicate visits.

- **Indexer** (`indexer.py`): Tokenises extracted page text into lowercase words using a regex pattern that preserves contractions (e.g. `don't`). For each token the indexer records its frequency and every position within the page, producing an inverted index of the form `{word: {url: {count, positions}}}`. Page-level metadata (title, total word count) is stored alongside.

- **Search Engine** (`search.py`): Queries the inverted index using **TF-IDF** scoring. Term Frequency is normalised by the page's total word count (`tf = count / word_count`), and Inverse Document Frequency uses a smoothed logarithm (`idf = log(1 + N / (1 + df))`). Multi-word queries require all terms to appear (AND semantics) and receive a bonus for consecutive phrase matches. Results are sorted by descending TF-IDF score.

- **CLI** (`main.py`): Provides the `build`, `load`, `print`, and `find` subcommands via `argparse`, delegating to the modules above.

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| BFS crawl order | Visits pages closest to the root first, ensuring the most linked (and likely most relevant) pages are indexed early. |
| JSON index format | Human-readable, easy to inspect and debug; sufficient for a site of ~200 pages. |
| Positions stored per word | Enables exact phrase matching without re-parsing pages at query time. |
| TF-IDF over raw frequency | Raw frequency biases towards longer pages. TF-IDF normalises for document length and penalises ubiquitous words. |
| Smoothed IDF (`1 + N / (1 + df)`) | Avoids division by zero and reduces extreme IDF values for very rare terms. |
| AND semantics for multi-word queries | Ensures all returned pages are relevant to every query term, improving precision. |

### Complexity

| Operation | Time Complexity |
|-----------|----------------|
| Crawl | O(P) page fetches, where P = number of pages (dominated by network I/O and politeness delay) |
| Index build | O(W) where W = total words across all pages |
| Save / Load | O(S) where S = index file size |
| Single-word search | O(D) where D = number of pages containing the term |
| Multi-word search | O(T × D) where T = number of query terms, D = candidate pages |
| Phrase matching | O(D × L) where L = positions per term per page |

## Project Structure

```text
src/
  crawler.py          # Web crawler with BFS traversal and politeness delay
  indexer.py          # Inverted index construction and JSON persistence
  search.py           # TF-IDF search engine with phrase matching
  main.py             # CLI entry point (build, load, print, find)
  __init__.py
tests/
  conftest.py         # Pytest path configuration
  test_crawler.py     # 22 tests — link extraction, politeness, edge cases
  test_indexer.py     # 24 tests — tokenisation, index build, save/load
  test_search.py      # 31 tests — TF-IDF scoring, phrase matching, CLI
data/
  index.json          # Compiled inverted index (214 pages, 4654 terms)
requirements.txt
README.md
```

## Installation

1. Create and activate a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install the dependencies:

```bash
python3 -m pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `requests` | HTTP requests for crawling |
| `beautifulsoup4` | HTML parsing and text extraction |
| `pytest` | Test framework |
| `pytest-cov` | Coverage reporting (optional) |

## Usage

All commands are run through the CLI entry point:

```bash
python3 -m src.main <command> [options]
```

### `build`

Crawls the target website, respects the 6-second politeness delay, builds the inverted index with TF-IDF statistics, and saves it to `data/index.json`.

```bash
python3 -m src.main build
```

Optional arguments:

```bash
python3 -m src.main build --url https://quotes.toscrape.com/ --delay 6 --timeout 10 --index-file data/index.json
```

Example output:

```
Built index with 4654 terms across 214 pages and saved it to data/index.json.
```

### `load`

Loads an existing compiled index file and reports a summary.

```bash
python3 -m src.main load
```

Example output:

```
Loaded index with 4654 terms across 214 pages from data/index.json.
```

### `print`

Prints the inverted index entry for a specific word, showing every page it appears on with frequency and positions.

```bash
python3 -m src.main print nonsense
```

Example output:

```
Index entry for 'nonsense':
- https://quotes.toscrape.com/page/2/ | title=Quotes to Scrape | count=1 | positions=[414]
- https://quotes.toscrape.com/page/7/ | title=Quotes to Scrape | count=1 | positions=[318]
- https://quotes.toscrape.com/tag/fantasy/page/1/ | title=Quotes to Scrape | count=1 | positions=[9]
```

### `find`

Searches for one or more words. Returns all pages containing every query term, ranked by TF-IDF score. Consecutive phrase matches receive a scoring bonus.

```bash
python3 -m src.main find indifference
python3 -m src.main find good friends
```

Example output:

```
Results for query: 'good friends'
- https://quotes.toscrape.com/tag/contentment/page/1/ | title=Quotes to Scrape | score=1.1213 | good=2, friends=3 | tfidf=[good=0.0733, friends=0.0479], phrase_matches=1
- https://quotes.toscrape.com/tag/friends/ | title=Quotes to Scrape | score=1.0472 | good=3, friends=11 | tfidf=[good=0.0182, friends=0.0290], phrase_matches=1
```

Edge cases are handled gracefully:

```bash
python3 -m src.main find xyznotexist     # → "No pages found for query: 'xyznotexist'"
python3 -m src.main find                  # → "Query must contain at least one searchable word."
```

## Testing

Run the full test suite:

```bash
pytest
```

Run with coverage report:

```bash
pytest --cov=src --cov-report=term-missing
```

### Test Summary

| Module | Tests | Coverage |
|--------|-------|----------|
| `test_crawler.py` | 22 | `crawler.py` 100% |
| `test_indexer.py` | 24 | `indexer.py` 100% |
| `test_search.py` | 31 | `search.py` 100%, `main.py` 98% |
| **Total** | **77** | **99%** |

### Testing Strategy

- **Unit tests**: Each module (crawler, indexer, search) is tested independently using fake HTTP sessions and in-memory data to avoid network dependencies.
- **Edge cases**: Empty pages, Unicode content, special characters, non-HTTP schemes (`javascript:`, `mailto:`), missing index files, punctuation-only queries.
- **Parametrised tests**: `pytest.mark.parametrize` covers multiple input variations (case sensitivity, non-HTTP link schemes) without code duplication.
- **TF-IDF correctness**: Verifies the scoring formula mathematically — `tf × idf` values are computed independently and compared against engine output.
- **Phrase matching**: Tests consecutive and non-consecutive term positions to verify exact phrase detection.
- **CLI integration**: Every subcommand path is tested including success, error, and missing-file scenarios.
- **Performance**: Stress tests build and search a 100-page index to verify the engine handles larger datasets without issues.
