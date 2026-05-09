"""Web crawler for quotes.toscrape.com.

Implements a BFS-based crawler that discovers and fetches all pages
on the target website, respecting a configurable politeness window
between successive HTTP requests.
"""

import time
import logging
from collections import deque
from typing import Optional
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://quotes.toscrape.com/"
DEFAULT_POLITENESS_DELAY = 6  # seconds between requests


class CrawlResult:
    """Holds the content extracted from a single crawled page."""

    def __init__(self, url: str, text: str, title: str = ""):
        self.url = url
        self.text = text
        self.title = title

    def __repr__(self) -> str:
        return f"CrawlResult(url={self.url!r}, words={len(self.text.split())})"


class Crawler:
    """BFS web crawler with politeness delay and error handling.

    Args:
        base_url: The starting URL to crawl from.
        delay: Minimum seconds to wait between consecutive requests.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        delay: float = DEFAULT_POLITENESS_DELAY,
        timeout: float = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.delay = delay
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "SearchEngineTool/1.0 (University Coursework)"
        })
        self._base_domain = urlparse(self.base_url).netloc

    def crawl(self) -> list[CrawlResult]:
        """Crawl all reachable pages starting from base_url.

        Uses BFS to discover links. Respects the politeness delay between
        successive HTTP requests. Only follows links within the same domain.

        Returns:
            A list of CrawlResult objects, one per successfully crawled page.
        """
        visited: set[str] = set()
        queue: deque[str] = deque()
        results: list[CrawlResult] = []

        start_url = self._normalize_url(self.base_url)
        queue.append(start_url)
        visited.add(start_url)

        last_request_time: float = 0

        while queue:
            url = queue.popleft()

            elapsed = time.time() - last_request_time
            if elapsed < self.delay and last_request_time > 0:
                wait = self.delay - elapsed
                logger.info(f"Politeness delay: waiting {wait:.1f}s")
                time.sleep(wait)

            logger.info(f"Crawling: {url}")
            html = self._fetch_page(url)
            last_request_time = time.time()

            if html is None:
                continue

            soup = BeautifulSoup(html, "lxml")

            text = self._extract_text(soup)
            title = self._extract_title(soup)
            results.append(CrawlResult(url=url, text=text, title=title))
            logger.info(f"  Extracted {len(text.split())} words from {url}")

            for link in self._extract_links(soup, url):
                normalized = self._normalize_url(link)
                if normalized not in visited and self._is_same_domain(normalized):
                    visited.add(normalized)
                    queue.append(normalized)

        logger.info(f"Crawling complete: {len(results)} pages fetched")
        return results

    def _fetch_page(self, url: str) -> Optional[str]:
        """Fetch a single page and return its HTML content.

        Returns None if the request fails for any reason.
        """
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout fetching {url}")
            return None
        except requests.exceptions.ConnectionError:
            logger.warning(f"Connection error fetching {url}")
            return None
        except requests.exceptions.HTTPError as e:
            logger.warning(f"HTTP error {e.response.status_code} for {url}")
            return None
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed for {url}: {e}")
            return None

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract visible text content from an HTML page.

        Removes script and style elements, then extracts text from
        the main content area when possible. Collapses whitespace for
        clean output.
        """
        for tag in soup.find_all(["script", "style", "noscript", "nav", "footer", "form"]):
            tag.decompose()

        content_root = soup.select_one(".container .row .col-md-8")
        if content_root is None:
            content_root = soup.find("body")

        if content_root is None:
            return ""

        for selector in (".pager", ".header-box"):
            for tag in content_root.select(selector):
                tag.decompose()

        text = content_root.get_text(separator=" ", strip=True)
        words = text.split()
        return " ".join(words)

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract the page title."""
        title_tag = soup.find("title")
        return title_tag.get_text(strip=True) if title_tag else ""

    def _extract_links(self, soup: BeautifulSoup, current_url: str) -> list[str]:
        """Extract all valid hyperlinks from the page.

        Resolves relative URLs against the current page URL.
        Filters out fragment-only links and non-HTTP schemes.
        """
        links = []
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"].strip()

            if href.startswith(("#", "mailto:", "javascript:", "tel:")):
                continue

            absolute_url = urljoin(current_url, href)

            parsed = urlparse(absolute_url)
            if parsed.scheme not in ("http", "https"):
                continue

            clean_url = urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                "",  # strip params
                "",  # strip query
                "",  # strip fragment
            ))

            links.append(clean_url)

        return links

    def _normalize_url(self, url: str) -> str:
        """Normalize a URL for consistent deduplication.

        Ensures trailing slash on paths, strips query strings and fragments.
        """
        parsed = urlparse(url)
        path = parsed.path
        if not path:
            path = "/"
        elif not path.endswith("/") and "." not in path.split("/")[-1]:
            path += "/"

        return urlunparse((
            parsed.scheme,
            parsed.netloc,
            path,
            "",
            "",
            "",
        ))

    def _is_same_domain(self, url: str) -> bool:
        """Check whether a URL belongs to the same domain as base_url."""
        return urlparse(url).netloc == self._base_domain
