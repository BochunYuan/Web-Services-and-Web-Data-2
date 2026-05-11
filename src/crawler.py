"""Web crawler used to build the search index."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import re
import time
from typing import Callable
from urllib.parse import urldefrag, urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup

DEFAULT_TARGET_URL = "https://quotes.toscrape.com/"
DEFAULT_USER_AGENT = "CWKSearchEngine/1.0"


@dataclass(slots=True)
class PageData:
    """Represents the searchable content extracted from a single page."""

    url: str
    title: str
    text: str
    links: list[str]


class WebCrawler:
    """Crawl pages within a single site while respecting a politeness delay."""

    def __init__(
        self,
        base_url: str = DEFAULT_TARGET_URL,
        politeness_delay: float = 6.0,
        session: requests.Session | None = None,
        sleep_func: Callable[[float], None] = time.sleep,
        clock_func: Callable[[], float] = time.monotonic,
        timeout: float = 10.0,
    ) -> None:
        """Initialise the crawler with a target site and politeness settings."""
        self.base_url = self._normalise_url(base_url)
        self.base_netloc = urlsplit(self.base_url).netloc
        self.politeness_delay = politeness_delay
        self.sleep_func = sleep_func
        self.clock_func = clock_func
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", DEFAULT_USER_AGENT)
        self.errors: dict[str, str] = {}
        self._last_request_started_at: float | None = None

    def crawl(self) -> list[PageData]:
        """Breadth-first crawl of all reachable internal pages."""

        queue: deque[str] = deque([self.base_url])
        discovered = {self.base_url}
        pages: list[PageData] = []

        while queue:
            url = queue.popleft()
            try:
                page = self._fetch_page(url)
            except requests.RequestException as exc:
                self.errors[url] = str(exc)
                continue

            pages.append(page)
            for link in page.links:
                if link in discovered:
                    continue
                discovered.add(link)
                queue.append(link)

        return pages

    def _fetch_page(self, url: str) -> PageData:
        """Fetch a single page, parse its content, and return structured data."""
        self._wait_if_needed()
        request_started_at = self.clock_func()
        self._last_request_started_at = request_started_at

        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        return PageData(
            url=url,
            title=self._extract_title(soup, url),
            text=self._extract_text(soup),
            links=sorted(self._extract_links(soup, url)),
        )

    def _wait_if_needed(self) -> None:
        """Sleep if the politeness delay has not yet elapsed since the last request."""
        if self._last_request_started_at is None:
            return

        elapsed = self.clock_func() - self._last_request_started_at
        remaining = self.politeness_delay - elapsed
        if remaining > 0:
            self.sleep_func(remaining)

    def _extract_links(self, soup: BeautifulSoup, current_url: str) -> set[str]:
        """Return all normalised internal links found on the page."""
        links: set[str] = set()
        for anchor in soup.find_all("a", href=True):
            normalised = self._normalise_url(anchor["href"], current_url)
            if normalised and self._is_internal_url(normalised):
                links.add(normalised)
        return links

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract visible text from the page, stripping scripts, styles, and excess whitespace."""
        body = soup.body or soup
        for element in body(["script", "style", "noscript"]):
            element.decompose()

        text = body.get_text(separator=" ", strip=True)
        return re.sub(r"\s+", " ", text).strip()

    def _extract_title(self, soup: BeautifulSoup, url: str) -> str:
        """Return the page title from <title>, <h1>, or the URL as a fallback."""
        if soup.title and soup.title.string:
            return " ".join(soup.title.string.split())

        heading = soup.find("h1")
        if heading:
            return " ".join(heading.get_text(" ", strip=True).split())

        return url

    def _is_internal_url(self, url: str) -> bool:
        """Check whether a URL belongs to the same host as the crawl target."""
        parts = urlsplit(url)
        return parts.scheme in {"http", "https"} and parts.netloc == self.base_netloc

    def _normalise_url(self, url: str, current_url: str | None = None) -> str:
        """Resolve, defragment, and lowercase a URL for consistent deduplication."""
        if current_url:
            url = urljoin(current_url, url)

        url, _fragment = urldefrag(url)
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            return ""

        path = parts.path or "/"
        return urlunsplit(
            (
                parts.scheme.lower(),
                parts.netloc.lower(),
                path,
                parts.query,
                "",
            )
        )

