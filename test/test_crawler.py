"""Tests for the crawler module."""

from __future__ import annotations

import pytest
import requests

from src.crawler import PageData, WebCrawler


class FakeResponse:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, pages: dict[str, FakeResponse]) -> None:
        self.pages = pages
        self.calls: list[str] = []
        self.headers: dict[str, str] = {}

    def get(self, url: str, timeout: float) -> FakeResponse:
        self.calls.append(url)
        if url not in self.pages:
            raise requests.HTTPError("missing page")
        return self.pages[url]


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


BASE = "https://quotes.toscrape.com/"


def make_crawler(
    pages: dict[str, FakeResponse],
    delay: float = 6.0,
) -> tuple[WebCrawler, FakeSession, FakeClock]:
    session = FakeSession(pages)
    clock = FakeClock()
    crawler = WebCrawler(
        base_url=BASE,
        politeness_delay=delay,
        session=session,
        sleep_func=clock.sleep,
        clock_func=clock.time,
    )
    return crawler, session, clock


# ---------- Basic crawling ----------

def test_crawler_visits_internal_pages_and_respects_politeness_delay() -> None:
    next_page = f"{BASE}page/2/"
    pages = {
        BASE: FakeResponse(
            """
            <html>
              <head><title>Quotes Home</title></head>
              <body>
                <h1>Quotes</h1>
                <a href="/page/2/">Next</a>
                <a href="#top">Top</a>
                <a href="https://example.com/">External</a>
              </body>
            </html>
            """
        ),
        next_page: FakeResponse(
            """
            <html>
              <head><title>Page Two</title></head>
              <body>
                <p>Friendship improves happiness.</p>
                <a href="/">Back</a>
              </body>
            </html>
            """
        ),
    }
    crawler, session, clock = make_crawler(pages)

    result = crawler.crawl()

    assert [p.url for p in result] == [BASE, next_page]
    assert session.calls == [BASE, next_page]
    assert clock.now == 6.0
    assert "https://example.com/" not in {link for p in result for link in p.links}
    assert result[0].title == "Quotes Home"
    assert "Quotes" in result[0].text


def test_crawler_records_errors_and_continues() -> None:
    broken = f"{BASE}broken/"
    pages = {
        BASE: FakeResponse(
            f'<html><body><a href="/broken/">Broken</a><p>Hello world</p></body></html>'
        ),
        broken: FakeResponse("", status_code=500),
    }
    crawler, _, _ = make_crawler(pages, delay=0.0)

    result = crawler.crawl()

    assert [p.url for p in result] == [BASE]
    assert broken in crawler.errors


# ---------- Politeness delay ----------

def test_no_delay_on_first_request() -> None:
    pages = {BASE: FakeResponse("<html><body>Hello</body></html>")}
    crawler, _, clock = make_crawler(pages)

    crawler.crawl()

    assert clock.sleeps == []


def test_delay_applied_between_successive_requests() -> None:
    page2 = f"{BASE}page/2/"
    page3 = f"{BASE}page/3/"
    pages = {
        BASE: FakeResponse(
            '<html><body><a href="/page/2/">2</a><a href="/page/3/">3</a></body></html>'
        ),
        page2: FakeResponse("<html><body>Page 2</body></html>"),
        page3: FakeResponse("<html><body>Page 3</body></html>"),
    }
    crawler, _, clock = make_crawler(pages, delay=6.0)

    crawler.crawl()

    assert len(clock.sleeps) == 2
    assert all(s == 6.0 for s in clock.sleeps)


# ---------- Link extraction ----------

def test_filters_external_links() -> None:
    pages = {
        BASE: FakeResponse(
            """
            <html><body>
              <a href="https://external.com/">External</a>
              <a href="/internal/">Internal</a>
            </body></html>
            """
        ),
        f"{BASE}internal/": FakeResponse("<html><body>Internal</body></html>"),
    }
    crawler, _, _ = make_crawler(pages, delay=0.0)

    result = crawler.crawl()

    all_links = {link for p in result for link in p.links}
    assert "https://external.com/" not in all_links
    assert f"{BASE}internal/" in all_links


def test_removes_fragment_identifiers() -> None:
    pages = {
        BASE: FakeResponse(
            '<html><body><a href="/#section">Fragment</a></body></html>'
        ),
    }
    crawler, session, _ = make_crawler(pages, delay=0.0)

    crawler.crawl()

    assert len(session.calls) == 1


def test_resolves_relative_links() -> None:
    page2 = f"{BASE}page/2/"
    pages = {
        BASE: FakeResponse(
            '<html><body><a href="page/2/">Relative</a></body></html>'
        ),
        page2: FakeResponse("<html><body>Page 2</body></html>"),
    }
    crawler, session, _ = make_crawler(pages, delay=0.0)

    crawler.crawl()

    assert page2 in session.calls


def test_deduplicates_urls() -> None:
    pages = {
        BASE: FakeResponse(
            """
            <html><body>
              <a href="/">Self</a>
              <a href="/">Self again</a>
              <a href="https://quotes.toscrape.com/">Absolute self</a>
            </body></html>
            """
        ),
    }
    crawler, session, _ = make_crawler(pages, delay=0.0)

    crawler.crawl()

    assert session.calls == [BASE]


# ---------- Content extraction ----------

def test_extracts_title_from_title_tag() -> None:
    pages = {
        BASE: FakeResponse("<html><head><title>My Title</title></head><body></body></html>")
    }
    crawler, _, _ = make_crawler(pages, delay=0.0)

    result = crawler.crawl()

    assert result[0].title == "My Title"


def test_falls_back_to_h1_when_no_title() -> None:
    pages = {
        BASE: FakeResponse("<html><body><h1>Heading One</h1></body></html>")
    }
    crawler, _, _ = make_crawler(pages, delay=0.0)

    result = crawler.crawl()

    assert result[0].title == "Heading One"


def test_falls_back_to_url_when_no_title_or_h1() -> None:
    pages = {
        BASE: FakeResponse("<html><body><p>No title here</p></body></html>")
    }
    crawler, _, _ = make_crawler(pages, delay=0.0)

    result = crawler.crawl()

    assert result[0].title == BASE


def test_strips_script_and_style_from_text() -> None:
    pages = {
        BASE: FakeResponse(
            """
            <html><body>
              <script>var x = 1;</script>
              <style>.hidden { display: none; }</style>
              <noscript>Enable JS</noscript>
              <p>Visible content here</p>
            </body></html>
            """
        ),
    }
    crawler, _, _ = make_crawler(pages, delay=0.0)

    result = crawler.crawl()

    assert "var x" not in result[0].text
    assert "display" not in result[0].text
    assert "Enable JS" not in result[0].text
    assert "Visible content here" in result[0].text


def test_collapses_whitespace_in_extracted_text() -> None:
    pages = {
        BASE: FakeResponse(
            "<html><body><p>  Hello   world  \n\n  test  </p></body></html>"
        ),
    }
    crawler, _, _ = make_crawler(pages, delay=0.0)

    result = crawler.crawl()

    assert result[0].text == "Hello world test"


# ---------- Edge cases ----------

def test_empty_body_page() -> None:
    pages = {BASE: FakeResponse("<html><body></body></html>")}
    crawler, _, _ = make_crawler(pages, delay=0.0)

    result = crawler.crawl()

    assert len(result) == 1
    assert result[0].text == ""


def test_page_with_no_links() -> None:
    pages = {
        BASE: FakeResponse("<html><body><p>No links at all</p></body></html>")
    }
    crawler, _, _ = make_crawler(pages, delay=0.0)

    result = crawler.crawl()

    assert result[0].links == []


def test_handles_connection_error_gracefully() -> None:
    pages: dict[str, FakeResponse] = {}
    session = FakeSession(pages)
    clock = FakeClock()
    crawler = WebCrawler(
        base_url=BASE,
        politeness_delay=0.0,
        session=session,
        sleep_func=clock.sleep,
        clock_func=clock.time,
    )

    result = crawler.crawl()

    assert result == []
    assert BASE in crawler.errors


@pytest.mark.parametrize(
    "href",
    [
        "javascript:void(0)",
        "mailto:user@example.com",
        "tel:+1234567890",
        "ftp://files.example.com/",
    ],
    ids=["javascript", "mailto", "tel", "ftp"],
)
def test_ignores_non_http_schemes(href: str) -> None:
    pages = {
        BASE: FakeResponse(f'<html><body><a href="{href}">Link</a></body></html>')
    }
    crawler, _, _ = make_crawler(pages, delay=0.0)

    result = crawler.crawl()

    assert result[0].links == []


def test_handles_unicode_content() -> None:
    pages = {
        BASE: FakeResponse(
            "<html><head><title>名言</title></head>"
            "<body><p>生活是美好的 — life is beautiful</p></body></html>"
        ),
    }
    crawler, _, _ = make_crawler(pages, delay=0.0)

    result = crawler.crawl()

    assert result[0].title == "名言"
    assert "生活是美好的" in result[0].text


def test_normalises_url_case() -> None:
    pages = {
        BASE: FakeResponse(
            '<html><body><a href="https://QUOTES.TOSCRAPE.COM/page/2/">Upper</a></body></html>'
        ),
        f"{BASE}page/2/": FakeResponse("<html><body>Page 2</body></html>"),
    }
    crawler, session, _ = make_crawler(pages, delay=0.0)

    crawler.crawl()

    assert f"{BASE}page/2/" in session.calls
