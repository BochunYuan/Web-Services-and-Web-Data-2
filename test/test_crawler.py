"""Tests for the web crawler module."""

import time
from unittest.mock import patch, MagicMock

import pytest
import requests
from bs4 import BeautifulSoup

from src.crawler import Crawler, CrawlResult, BASE_URL


# ---------------------------------------------------------------------------
# CrawlResult
# ---------------------------------------------------------------------------

class TestCrawlResult:
    def test_attributes(self):
        cr = CrawlResult("http://example.com", "hello world", "Title")
        assert cr.url == "http://example.com"
        assert cr.text == "hello world"
        assert cr.title == "Title"

    def test_default_title(self):
        cr = CrawlResult("http://example.com", "text")
        assert cr.title == ""

    def test_repr(self):
        cr = CrawlResult("http://example.com", "one two three")
        assert "words=3" in repr(cr)
        assert "example.com" in repr(cr)


# ---------------------------------------------------------------------------
# Crawler initialisation
# ---------------------------------------------------------------------------

class TestCrawlerInit:
    def test_defaults(self):
        c = Crawler()
        assert c.base_url == BASE_URL.rstrip("/")
        assert c.delay == 6
        assert c.timeout == 30

    def test_custom_params(self):
        c = Crawler(base_url="http://test.com/", delay=1, timeout=5)
        assert c.base_url == "http://test.com"
        assert c.delay == 1
        assert c.timeout == 5

    def test_trailing_slash_stripped(self):
        c = Crawler(base_url="http://test.com///")
        assert c.base_url == "http://test.com"


# ---------------------------------------------------------------------------
# _fetch_page
# ---------------------------------------------------------------------------

class TestFetchPage:
    def setup_method(self):
        self.crawler = Crawler(delay=0)

    @patch.object(requests.Session, "get")
    def test_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = "<html><body>Hello</body></html>"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = self.crawler._fetch_page("http://example.com")
        assert result == "<html><body>Hello</body></html>"
        mock_get.assert_called_once_with("http://example.com", timeout=30)

    @patch.object(requests.Session, "get", side_effect=requests.exceptions.Timeout)
    def test_timeout(self, mock_get):
        assert self.crawler._fetch_page("http://example.com") is None

    @patch.object(requests.Session, "get", side_effect=requests.exceptions.ConnectionError)
    def test_connection_error(self, mock_get):
        assert self.crawler._fetch_page("http://example.com") is None

    @patch.object(requests.Session, "get")
    def test_http_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_resp
        )
        mock_get.return_value = mock_resp
        assert self.crawler._fetch_page("http://example.com") is None

    @patch.object(requests.Session, "get", side_effect=requests.exceptions.RequestException("fail"))
    def test_generic_request_error(self, mock_get):
        assert self.crawler._fetch_page("http://example.com") is None


# ---------------------------------------------------------------------------
# _extract_text
# ---------------------------------------------------------------------------

class TestExtractText:
    def setup_method(self):
        self.crawler = Crawler(delay=0)

    def _soup(self, html):
        return BeautifulSoup(html, "lxml")

    def test_basic_text(self):
        html = "<html><body><p>Hello World</p></body></html>"
        text = self.crawler._extract_text(self._soup(html))
        assert "Hello World" in text

    def test_strips_script_and_style(self):
        html = """<html><body>
            <script>var x=1;</script>
            <style>.a{color:red}</style>
            <noscript>No JS</noscript>
            <p>Keep this</p>
        </body></html>"""
        text = self.crawler._extract_text(self._soup(html))
        assert "Keep this" in text
        assert "var x" not in text
        assert "color" not in text
        assert "No JS" not in text

    def test_prefers_main_content_container(self):
        html = """<html><body>
            <nav>Navigation text</nav>
            <div class="container">
              <div class="row">
                <div class="col-md-8">
                  <p>Main content stays</p>
                  <ul class="pager"><li>Next</li></ul>
                </div>
              </div>
            </div>
            <footer>Footer text</footer>
        </body></html>"""
        text = self.crawler._extract_text(self._soup(html))
        assert "Main content stays" in text
        assert "Navigation text" not in text
        assert "Footer text" not in text
        assert "Next" not in text

    def test_strips_form_content(self):
        html = """<html><body>
            <form><input value="secret" />Form text</form>
            <p>Visible text</p>
        </body></html>"""
        text = self.crawler._extract_text(self._soup(html))
        assert "Visible text" in text
        assert "Form text" not in text

    def test_collapses_whitespace(self):
        html = "<html><body><p>  lots   of    space  </p></body></html>"
        text = self.crawler._extract_text(self._soup(html))
        assert "  " not in text

    def test_no_body(self):
        html = "<html><head><title>T</title></head></html>"
        text = self.crawler._extract_text(self._soup(html))
        assert text == ""

    def test_empty_body(self):
        html = "<html><body></body></html>"
        text = self.crawler._extract_text(self._soup(html))
        assert text == ""


# ---------------------------------------------------------------------------
# _extract_title
# ---------------------------------------------------------------------------

class TestExtractTitle:
    def setup_method(self):
        self.crawler = Crawler(delay=0)

    def _soup(self, html):
        return BeautifulSoup(html, "lxml")

    def test_with_title(self):
        html = "<html><head><title>My Page</title></head><body></body></html>"
        assert self.crawler._extract_title(self._soup(html)) == "My Page"

    def test_without_title(self):
        html = "<html><head></head><body></body></html>"
        assert self.crawler._extract_title(self._soup(html)) == ""

    def test_title_whitespace(self):
        html = "<html><head><title>  Spaced  </title></head><body></body></html>"
        assert self.crawler._extract_title(self._soup(html)) == "Spaced"


# ---------------------------------------------------------------------------
# _extract_links
# ---------------------------------------------------------------------------

class TestExtractLinks:
    def setup_method(self):
        self.crawler = Crawler(delay=0)

    def _soup(self, html):
        return BeautifulSoup(html, "lxml")

    def test_absolute_link(self):
        html = '<html><body><a href="http://example.com/page">link</a></body></html>'
        links = self.crawler._extract_links(self._soup(html), "http://example.com/")
        assert "http://example.com/page" in links

    def test_relative_link(self):
        html = '<html><body><a href="/page2/">link</a></body></html>'
        links = self.crawler._extract_links(self._soup(html), "http://example.com/page1/")
        assert "http://example.com/page2/" in links

    def test_filters_fragment(self):
        html = '<html><body><a href="#section">skip</a></body></html>'
        links = self.crawler._extract_links(self._soup(html), "http://example.com/")
        assert len(links) == 0

    def test_filters_mailto(self):
        html = '<html><body><a href="mailto:a@b.com">mail</a></body></html>'
        links = self.crawler._extract_links(self._soup(html), "http://example.com/")
        assert len(links) == 0

    def test_filters_javascript(self):
        html = '<html><body><a href="javascript:void(0)">js</a></body></html>'
        links = self.crawler._extract_links(self._soup(html), "http://example.com/")
        assert len(links) == 0

    def test_filters_tel(self):
        html = '<html><body><a href="tel:123456">call</a></body></html>'
        links = self.crawler._extract_links(self._soup(html), "http://example.com/")
        assert len(links) == 0

    def test_filters_non_http_scheme(self):
        html = '<html><body><a href="ftp://example.com/file">ftp</a></body></html>'
        links = self.crawler._extract_links(self._soup(html), "http://example.com/")
        assert links == []

    def test_strips_query_and_fragment(self):
        html = '<html><body><a href="/page?q=1#top">link</a></body></html>'
        links = self.crawler._extract_links(self._soup(html), "http://example.com/")
        assert links[0] == "http://example.com/page"

    def test_no_links(self):
        html = "<html><body><p>No links here</p></body></html>"
        links = self.crawler._extract_links(self._soup(html), "http://example.com/")
        assert links == []

    def test_multiple_links(self):
        html = """<html><body>
            <a href="/a">A</a>
            <a href="/b">B</a>
            <a href="/c">C</a>
        </body></html>"""
        links = self.crawler._extract_links(self._soup(html), "http://example.com/")
        assert len(links) == 3


# ---------------------------------------------------------------------------
# _normalize_url
# ---------------------------------------------------------------------------

class TestNormalizeUrl:
    def setup_method(self):
        self.crawler = Crawler(delay=0)

    def test_adds_trailing_slash(self):
        assert self.crawler._normalize_url("http://example.com/page") == "http://example.com/page/"

    def test_keeps_existing_slash(self):
        assert self.crawler._normalize_url("http://example.com/page/") == "http://example.com/page/"

    def test_no_slash_for_file_extension(self):
        result = self.crawler._normalize_url("http://example.com/file.html")
        assert not result.endswith(".html/")

    def test_strips_query_and_fragment(self):
        result = self.crawler._normalize_url("http://example.com/page?q=1#sec")
        assert "?" not in result
        assert "#" not in result


# ---------------------------------------------------------------------------
# _is_same_domain
# ---------------------------------------------------------------------------

class TestIsSameDomain:
    def test_same_domain(self):
        c = Crawler(base_url="http://example.com/", delay=0)
        assert c._is_same_domain("http://example.com/page/") is True

    def test_different_domain(self):
        c = Crawler(base_url="http://example.com/", delay=0)
        assert c._is_same_domain("http://other.com/page/") is False

    def test_subdomain_is_different(self):
        c = Crawler(base_url="http://example.com/", delay=0)
        assert c._is_same_domain("http://sub.example.com/") is False


# ---------------------------------------------------------------------------
# crawl() integration with mocked HTTP
# ---------------------------------------------------------------------------

class TestCrawlIntegration:
    PAGE1_HTML = """<html>
    <head><title>Page 1</title></head>
    <body>
        <p>Hello world</p>
        <a href="/page2/">Next</a>
        <a href="http://external.com/">External</a>
    </body></html>"""

    PAGE2_HTML = """<html>
    <head><title>Page 2</title></head>
    <body>
        <p>Goodbye world</p>
        <a href="/">Back</a>
    </body></html>"""

    def _mock_get(self, url, **kwargs):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        if "/page2/" in url:
            mock_resp.text = self.PAGE2_HTML
        else:
            mock_resp.text = self.PAGE1_HTML
        return mock_resp

    @patch.object(requests.Session, "get")
    def test_crawl_discovers_two_pages(self, mock_get):
        mock_get.side_effect = self._mock_get
        c = Crawler(base_url="http://example.com/", delay=0)
        results = c.crawl()

        assert len(results) == 2
        urls = {r.url for r in results}
        assert "http://example.com/" in urls
        assert "http://example.com/page2/" in urls

    @patch.object(requests.Session, "get")
    def test_crawl_does_not_follow_external(self, mock_get):
        mock_get.side_effect = self._mock_get
        c = Crawler(base_url="http://example.com/", delay=0)
        results = c.crawl()

        urls = {r.url for r in results}
        assert "http://external.com/" not in urls

    @patch.object(requests.Session, "get")
    def test_crawl_no_duplicates(self, mock_get):
        mock_get.side_effect = self._mock_get
        c = Crawler(base_url="http://example.com/", delay=0)
        results = c.crawl()

        urls = [r.url for r in results]
        assert len(urls) == len(set(urls))

    @patch.object(requests.Session, "get")
    def test_crawl_extracts_titles(self, mock_get):
        mock_get.side_effect = self._mock_get
        c = Crawler(base_url="http://example.com/", delay=0)
        results = c.crawl()

        titles = {r.title for r in results}
        assert "Page 1" in titles
        assert "Page 2" in titles

    @patch.object(requests.Session, "get")
    def test_crawl_skips_failed_pages(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError
        c = Crawler(base_url="http://example.com/", delay=0)
        results = c.crawl()
        assert results == []

    @patch.object(requests.Session, "get")
    @patch("src.crawler.time.sleep")
    def test_politeness_delay(self, mock_sleep, mock_get):
        """Verify time.sleep is called when requests are faster than delay."""
        mock_resp = MagicMock()
        mock_resp.text = "<html><body>No links</body></html>"
        mock_resp.raise_for_status = MagicMock()

        call_count = 0
        def side_effect(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                mock_resp_with_link = MagicMock()
                mock_resp_with_link.text = '<html><body><a href="/p2/">x</a></body></html>'
                mock_resp_with_link.raise_for_status = MagicMock()
                return mock_resp_with_link
            return mock_resp

        mock_get.side_effect = side_effect
        c = Crawler(base_url="http://example.com/", delay=10)
        c.crawl()

        assert mock_sleep.called
