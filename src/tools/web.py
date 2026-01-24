"""Web search tool using DuckDuckGo or SearXNG."""

import logging
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)


class WebSearchTool:
    """
    Web search using DuckDuckGo or self-hosted SearXNG.

    Prioritizes SearXNG if configured, falls back to DuckDuckGo.
    """

    def __init__(self) -> None:
        """Initialize the web search tool."""
        self.searxng_url = getattr(settings, "searxng_url", None)
        self._backend: str | None = None

    def _detect_backend(self) -> str | None:
        """Detect which search backend is available."""
        if self._backend:
            return self._backend

        # Try SearXNG first if configured
        if self.searxng_url:
            try:
                import httpx
                response = httpx.get(
                    f"{self.searxng_url}/config",
                    timeout=2.0,
                )
                if response.status_code == 200:
                    self._backend = "searxng"
                    logger.info("Detected SearXNG backend")
                    return "searxng"
            except Exception:
                pass

        # Try DuckDuckGo
        try:
            from ddgs import DDGS
            self._backend = "duckduckgo"
            logger.info("Using DuckDuckGo backend")
            return "duckduckgo"
        except ImportError:
            logger.warning("ddgs package not installed")

        logger.warning("No web search backend available")
        return None

    def check_availability(self) -> bool:
        """Check if any search backend is available."""
        return self._detect_backend() is not None

    def search(
        self,
        query: str,
        max_results: int = 5,
        region: str = "wt-wt",
    ) -> list[dict[str, Any]]:
        """
        Perform a web search.

        Args:
            query: Search query
            max_results: Maximum number of results to return
            region: Region code for search (default: worldwide)

        Returns:
            List of search results with 'title', 'url', 'snippet'
        """
        backend = self._detect_backend()

        if not backend:
            logger.error("No search backend available")
            return []

        try:
            if backend == "searxng":
                return self._search_searxng(query, max_results)
            else:
                return self._search_duckduckgo(query, max_results, region)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def _search_duckduckgo(
        self,
        query: str,
        max_results: int,
        region: str,
    ) -> list[dict[str, Any]]:
        """Search using DuckDuckGo."""
        from ddgs import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, region=region, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("link", "")),
                    "snippet": r.get("body", r.get("snippet", "")),
                })

        logger.info(f"DuckDuckGo returned {len(results)} results for: {query}")
        return results

    def _search_searxng(
        self,
        query: str,
        max_results: int,
    ) -> list[dict[str, Any]]:
        """Search using SearXNG."""
        import httpx

        params = {
            "q": query,
            "format": "json",
            "categories": "general",
        }

        response = httpx.get(
            f"{self.searxng_url}/search",
            params=params,
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for r in data.get("results", [])[:max_results]:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
            })

        logger.info(f"SearXNG returned {len(results)} results for: {query}")
        return results

    def search_news(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search for news articles.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of news results
        """
        backend = self._detect_backend()

        if backend == "duckduckgo":
            try:
                from ddgs import DDGS

                results = []
                with DDGS() as ddgs:
                    for r in ddgs.news(query, max_results=max_results):
                        results.append({
                            "title": r.get("title", ""),
                            "url": r.get("url", r.get("link", "")),
                            "snippet": r.get("body", ""),
                            "date": r.get("date", ""),
                            "source": r.get("source", ""),
                        })

                logger.info(f"DuckDuckGo news returned {len(results)} results")
                return results
            except Exception as e:
                logger.error(f"News search failed: {e}")
                return []

        elif backend == "searxng":
            # SearXNG news category
            try:
                import httpx

                params = {
                    "q": query,
                    "format": "json",
                    "categories": "news",
                }

                response = httpx.get(
                    f"{self.searxng_url}/search",
                    params=params,
                    timeout=10.0,
                )
                response.raise_for_status()
                data = response.json()

                results = []
                for r in data.get("results", [])[:max_results]:
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("content", ""),
                        "date": r.get("publishedDate", ""),
                        "source": r.get("engine", ""),
                    })

                return results
            except Exception as e:
                logger.error(f"SearXNG news search failed: {e}")
                return []

        return []

    def fetch_page_content(
        self,
        url: str,
        max_length: int = 5000,
    ) -> str | None:
        """
        Fetch and extract main content from a webpage.

        Args:
            url: URL to fetch
            max_length: Maximum content length to return

        Returns:
            Extracted text content or None if failed
        """
        try:
            import httpx

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
            }

            response = httpx.get(url, headers=headers, timeout=10.0, follow_redirects=True)
            response.raise_for_status()

            # Try to extract text content
            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type.lower():
                return None

            html = response.text

            # Try BeautifulSoup if available
            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(html, "html.parser")

                # Remove script and style elements
                for script in soup(["script", "style", "nav", "header", "footer"]):
                    script.decompose()

                # Get text
                text = soup.get_text(separator="\n", strip=True)

                # Clean up whitespace
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                text = "\n".join(lines)

                if len(text) > max_length:
                    text = text[:max_length] + "..."

                return text

            except ImportError:
                # Basic HTML stripping without BeautifulSoup
                import re
                text = re.sub(r"<[^>]+>", " ", html)
                text = re.sub(r"\s+", " ", text).strip()

                if len(text) > max_length:
                    text = text[:max_length] + "..."

                return text

        except Exception as e:
            logger.error(f"Failed to fetch page content: {e}")
            return None
