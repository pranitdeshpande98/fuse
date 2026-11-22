"""Web search and page reading, with no API key.

Search uses DuckDuckGo through the `ddgs` package. It is free and needs no account, but it
is unofficial and can rate-limit; the SearchProvider interface lets a keyed service (Brave,
Tavily) replace it without touching the tools.

Reading a page is where the risk is, so `read_web_page` is careful:

- SSRF: Fuse runs on your machine, next to your database and router. A model (or a page
  telling it what to do) could ask it to "read" http://127.0.0.1:55432 or your router's
  admin page. Every URL, including each redirect, must resolve to public internet
  addresses only; private, loopback and link-local ones are refused.
- Size and type: at most 2 MB, HTML or plain text only, and a 15-second timeout.
- Prompt injection: page text is wrapped as untrusted content, like email.

Known limit: the address is checked, then httpx resolves it again to connect. A hostile
DNS server could answer differently the second time ("DNS rebinding"). Closing that gap
needs connecting to the checked IP directly; for a personal agent, the check above plus
the approval gate on every action is a reasonable line.
"""

import asyncio
import ipaddress
import logging
import socket
from collections.abc import Awaitable, Callable
from typing import Protocol
from urllib.parse import urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import BaseTool, tool

from fuse.google.tools import wrap_untrusted

log = logging.getLogger("fuse.web")

MAX_BYTES = 2_000_000
MAX_PAGE_CHARS = 8000  # what the model gets from one page
MAX_REDIRECTS = 3
TIMEOUT_SECONDS = 15
USER_AGENT = "Mozilla/5.0 (compatible; FusePersonalAgent/1.0)"
NOISE_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg"]


class WebError(Exception):
    """A readable reason a search or page read didn't work."""


class SearchProvider(Protocol):
    async def search(self, query: str, max_results: int) -> list[dict[str, str]]: ...


class DuckDuckGoSearch:
    async def search(self, query: str, max_results: int) -> list[dict[str, str]]:
        from ddgs import DDGS

        def run() -> list[dict]:
            return DDGS().text(query, max_results=max_results, safesearch="moderate")

        try:
            rows = await asyncio.to_thread(run)
        except Exception as exc:
            raise WebError(f"search failed ({type(exc).__name__})") from exc
        return [
            {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
            for r in rows
        ]


Resolver = Callable[[str], Awaitable[list[str]]]


async def resolve(host: str) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return sorted({info[4][0] for info in infos})


async def check_public_url(url: str, resolver: Resolver = resolve) -> None:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise WebError("only http and https links can be read")
    if not parts.hostname:
        raise WebError("that isn't a complete link")
    if parts.username or parts.password:
        raise WebError("links with a username or password aren't allowed")
    try:
        addresses = await resolver(parts.hostname)
    except OSError as exc:
        raise WebError(f"couldn't find {parts.hostname}") from exc
    if not addresses:
        raise WebError(f"couldn't find {parts.hostname}")
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%")[0])
        if not ip.is_global:  # loopback, private (10.x, 192.168.x), link-local, reserved...
            raise WebError("that address is on a private network, so I won't open it")


def page_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(NOISE_TAGS):
        tag.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    body = soup.body or soup
    lines = [" ".join(line.split()) for line in body.get_text("\n").splitlines()]
    text = "\n".join(line for line in lines if line)
    return f"{title}\n\n{text}" if title else text


class PageReader:
    def __init__(
        self,
        transport: httpx.AsyncBaseTransport | None = None,
        resolver: Resolver = resolve,
    ) -> None:
        self._transport, self._resolver = transport, resolver

    async def read(self, url: str) -> str:
        async with httpx.AsyncClient(
            transport=self._transport,
            follow_redirects=False,  # each redirect is checked like the first link
            timeout=TIMEOUT_SECONDS,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            for _ in range(MAX_REDIRECTS + 1):
                await check_public_url(url, self._resolver)
                try:
                    async with client.stream("GET", url) as response:
                        if response.is_redirect and response.headers.get("location"):
                            url = urljoin(url, response.headers["location"])
                            continue
                        return await self._text(response)
                except httpx.HTTPError as exc:
                    raise WebError(f"couldn't load the page ({type(exc).__name__})") from exc
        raise WebError("too many redirects")

    async def _text(self, response: httpx.Response) -> str:
        if response.status_code >= 400:
            raise WebError(f"the site answered {response.status_code}")
        kind = response.headers.get("content-type", "").split(";")[0].strip().lower()
        if kind not in ("text/html", "text/plain", "application/xhtml+xml", ""):
            raise WebError(f"that's not a web page ({kind}); I can only read HTML or text")
        body = bytearray()
        async for chunk in response.aiter_bytes():
            body += chunk
            if len(body) > MAX_BYTES:
                raise WebError("the page is too large to read")
        text = body.decode(response.encoding or "utf-8", errors="replace")
        return text if kind == "text/plain" else page_text(text)


def make_web_tools(
    search: SearchProvider | None = None, reader: PageReader | None = None
) -> list[BaseTool]:
    search = search or DuckDuckGoSearch()
    reader = reader or PageReader()

    @tool
    async def web_search(query: str, max_results: int = 5) -> str:
        """Search the web. Returns titles, links and short snippets. Use read_web_page to
        read a result in full before relying on it."""
        try:
            results = await search.search(query, max(1, min(max_results, 10)))
        except WebError as exc:
            return f"Web search didn't work: {exc}"
        if not results:
            return "No results."
        lines = [
            f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}" for i, r in enumerate(results, 1)
        ]
        return wrap_untrusted("\n".join(lines))

    @tool
    async def read_web_page(url: str) -> str:
        """Read a public web page as text (HTML or plain text, up to about 8,000 characters)."""
        try:
            text = await reader.read(url)
        except WebError as exc:
            return f"Couldn't read that page: {exc}"
        if len(text) > MAX_PAGE_CHARS:
            text = text[:MAX_PAGE_CHARS] + "\n[...cut: the page is longer]"
        return wrap_untrusted(f"Source: {url}\n\n{text}")

    return [web_search, read_web_page]
