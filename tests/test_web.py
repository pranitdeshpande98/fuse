import httpx
import pytest

from fuse.tools.web import PageReader, WebError, check_public_url, make_web_tools, page_text

PUBLIC = "93.184.216.34"


def resolver_for(mapping):
    async def resolve(host):
        if host not in mapping:
            raise OSError("not found")
        return mapping[host]

    return resolve


HTML = """<html><head><title>Acme careers</title><script>var x=1</script></head>
<body><nav>Menu Home About</nav><h1>Software Engineer II</h1>
<p>Remote, US.   Apply by   Dec 1.</p><footer>(c) Acme</footer></body></html>"""


def test_page_text_keeps_content_and_drops_noise():
    text = page_text(HTML)
    assert text.startswith("Acme careers")
    assert "Software Engineer II" in text and "Remote, US. Apply by Dec 1." in text
    assert "var x" not in text and "Menu" not in text and "(c) Acme" not in text


@pytest.mark.parametrize(
    "url,addresses,reason",
    [
        ("http://localhost:55432/", ["127.0.0.1"], "private network"),
        ("http://router.lan/admin", ["192.168.1.1"], "private network"),
        ("http://metadata/latest", ["169.254.169.254"], "private network"),
        ("http://v6/", ["::1"], "private network"),
        ("file:///etc/passwd", [], "only http"),
        ("http://user:pw@example.com/", [PUBLIC], "username"),
    ],
)
async def test_private_and_odd_urls_are_refused(url, addresses, reason):
    host = httpx.URL(url).host or "none"
    with pytest.raises(WebError, match=reason):
        await check_public_url(url, resolver_for({host: addresses}))


async def test_a_redirect_to_a_private_address_is_refused():
    def handler(request):
        if request.url.host == "short.link":
            return httpx.Response(302, headers={"location": "http://127.0.0.1:55432/"})
        return httpx.Response(200, text="secret")

    reader = PageReader(
        httpx.MockTransport(handler),
        resolver_for({"short.link": [PUBLIC], "127.0.0.1": ["127.0.0.1"]}),
    )
    with pytest.raises(WebError, match="private network"):
        await reader.read("http://short.link/x")


async def test_reads_html_follows_a_safe_redirect_and_refuses_pdfs():
    def handler(request):
        if request.url.path == "/old":
            return httpx.Response(301, headers={"location": "/jobs"})
        if request.url.path == "/file.pdf":
            return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF")
        return httpx.Response(200, headers={"content-type": "text/html"}, text=HTML)

    reader = PageReader(httpx.MockTransport(handler), resolver_for({"acme.com": [PUBLIC]}))
    assert "Software Engineer II" in await reader.read("https://acme.com/old")
    with pytest.raises(WebError, match="not a web page"):
        await reader.read("https://acme.com/file.pdf")


async def test_huge_pages_are_refused():
    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"x" * 3_000_000)

    reader = PageReader(httpx.MockTransport(handler), resolver_for({"big.com": [PUBLIC]}))
    with pytest.raises(WebError, match="too large"):
        await reader.read("https://big.com/")


class FakeSearch:
    def __init__(self, results=None, error=None):
        self.results, self.error = results or [], error

    async def search(self, query, max_results):
        if self.error:
            raise self.error
        return self.results[:max_results]


async def test_search_results_are_wrapped_as_untrusted():
    results = [{"title": "Acme jobs", "url": "https://acme.com/jobs", "snippet": "SWE II"}]
    tools = {t.name: t for t in make_web_tools(FakeSearch(results), PageReader())}
    out = await tools["web_search"].ainvoke({"query": "acme jobs"})
    assert out.startswith("<untrusted_content>") and "https://acme.com/jobs" in out
    failing = {t.name: t for t in make_web_tools(FakeSearch(error=WebError("rate limited")))}
    assert "rate limited" in await failing["web_search"].ainvoke({"query": "x"})


async def test_read_tool_reports_problems_instead_of_crashing():
    reader = PageReader(resolver=resolver_for({"localhost": ["127.0.0.1"]}))
    tools = {t.name: t for t in make_web_tools(FakeSearch(), reader)}
    out = await tools["read_web_page"].ainvoke({"url": "http://localhost:55432"})
    assert out.startswith("Couldn't read that page") and "private network" in out
