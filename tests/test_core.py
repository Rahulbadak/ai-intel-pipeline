import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from src.dates import extract_date_from_html, is_fresh, parse_date
from src.entities import EntityResolver
from src.llm import AllTiersFailed, LLMEngine, Ungrounded, is_grounded, smart_truncate
from src.pricing import infer_pricing
from src.sources.papers import find_github, parse_arxiv
from src.sources.signals import extract_article_text, parse_feed, role_family
from src.store import SeenStore
from src.util import canonical_url

NOW = datetime(2026, 11, 23, 12, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------- entities
@pytest.mark.parametrize("raw", ["OpenAI", "OpenAI, Inc.", "Open AI", "openai", "OPENAI INC", "Open-AI"])
def test_openai_variants(raw):
    assert EntityResolver().resolve(raw).canonical == "OpenAI"


def test_aliases_and_no_false_positive():
    r = EntityResolver()
    assert r.resolve("Mistral").canonical == "Mistral AI"
    assert r.resolve("Weights and Biases").canonical == "Weights & Biases"
    assert r.resolve("Hugging Face, Inc.").canonical == "Hugging Face"
    assert r.resolve("Coherent").canonical != "Cohere"
    assert r.resolve("Adapt").canonical != "Adept AI"


def test_dynamic_dedupe_and_log():
    r = EntityResolver()
    a = r.resolve("Acme Robotics Inc", "STARTUP", "http://x")
    b = r.resolve("AcmeRobotics", "STARTUP", "http://y")
    assert a.canonical == b.canonical == "Acme Robotics"
    assert len(r.log) == 2 and r.log[0]["method"] == "new"


# ---------------------------------------------------------------- dates
def test_relative_dates():
    assert parse_date("2 hours ago", NOW).dt == NOW - timedelta(hours=2)
    assert parse_date("Posted 3h ago", NOW).dt == NOW - timedelta(hours=3)
    assert parse_date("an hour ago", NOW).dt == NOW - timedelta(hours=1)
    assert parse_date("45 minutes ago", NOW).dt == NOW - timedelta(minutes=45)
    assert parse_date("just now", NOW).dt == NOW
    assert parse_date("3 days ago", NOW).dt == NOW - timedelta(days=3)


def test_absolute_dates_and_epoch():
    assert parse_date("Mon, 23 Nov 2026 09:00:00 GMT", NOW).dt == datetime(2026, 11, 23, 9, tzinfo=timezone.utc)
    assert parse_date("2026-11-23T08:12:03", NOW).dt.hour == 8  # naive -> UTC
    assert parse_date("2026-11-23T14:30:00+05:30", NOW).dt.hour == 9
    assert parse_date(int(NOW.timestamp()), NOW).dt == NOW
    assert parse_date(int(NOW.timestamp() * 1000), NOW).dt == NOW
    assert parse_date("garbage", NOW) is None and parse_date(None, NOW) is None


def test_freshness_boundaries():
    assert is_fresh(parse_date("23 hours ago", NOW), NOW)
    assert not is_fresh(parse_date("25 hours ago", NOW), NOW)
    assert not is_fresh(parse_date("2026-11-22", NOW), NOW)  # date-only yesterday: unprovable
    assert is_fresh(parse_date("2026-11-23", NOW), NOW)
    assert not is_fresh(None, NOW)


def test_html_date_extraction_priority():
    html = ('<html><head><meta property="article:modified_time" content="2026-11-23T11:00:00Z">'
            '<meta property="article:published_time" content="2026-11-22T20:00:00Z"></head></html>')
    pd = extract_date_from_html(html, NOW)
    assert pd.dt.day == 22 and pd.method.startswith("meta")  # published, not modified
    ld = '<script type="application/ld+json">{"@graph":[{"datePublished":"2026-11-23T10:00:00Z"}]}</script>'
    assert extract_date_from_html(ld, NOW).method == "json-ld"
    assert extract_date_from_html("<p>Published 5 hours ago by X</p>", NOW).dt == NOW - timedelta(hours=5)
    assert extract_date_from_html("<p>nothing</p>", NOW) is None


# ---------------------------------------------------------------- truncation + grounding
def test_smart_truncate_keeps_lead_and_budget():
    paras = ["LEAD paragraph about Acme raising $50 million in 2026."] + \
            ["filler filler filler filler filler filler"] * 200 + \
            ["Acme CEO Jane Doe said revenue grew 300% to $12M in Q3 2026."]
    text = "\n".join(paras)
    out = smart_truncate(text, 400)
    assert len(out) <= 400
    assert out.startswith("LEAD") and "300%" in out
    assert smart_truncate("short", 400) == "short"


def test_grounding():
    src = "Acme Robotics, Inc. launched Foo on Nov 3."
    assert is_grounded("acme robotics", src) and not is_grounded("Globex", src)


# ---------------------------------------------------------------- LLM engine with fake transport
def ok(content):
    return 200, {}, {"choices": [{"message": {"content": json.dumps(content)}}]}


def make_engine(script, tiers=None):
    calls = []

    async def transport(tier, payload):
        calls.append((tier["name"], len(payload["messages"][1]["content"])))
        return script(tier["name"], len(calls))

    async def nosleep(_):
        pass

    tiers = tiers or [{"name": n, "url": "", "model": "m", "api_key": "k", "max_chars": 4000} for n in ("t1", "t2", "t3")]
    return LLMEngine(tiers=tiers, transport=transport, sleep=nosleep), calls


def test_429_backoff_then_success():
    eng, calls = make_engine(lambda t, n: (429, {"retry-after": "1"}, {}) if n < 3 else ok({"a": 1}))
    data, tier = asyncio.run(eng.complete_json("sys", "text"))
    assert data == {"a": 1} and tier == "t1" and eng.stats["429"] == 2


def test_long_retry_after_falls_through_to_next_tier():
    eng, calls = make_engine(lambda t, n: (429, {"retry-after": "600"}, {}) if t == "t1" else ok({"a": 1}))
    data, tier = asyncio.run(eng.complete_json("sys", "text"))
    assert tier == "t2" and eng.stats["fallbacks"] == 1


def test_413_rechunks_then_succeeds():
    eng, calls = make_engine(lambda t, n: (413, {}, {}) if calls[-1][1] > 1500 else ok({"a": 1}))
    big = "\n".join(f"Paragraph number {i} with several distinct words and 2026 figures." for i in range(400))
    data, tier = asyncio.run(eng.complete_json("sys", big))
    assert tier == "t1" and calls[-1][1] <= 1500 and eng.stats["413"] >= 1


def test_context_400_treated_like_413():
    eng, calls = make_engine(lambda t, n: (400, {}, {"error": "maximum context length exceeded"}) if calls[-1][1] > 1500 else ok({"a": 1}))
    big = "\n".join(f"Paragraph {i} with several distinct words and 2026 figures." for i in range(400))
    assert asyncio.run(eng.complete_json("sys", big))[1] == "t1"


def test_all_tiers_fail():
    eng, _ = make_engine(lambda t, n: (500, {}, {}))
    eng.max_retries = 1
    with pytest.raises(AllTiersFailed):
        asyncio.run(eng.complete_json("sys", "text"))


def test_extract_rejects_hallucination():
    src = "Acme Robotics launched a free plan for developers."
    eng, _ = make_engine(lambda t, n: ok({"companies": ["Globex"]}))
    with pytest.raises(Ungrounded):
        asyncio.run(eng.extract("x", src, ["companies"], grounded=("companies",)))
    eng2, _ = make_engine(lambda t, n: ok({"companies": ["Acme Robotics"]}))
    assert asyncio.run(eng2.extract("x", src, ["companies"], grounded=("companies",)))[0]["companies"] == ["Acme Robotics"]
    eng3, _ = make_engine(lambda t, n: ok({"pricingModel": "FREE", "pricingModel_evidence": "totally made up"}))
    with pytest.raises(Ungrounded):
        asyncio.run(eng3.extract("x", src, ["pricingModel"], evidence=("pricingModel",)))


# ---------------------------------------------------------------- pricing / parsers / store
def test_pricing():
    assert infer_pricing("Start for free. Pricing: Pro $20/month")[0] == "FREEMIUM"
    assert infer_pricing("Plans from $49 per seat")[0] == "PAID"
    assert infer_pricing("Contact sales for a custom quote. Book a demo.")[0] == "ENTERPRISE"
    assert infer_pricing("We are open-source and free")[0] == "FREE"
    assert infer_pricing("We build things")[0] is None


ARXIV = """<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
<entry><id>http://arxiv.org/abs/2601.01234v2</id><published>2026-01-05T10:00:00Z</published>
<title>A  Great
 Paper</title><summary>Code at https://github.com/foo/bar-baz.</summary>
<author><name>Ann A</name></author><author><name>Bob B</name></author></entry></feed>"""


def test_arxiv_and_github_parsing():
    e = parse_arxiv(ARXIV)[0]
    assert e["arxiv_id"] == "2601.01234" and e["title"] == "A Great Paper"
    assert e["authors"] == ["Ann A", "Bob B"] and e["github"] == ("foo", "bar-baz")
    assert find_github("see github.com/sponsors/x and github.com/a/b.git") == ("a", "b")
    assert find_github("no link") is None


RSS = """<rss xmlns:content="http://purl.org/rss/1.0/modules/content/" xmlns:dc="http://purl.org/dc/elements/1.1/"><channel>
<item><title>Big AI news</title><link>https://ex.com/a?utm_source=x</link><pubDate>Mon, 23 Nov 2026 09:00:00 GMT</pubDate>
<content:encoded><![CDATA[<p>Hello <b>world</b></p>]]></content:encoded></item>
<item><title>No date</title><link>https://ex.com/b</link></item></channel></rss>"""


def test_feed_and_article_parsing():
    items = parse_feed(RSS)
    assert len(items) == 2 and items[0]["body"] == "Hello world" and items[1]["date_raw"] is None
    html = "<nav>menu</nav><article><p>" + "Real article sentence number one is long enough. " * 3 + "</p></article>"
    assert "Real article" in extract_article_text(html) and "menu" not in extract_article_text(html)
    assert role_family("Senior ML Engineer") == "Engineering" and role_family("Product Manager, AI") == "Product"


def test_store_exactly_once(tmp_path):
    s = SeenStore(tmp_path / "s.db")
    assert not s.source_known("src")
    assert s.claim("k1", "src") is True and s.claim("k1", "src") is False
    assert s.source_known("src")
    assert canonical_url("https://www.Ex.com/a/?utm_source=x&id=1#frag") == "https://ex.com/a?id=1"
