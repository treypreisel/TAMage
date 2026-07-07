#!/usr/bin/env python3
"""Lane A: deterministic site scrape for the TAMage demo dataset.

Reads the curated TAM CSV, fetches each company's homepage plus up to 4
discovered subpages (about/product/pricing/careers, found by anchor text),
and appends one JSON line per company to dataset/scraped.jsonl.

Resumable: already-scraped ids are skipped on restart. Polite: honors
robots.txt, identifies itself, caps request rate via bounded concurrency.

Usage:
  python dataset/scrape.py                 # full run
  python dataset/scrape.py --limit 30      # pilot
"""

import argparse
import asyncio
import csv
import json
import os
import re
import time
import urllib.robotparser
from datetime import datetime, timezone

import aiohttp
from bs4 import BeautifulSoup

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(HERE, "exports", "tam_main.csv")
OUT_PATH = os.path.join(HERE, "scraped.jsonl")

UA = "TAMage-research/0.1 (open-source GTM research; +https://github.com/treypreisel/tamage)"
CONCURRENCY = 80
PAGE_TIMEOUT = aiohttp.ClientTimeout(total=12)
MAX_HTML_BYTES = 2_000_000
HOME_TEXT_CAP = 15_000
SUB_TEXT_CAP = 8_000

SUBPAGE_PATTERNS = {
    "about": re.compile(r"\babout\b|\bcompany\b|who we are", re.I),
    "product": re.compile(r"\bproducts?\b|\bplatform\b|\bsolutions?\b|\bfeatures\b", re.I),
    "pricing": re.compile(r"\bpricing\b|\bplans\b", re.I),
    "careers": re.compile(r"\bcareers?\b|\bjobs\b|join (us|our team)|we'?re hiring", re.I),
}

TECH_HINTS = {
    "google_analytics": re.compile(r"gtag\(|googletagmanager|google-analytics", re.I),
    "hubspot": re.compile(r"hubspot|hs-scripts", re.I),
    "segment": re.compile(r"cdn\.segment\.com|analytics\.js", re.I),
    "intercom": re.compile(r"intercom", re.I),
    "drift": re.compile(r"drift\.com|driftt", re.I),
    "marketo": re.compile(r"marketo", re.I),
    "stripe": re.compile(r"js\.stripe\.com", re.I),
    "calendly": re.compile(r"calendly", re.I),
    "shopify": re.compile(r"shopify", re.I),
    "wordpress": re.compile(r"wp-content|wp-includes", re.I),
    "webflow": re.compile(r"webflow", re.I),
    "wix": re.compile(r"wix\.com|wixstatic", re.I),
    "squarespace": re.compile(r"squarespace", re.I),
    "nextjs": re.compile(r"/_next/", re.I),
    "react": re.compile(r"react(-dom)?(\.production)?\.min\.js|data-reactroot", re.I),
}


def visible_text(soup, cap):
    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    return text[:cap]


async def fetch(session, url):
    """GET a page. Returns (final_url, html) or (None, error_string)."""
    try:
        async with session.get(url, timeout=PAGE_TIMEOUT, allow_redirects=True) as r:
            if r.status != 200:
                return None, f"http {r.status}"
            ctype = r.headers.get("Content-Type", "")
            if "html" not in ctype and ctype:
                return None, f"non-html ({ctype.split(';')[0]})"
            raw = await r.content.read(MAX_HTML_BYTES)
            return str(r.url), raw.decode(r.charset or "utf-8", errors="replace")
    except asyncio.TimeoutError:
        return None, "timeout"
    except aiohttp.ClientError as e:
        return None, f"client error: {type(e).__name__}"
    except Exception as e:
        return None, f"error: {type(e).__name__}"


async def robots_allows(session, base):
    try:
        _, body = await fetch(session, base + "/robots.txt")
        if body is None or body.startswith(("http ", "non-html", "timeout", "client", "error")):
            return True  # no readable robots.txt -> allowed
        rp = urllib.robotparser.RobotFileParser()
        rp.parse(body.splitlines())
        return rp.can_fetch(UA, base + "/")
    except Exception:
        return True


async def scrape_company(session, row):
    domain = (row.get("website") or "").strip().lower()
    rec = {"id": row["id"], "domain": domain, "fetched_at":
           datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if not domain:
        rec.update(ok=False, error="no domain")
        return rec

    html, final_url, err = None, None, None
    for candidate in (f"https://{domain}", f"https://www.{domain}", f"http://{domain}"):
        base = candidate.rsplit("/", 1)[0] if candidate.count("/") > 2 else candidate
        if not await robots_allows(session, base):
            rec.update(ok=False, error="robots disallow")
            return rec
        final_url, html = await fetch(session, candidate)
        if final_url:
            break
        err = html  # fetch returns error string in second slot
        html = None
    if html is None:
        rec.update(ok=False, error=err or "unreachable")
        return rec

    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else None
    meta = soup.find("meta", attrs={"name": "description"}) or \
        soup.find("meta", attrs={"property": "og:description"})
    rec.update(
        ok=True, final_url=final_url, title=title,
        meta_description=(meta.get("content", "").strip() or None) if meta else None,
        tech_hints=sorted(k for k, rx in TECH_HINTS.items() if rx.search(html)),
    )

    # discover subpages by anchor text/href, same site only, first match per kind
    from urllib.parse import urljoin, urlparse
    host = urlparse(final_url).netloc.removeprefix("www.")
    found = {}
    for a in soup.find_all("a", href=True):
        try:
            label = f"{a.get_text(' ', strip=True)} {a['href']}"[:200]
            target = urljoin(final_url, a["href"].split("#")[0])
            if urlparse(target).netloc.removeprefix("www.") != host:
                continue
        except ValueError:  # malformed hrefs are common in the wild; skip them
            continue
        for kind, rx in SUBPAGE_PATTERNS.items():
            if kind not in found and rx.search(label):
                found[kind] = target
    pages = {}
    for kind, url in list(found.items())[:4]:
        sub_url, sub_html = await fetch(session, url)
        if sub_url:
            sub_soup = BeautifulSoup(sub_html, "html.parser")
            pages[kind] = visible_text(sub_soup, SUB_TEXT_CAP)
            if kind == "careers":
                rec["careers_link_count"] = len(sub_soup.find_all("a", href=True))
    rec["homepage_text"] = visible_text(soup, HOME_TEXT_CAP)
    rec["pages"] = pages
    return rec


async def main(limit=None):
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    done_ids = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    todo = [r for r in rows if r["id"] not in done_ids]
    if limit:
        todo = todo[:limit]
    print(f"{len(rows):,} rows in TAM · {len(done_ids):,} already scraped · {len(todo):,} to go")

    sem = asyncio.Semaphore(CONCURRENCY)
    out = open(OUT_PATH, "a", encoding="utf-8")
    lock = asyncio.Lock()
    stats = {"done": 0, "ok": 0, "t0": time.time()}

    async def worker(row):
        try:
            async with sem:
                rec = await scrape_company(session, row)
        except Exception as e:  # one company's weirdness must never kill the run
            rec = {"id": row["id"], "domain": (row.get("website") or "").strip().lower(),
                   "ok": False, "error": f"scrape crash: {type(e).__name__}"}
        async with lock:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["done"] += 1
            stats["ok"] += 1 if rec.get("ok") else 0
            if stats["done"] % 250 == 0:
                rate = stats["done"] / (time.time() - stats["t0"])
                eta_min = (len(todo) - stats["done"]) / rate / 60
                out.flush()
                print(f"{stats['done']:,}/{len(todo):,} · {stats['ok']/stats['done']:.0%} ok "
                      f"· {rate:.1f}/s · ~{eta_min:.0f} min left", flush=True)

    connector = aiohttp.TCPConnector(limit=CONCURRENCY, ttl_dns_cache=300)
    async with aiohttp.ClientSession(headers={"User-Agent": UA},
                                     connector=connector) as session:
        await asyncio.gather(*(worker(r) for r in todo), return_exceptions=True)
    out.flush()
    out.close()
    print(f"finished: {stats['done']:,} scraped, {stats['ok']:,} ok "
          f"({stats['ok']/max(stats['done'],1):.0%}) in "
          f"{(time.time()-stats['t0'])/60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    asyncio.run(main(args.limit))
