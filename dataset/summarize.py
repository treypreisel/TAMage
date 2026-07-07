#!/usr/bin/env python3
"""Lane B: LLM summarization of scraped company sites into four structured fields.

Reads dataset/scraped.jsonl (Lane A output) + exports/tam_main.csv (names),
calls Gemini Flash on Vertex AI (ADC auth, no API key), and appends one JSON
line per company to dataset/enriched.jsonl.

Grounding fence: the model sees ONLY scraped text and must return null for
anything the text doesn't support. Schema is enforced by constrained decoding
(response_schema), then re-validated programmatically. Resumable by id.

Usage:
  python dataset/summarize.py --limit 100          # pilot
  python dataset/summarize.py                      # full run
"""

import argparse
import csv
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types

HERE = os.path.dirname(os.path.abspath(__file__))
SCRAPED = os.path.join(HERE, "scraped.jsonl")
TAM_CSV = os.path.join(HERE, "exports", "tam_main.csv")
OUT = os.path.join(HERE, "enriched.jsonl")

PROJECT = "gen-lang-client-0437407521"
LOCATION = "us-central1"
MODEL = "gemini-2.5-flash"
WORKERS = 16
MIN_TEXT = 200          # below this many chars of total text, skip the API call
CAPS = {"company_description": 600, "product_description": 400}

SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "company_description": types.Schema(type=types.Type.STRING, nullable=True),
        "product_description": types.Schema(type=types.Type.STRING, nullable=True),
        "gtm_motion_hint": types.Schema(type=types.Type.STRING, nullable=True,
                                        enum=["self-serve", "sales-led", "hybrid"]),
    },
    required=["company_description", "product_description", "gtm_motion_hint"],
)

PROMPT = """You are extracting factual company data. Use ONLY the provided text.
Never use outside knowledge, never guess. If the text doesn't support a field,
return null for it.

RULES
- company_description: 2-3 sentences, what the company is and does. Factual, no praise.
- product_description: 1-2 sentences, what they sell and who buys it.
- gtm_motion_hint: "self-serve" if visitors can sign up/buy without talking to
  sales, "sales-led" if contact-sales/demo-request is the only path, "hybrid"
  if both. null if the text doesn't show it.
- Ban marketing adjectives (leading, innovative, best-in-class, cutting-edge,
  world-class, revolutionary). Prefer nouns and specifics: industries, use
  cases, buyer types, integrations, price points.
- If the text is thin, boilerplate, a parked domain, or not a company site,
  return null for every field.

INPUT
company name: {name}
domain: {domain}
page title: {title}
meta description: {meta}
homepage text: {home}
{subpages}"""


def build_input(rec, name):
    parts = []
    for kind, text in (rec.get("pages") or {}).items():
        if kind == "careers":  # hiring_signal was cut; don't pay for careers text
            continue
        parts.append(f"{kind} page text: {text[:2500]}")
    return PROMPT.format(
        name=name, domain=rec.get("domain", ""),
        title=rec.get("title") or "(none)",
        meta=rec.get("meta_description") or "(none)",
        home=(rec.get("homepage_text") or "(none)")[:6000],
        subpages="\n".join(parts) or "(no subpages found)",
    )


def total_text(rec):
    return (len(rec.get("homepage_text") or "") +
            len(rec.get("meta_description") or "") +
            sum(len(t) for t in (rec.get("pages") or {}).values()))


def validate(d):
    """Programmatic re-check behind the constrained decoder."""
    out = {}
    for k in ("company_description", "product_description"):
        v = d.get(k)
        out[k] = v.strip()[:CAPS[k]] if isinstance(v, str) and v.strip() else None
    g = d.get("gtm_motion_hint")
    out["gtm_motion_hint"] = g if g in ("self-serve", "sales-led", "hybrid") else None
    return out


def summarize_one(client, rec, name, stats, lock):
    row = {"id": rec["id"], "domain": rec.get("domain", "")}
    if total_text(rec) < MIN_TEXT:
        row.update(company_description=None, product_description=None,
                   gtm_motion_hint=None, skipped="insufficient text")
        return row
    delay = 2.0
    for attempt in range(4):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=build_input(rec, name),
                config=types.GenerateContentConfig(
                    temperature=0.2, max_output_tokens=1024,
                    response_mime_type="application/json",
                    response_schema=SCHEMA,
                ),
            )
            row.update(validate(json.loads(resp.text)))
            u = resp.usage_metadata
            with lock:
                stats["in_tok"] += u.prompt_token_count or 0
                stats["out_tok"] += u.candidates_token_count or 0
            return row
        except Exception as e:
            msg = str(e)
            if attempt == 3:
                row.update(company_description=None, product_description=None,
                           gtm_motion_hint=None,
                           error=f"{type(e).__name__}: {msg[:150]}")
                return row
            # back off harder on rate limits, gently on everything else
            time.sleep(delay * (3 if "429" in msg or "RESOURCE_EXHAUSTED" in msg else 1))
            delay *= 2


def main(limit=None, seed=7):
    names = {}
    with open(TAM_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            names[r["id"]] = r["name"]
    done = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if not r.get("error"):  # error rows get retried on resume
                        done.add(r["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    todo = []
    with open(SCRAPED, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("ok") and rec["id"] not in done:
                todo.append(rec)
    if limit:
        random.Random(seed).shuffle(todo)  # pilot samples the whole spectrum
        todo = todo[:limit]
    print(f"{len(todo):,} companies to summarize ({len(done):,} already done)")

    client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)
    stats = {"in_tok": 0, "out_tok": 0}
    lock = threading.Lock()
    t0 = time.time()
    with open(OUT, "a", encoding="utf-8") as out, ThreadPoolExecutor(WORKERS) as ex:
        futures = {ex.submit(summarize_one, client, r, names.get(r["id"], ""),
                             stats, lock): r for r in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            with lock:
                out.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
                if i % 25 == 0:
                    out.flush()
                    rate = i / (time.time() - t0)
                    print(f"{i:,}/{len(todo):,} · {rate:.1f}/s · "
                          f"~{(len(todo)-i)/rate/60:.0f} min left", flush=True)
    # flash pricing ballpark: $0.30/M input, $2.50/M output (check current)
    cost = stats["in_tok"] / 1e6 * 0.30 + stats["out_tok"] / 1e6 * 2.50
    print(f"done in {(time.time()-t0)/60:.1f} min · "
          f"{stats['in_tok']:,} in / {stats['out_tok']:,} out tokens · ~${cost:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    main(args.limit)
