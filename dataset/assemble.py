#!/usr/bin/env python3
"""Assemble the canonical TAMage input CSV from the three dataset layers.

Joins by id:
  exports/tam_main.csv   - PDL structured fields (all 55,360 rows, always kept)
  scraped.jsonl          - Lane A: reachability, title, meta, tech hints
  enriched.jsonl         - Lane B: AI fields (last write wins on duplicates)

Every curated row appears in the output exactly once. Rows that failed a stage
carry their reason in enrich_status/kick_reason instead of being dropped -
TAMage's audit stage is supposed to see the messy reality.

Output: exports/tamage_input.csv
"""

import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TAM = os.path.join(HERE, "exports", "tam_main.csv")
SCRAPED = os.path.join(HERE, "scraped.jsonl")
ENRICHED = os.path.join(HERE, "enriched.jsonl")
OUT = os.path.join(HERE, "exports", "tamage_input.csv")

PDL_FIELDS = ["id", "name", "website", "founded", "size", "locality",
              "region", "country", "industry", "linkedin_url"]
OUT_FIELDS = PDL_FIELDS + [
    "reachable", "kick_reason", "title", "meta_description", "tech_hints",
    "company_description", "product_description", "gtm_motion_hint",
    "enrich_status",
]


def load_jsonl(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                d[r["id"]] = r  # last write wins
            except (json.JSONDecodeError, KeyError):
                continue
    return d


def main():
    scraped = load_jsonl(SCRAPED)
    enriched = load_jsonl(ENRICHED)
    n = {"total": 0, "summarized": 0, "thin": 0, "unreachable": 0, "error": 0}
    with open(TAM, newline="", encoding="utf-8") as fin, \
         open(OUT, "w", newline="", encoding="utf-8") as fout:
        w = csv.DictWriter(fout, fieldnames=OUT_FIELDS)
        w.writeheader()
        for row in csv.DictReader(fin):
            n["total"] += 1
            s = scraped.get(row["id"]) or {}
            e = enriched.get(row["id"]) or {}
            if not s.get("ok"):
                status = "unreachable"
            elif e.get("company_description"):
                status = "summarized"
            elif e.get("skipped"):
                status = "thin_text"
            elif e.get("error"):
                status = "llm_error"
            elif e:
                status = "model_null"  # processed; model judged junk/non-company site
            else:
                status = "not_processed"
            key = ("unreachable" if status == "unreachable" else
                   "summarized" if status == "summarized" else
                   "thin" if status == "thin_text" else "error")
            n[key] += 1
            w.writerow({
                **{k: row.get(k) for k in PDL_FIELDS},
                "reachable": "yes" if s.get("ok") else "no",
                "kick_reason": (s.get("error") or "") if not s.get("ok") else "",
                "title": s.get("title") or "",
                "meta_description": s.get("meta_description") or "",
                "tech_hints": ";".join(s.get("tech_hints") or []),
                "company_description": e.get("company_description") or "",
                "product_description": e.get("product_description") or "",
                "gtm_motion_hint": e.get("gtm_motion_hint") or "",
                "enrich_status": status,
            })
    print(f"wrote {OUT}")
    print(f"rows={n['total']:,} · summarized={n['summarized']:,} · "
          f"thin={n['thin']:,} · unreachable={n['unreachable']:,} · other={n['error']:,}")


if __name__ == "__main__":
    main()
