#!/usr/bin/env python3
"""Spike stage 1: serialize summarized companies to text and embed via Vertex.

Serialization judgment calls (deliberate, documented):
- Embed ONLY the semantic fields: company_description + product_description +
  gtm motion. What a company DOES should drive similarity.
- EXCLUDED from embedding: geography (messaging doesn't differ by city), founded
  year, employee size, company name (brand tokens are noise), title/meta
  (redundant with descriptions), tech_hints (weak signal; risk of artifact
  "wordpress cluster"). Size/founded/tech stay available as interpretation
  overlays and independent corroboration for the legacy-holdouts canary.
- Missing fields are silently omitted (no placeholders) per the project rule.

Model: gemini-embedding-001, task_type=CLUSTERING, 768 dims (re-normalized).
Output: spike/embeddings.npz  (ids, vectors, docs)
"""

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from google import genai
from google.genai import types

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(HERE, "..", "dataset", "exports", "tamage_input.csv")
OUT = os.path.join(HERE, "embeddings.npz")

PROJECT = "gen-lang-client-0437407521"
LOCATION = "us-central1"
MODEL = "gemini-embedding-001"
DIMS = 768
BATCH = 1    # Vertex gemini-embedding-001 allows ONE input text per request
WORKERS = 24


def serialize(row):
    parts = [row["company_description"]]
    if row.get("product_description"):
        parts.append(row["product_description"])
    if row.get("gtm_motion_hint"):
        parts.append(f"Go-to-market motion: {row['gtm_motion_hint']}.")
    return " ".join(p.strip() for p in parts if p and str(p).strip())


def main(limit=None):
    df = pd.read_csv(INPUT, dtype=str).fillna("")
    df = df[df["enrich_status"] == "summarized"].reset_index(drop=True)
    if limit:
        df = df.head(limit)
    docs = [serialize(r) for r in df.to_dict("records")]
    ids = df["id"].tolist()
    print(f"{len(docs):,} documents to embed")

    client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)
    vectors = [None] * len(docs)
    lock = threading.Lock()
    done = {"n": 0}
    t0 = time.time()

    def embed_batch(start):
        chunk = docs[start:start + BATCH]
        delay = 2.0
        for attempt in range(5):
            try:
                resp = client.models.embed_content(
                    model=MODEL, contents=chunk,
                    config=types.EmbedContentConfig(
                        task_type="CLUSTERING", output_dimensionality=DIMS),
                )
                assert len(resp.embeddings) == len(chunk), \
                    f"batch {start}: got {len(resp.embeddings)} embeddings for {len(chunk)} inputs"
                return start, [e.values for e in resp.embeddings]
            except Exception as e:
                if attempt == 4:
                    raise
                time.sleep(delay * (3 if "429" in str(e) or "RESOURCE" in str(e) else 1))
                delay *= 2

    with ThreadPoolExecutor(WORKERS) as ex:
        futures = [ex.submit(embed_batch, s) for s in range(0, len(docs), BATCH)]
        for fut in as_completed(futures):
            start, vecs = fut.result()
            with lock:
                for i, v in enumerate(vecs):
                    vectors[start + i] = v
                done["n"] += len(vecs)
                if done["n"] % (BATCH * 20) < BATCH:
                    rate = done["n"] / (time.time() - t0)
                    print(f"{done['n']:,}/{len(docs):,} · {rate:.0f}/s", flush=True)

    arr = np.array(vectors, dtype=np.float32)
    # 768-dim truncation of a Matryoshka embedding must be re-normalized
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    np.savez_compressed(OUT, ids=np.array(ids), vectors=arr,
                        docs=np.array(docs, dtype=object))
    print(f"saved {arr.shape} → {OUT} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    main(ap.parse_args().limit)
