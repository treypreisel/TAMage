#!/usr/bin/env python3
"""Spike stage 3+4: per-segment stats, then LLM naming/description/fit-scoring.

Winner config: u2 space (n_neighbors=30, n_components=15).
  coarse level = u2_m300 (22 segments)   fine level = u2_m150 (37 segments)

Commands:
  python spike/interpret.py stats                 -> spike/cluster_stats.json
  python spike/interpret.py name --level coarse [--variant A|B] [--only 3,7]
                                                  -> spike/segments_<level>_<variant>.json

The naming prompt is deliberately BLIND to the seed hypotheses — canary grading
happens separately so it can't flatter itself.
"""

import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LEVELS = {"coarse": "u2_m300", "fine": "u2_m150"}


def load_all():
    d = np.load(os.path.join(HERE, "embeddings.npz"), allow_pickle=True)
    L = np.load(os.path.join(HERE, "bakeoff_labels.npz"))
    df = pd.read_csv(os.path.join(HERE, "..", "dataset", "exports",
                                  "tamage_input.csv"), dtype=str).fillna("")
    df = df[df["enrich_status"] == "summarized"].reset_index(drop=True)
    assert list(df["id"]) == list(d["ids"]), "row order mismatch csv vs embeddings"
    return d, L, df


def distinguishing_terms(docs, labels, k_top=12):
    """Per-cluster distinguishing terms: p(t|c) * log(p(t|c)/p(t|corpus))."""
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(stop_words="english", ngram_range=(1, 2),
                         min_df=25, max_features=30000)
    X = cv.fit_transform(docs)
    vocab = np.array(cv.get_feature_names_out())
    corpus_p = np.asarray(X.sum(0)).ravel()
    corpus_p = corpus_p / corpus_p.sum()
    out = {}
    for c in sorted(set(labels[labels >= 0])):
        tf = np.asarray(X[labels == c].sum(0)).ravel()
        p = tf / max(tf.sum(), 1)
        score = p * np.log((p + 1e-12) / (corpus_p + 1e-12))
        top = np.argsort(-score)[:k_top]
        out[int(c)] = [vocab[i] for i in top if p[i] > 0]
    return out


def stats_cmd():
    d, L, df = load_all()
    docs = d["docs"]
    result = {}
    for level, key in LEVELS.items():
        labels, probs = L[key + "_labels"], L[key + "_probs"]
        terms = distinguishing_terms(docs, labels)
        clusters = {}
        for c in sorted(set(labels[labels >= 0])):
            idx = np.where(labels == c)[0]
            sub = df.iloc[idx]
            core = idx[np.argsort(-probs[idx])][:8]
            clusters[int(c)] = {
                "n": int(len(idx)),
                "terms": terms[int(c)],
                "exemplars": [str(docs[i])[:260] for i in core],
                "exemplar_names": [df.iloc[i]["name"] for i in core],
                "gtm": dict(Counter(x for x in sub["gtm_motion_hint"] if x)),
                "sizes": dict(Counter(sub["size"])),
                "founded_median": int(pd.to_numeric(
                    sub["founded"], errors="coerce").median() or 0),
                "top_tech": Counter(
                    t for hints in sub["tech_hints"] for t in hints.split(";")
                    if t).most_common(5),
                "countries": dict(Counter(sub["country"])),
            }
        result[level] = clusters
        print(f"{level}: {len(clusters)} clusters · "
              f"noise {(labels == -1).mean():.0%}")
    with open(os.path.join(HERE, "cluster_stats.json"), "w") as f:
        json.dump(result, f, indent=1)
    print("wrote cluster_stats.json")


PROMPT_A = """You are a senior GTM strategist analyzing a discovered market segment
for the vendor described below. Write for a sales team that will act on this.

<vendor_context>
{vendor}
</vendor_context>

<segment_data>
Companies in segment: {n}
Distinguishing terms (vs. whole market): {terms}
Go-to-market motion distribution: {gtm}
Company size distribution: {sizes}
Median founding year: {founded}
8 representative companies:
{exemplars}
</segment_data>

Rules: ground every claim in the data above; no invented facts; ban marketing
adjectives; the name must be specific enough that a rep instantly knows who's in
the segment (never "Tech Companies" or "Miscellaneous")."""

PROMPT_F = """You are a senior GTM strategist analyzing a discovered market segment
for the vendor described below. Write the segment card a sales team will act on.

<vendor_context>
{vendor}
</vendor_context>

<segment_data>
Companies in segment: {n}
Distinguishing terms (vs. whole market): {terms}
Go-to-market motion distribution: {gtm}
Company size distribution: {sizes}
Median founding year: {founded}
8 representative companies:
{exemplars}
</segment_data>

Rules:
- Ground every claim strictly in the data above. No invented product categories,
  no invented pricing models, no marketing adjectives.
- The name must identify what these companies SELL (prefer forms like
  "... Software Vendors", "... Platforms for X") — a rep reading only the name
  should correctly picture the next 10 companies in the segment.
- The opening angle must name a concrete operational struggle these specific
  companies plausibly have (drawn from what the exemplars do), written in
  outreach voice a rep could draft a first line from — never an instruction to
  the rep, never a restatement of the description.
- Score vendor_fit honestly and spread the scale: engage with how these
  companies actually price and bill. Flat per-seat subscription sellers are a
  WEAK fit for usage-based billing infrastructure. fit_risk must state the
  strongest reason this segment might be a bad fit."""

PROMPT_B = """Analyze this discovered market segment for the vendor below. You are
writing the segment card a GTM engineer reads before writing an outreach sequence.

<vendor_context>
{vendor}
</vendor_context>

<segment_data>
Size: {n} companies. Distinguishing terms: {terms}
GTM motions: {gtm} | Company sizes: {sizes} | Median founded: {founded}
Representative companies:
{exemplars}
</segment_data>

Think first about what SINGLE shared context unites these companies (what they
sell, who they sell to, how they price), then derive everything from that. Ground
strictly in the data; no invention; no praise adjectives; segment name must pass
the test "a rep reads only the name and correctly guesses 8 of the 10 next
companies in it"."""

SCHEMA_FIELDS = {
    "name": "3-6 word segment name",
    "description": "2-3 sentences: who they are, what unites them",
    "defining_traits": "array of 3-5 short trait strings",
    "vendor_fit": "integer 0-100: how valuable is this segment to the vendor",
    "fit_rationale": "1 sentence",
    "opening_angle": "1-2 sentences: the outreach angle for this segment",
    "fit_risk": "1 sentence: the strongest reason this segment might be a BAD fit for the vendor",
}


def name_cmd(level, variant, only):
    from google import genai
    from google.genai import types
    stats = json.load(open(os.path.join(HERE, "cluster_stats.json")))[level]
    vendor = open(os.path.join(HERE, "meterly_context.md")).read().split(
        "## Blessed seed hypotheses")[0]  # BLIND: strip the hypotheses section
    client = genai.Client(vertexai=True, project="gen-lang-client-0437407521",
                          location="us-central1")
    schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "name": types.Schema(type=types.Type.STRING,
                                 description=SCHEMA_FIELDS["name"]),
            "description": types.Schema(type=types.Type.STRING,
                                        description=SCHEMA_FIELDS["description"]),
            "defining_traits": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING),
                description=SCHEMA_FIELDS["defining_traits"]),
            "vendor_fit": types.Schema(type=types.Type.INTEGER,
                                       description=SCHEMA_FIELDS["vendor_fit"]),
            "fit_rationale": types.Schema(type=types.Type.STRING,
                                          description=SCHEMA_FIELDS["fit_rationale"]),
            "opening_angle": types.Schema(type=types.Type.STRING,
                                          description=SCHEMA_FIELDS["opening_angle"]),
            "fit_risk": types.Schema(type=types.Type.STRING,
                                     description=SCHEMA_FIELDS["fit_risk"]),
        },
        required=list(SCHEMA_FIELDS),
    )
    template = {"A": PROMPT_A, "B": PROMPT_B, "F": PROMPT_F}[variant]
    out = {}
    ids = only if only else list(stats.keys())
    for cid in ids:
        s = stats[str(cid)]
        prompt = template.format(
            vendor=vendor, n=s["n"], terms=", ".join(s["terms"]),
            gtm=s["gtm"], sizes=s["sizes"], founded=s["founded_median"],
            exemplars="\n".join(f"- {e}" for e in s["exemplars"]),
        )
        cfg = types.GenerateContentConfig(
            temperature=0.3, max_output_tokens=4096,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json", response_schema=schema)
        for attempt in range(3):
            resp = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt, config=cfg)
            try:
                out[str(cid)] = json.loads(resp.text)
                break
            except (json.JSONDecodeError, TypeError):
                if attempt == 2:
                    raise
        print(f"[{cid}] {out[str(cid)]['name']} (fit {out[str(cid)]['vendor_fit']})",
              flush=True)
    path = os.path.join(HERE, f"segments_{level}_{variant}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stats", "name"])
    ap.add_argument("--level", default="coarse", choices=list(LEVELS))
    ap.add_argument("--variant", default="F", choices=["A", "B", "F"])
    ap.add_argument("--only", default=None,
                    help="comma-separated cluster ids (tournament mode)")
    a = ap.parse_args()
    if a.cmd == "stats":
        stats_cmd()
    else:
        name_cmd(a.level, a.variant,
                 [x.strip() for x in a.only.split(",")] if a.only else None)
