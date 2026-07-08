#!/usr/bin/env python3
"""Spike stage 5: action queues, TAM map, per-account CSV, and the report.

Queue thresholds are spike judgment calls, documented here:
  assigned + prob>=0.60 + margin>=0.15:
      fit>=70 -> route_now | fit<45 -> suppress | else -> nurture
  assigned but prob<0.60 or margin<0.15  -> enrich_next  (ambiguous)
  noise with nearest-segment lean >=0.15 -> enrich_next  (leaning)
  noise with no meaningful lean          -> low_info

Outputs: spike/output/{segments.csv, tam_map.png, report.html}
"""

import base64
import html
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
KEY_C, KEY_F = "u2_m300", "u2_m150"


def load():
    d = np.load(os.path.join(HERE, "embeddings.npz"), allow_pickle=True)
    L = np.load(os.path.join(HERE, "bakeoff_labels.npz"))
    seg_c = json.load(open(os.path.join(HERE, "segments_coarse_F.json")))
    seg_f = json.load(open(os.path.join(HERE, "segments_fine_F.json")))
    stats = json.load(open(os.path.join(HERE, "cluster_stats.json")))
    df = pd.read_csv(os.path.join(HERE, "..", "dataset", "exports",
                                  "tamage_input.csv"), dtype=str).fillna("")
    df = df[df["enrich_status"] == "summarized"].reset_index(drop=True)
    xy = np.load(os.path.join(HERE, "umap2d.npy"))
    return d, L, seg_c, seg_f, stats, df, xy


def build_accounts(d, L, seg_c, df):
    labels = L[KEY_C + "_labels"]
    probs = L[KEY_C + "_probs"]
    t2i, t2p = L[KEY_C + "_top2_idx"], L[KEY_C + "_top2_prob"]
    rows = []
    for i in range(len(labels)):
        lab = int(labels[i])
        if lab >= 0:
            conf = float(probs[i])
            margin = float(t2p[i, 0] - t2p[i, 1])
            seg = seg_c[str(lab)]
            fit = int(seg["vendor_fit"])
            if conf >= 0.60 and margin >= 0.15:
                action = ("route_now" if fit >= 70 else
                          "suppress" if fit < 45 else "nurture")
            else:
                action = "enrich_next"
            seg_name = seg["name"]
        else:
            lean = float(t2p[i, 0])
            near = int(t2i[i, 0])
            if lean >= 0.15:
                action, conf, margin = "enrich_next", lean, 0.0
                seg_name, fit = f"~{seg_c[str(near)]['name']} (leaning)", \
                    int(seg_c[str(near)]["vendor_fit"])
            else:
                action, conf, margin, seg_name, fit = "low_info", 0.0, 0.0, \
                    "unassigned", 0
        rows.append({
            "id": df.iloc[i]["id"], "name": df.iloc[i]["name"],
            "website": df.iloc[i]["website"], "segment": seg_name,
            "segment_id": lab, "confidence": round(conf, 3),
            "margin": round(margin, 3), "vendor_fit": fit, "action": action,
        })
    return pd.DataFrame(rows)


def draw_map(xy, L, seg_c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = L[KEY_C + "_labels"]
    fig, ax = plt.subplots(figsize=(14, 10), facecolor="#0d0f12")
    ax.set_facecolor("#0d0f12")
    noise = labels < 0
    ax.scatter(xy[noise, 0], xy[noise, 1], s=1.2, c="#3a4048", alpha=.5,
               linewidths=0)
    cmap = plt.cm.tab20(np.linspace(0, 1, 20))
    ks = sorted(set(labels[labels >= 0]))
    for j, k in enumerate(ks):
        m = labels == k
        ax.scatter(xy[m, 0], xy[m, 1], s=1.6, color=cmap[j % 20], alpha=.75,
                   linewidths=0)
        cx, cy = np.median(xy[m, 0]), np.median(xy[m, 1])
        nm = seg_c[str(k)]["name"]
        ax.annotate(nm, (cx, cy), color="#e6e8eb", fontsize=7.5, ha="center",
                    fontweight="bold", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#0d0f12",
                              ec="#2a303a", alpha=.85))
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("TAM map — 25,169 B2B software companies, "
                 "22 discovered segments (gray = unassigned)",
                 color="#9aa3ad", fontsize=11, family="monospace", pad=14)
    path = os.path.join(OUT, "tam_map.png")
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="#0d0f12")
    plt.close(fig)
    return path


def subsegment_table(L, seg_c, seg_f):
    lc, lf = L[KEY_C + "_labels"], L[KEY_F + "_labels"]
    mapping = {}
    for k in sorted(set(lc[lc >= 0])):
        subs = Counter(lf[(lc == k) & (lf >= 0)])
        mapping[str(k)] = [(seg_f[str(s)]["name"], int(n))
                           for s, n in subs.most_common(4)]
    return mapping


def render_report(accounts, seg_c, stats, map_path, submap, df):
    q = accounts["action"].value_counts().to_dict()
    n = len(accounts)
    with open(map_path, "rb") as f:
        map_b64 = base64.b64encode(f.read()).decode()
    e = html.escape
    order = sorted(seg_c, key=lambda c: -seg_c[c]["vendor_fit"])
    cards = ""
    for cid in order:
        s, st = seg_c[cid], stats["coarse"][cid]
        subs = " · ".join(f"{e(nm)} ({cnt})" for nm, cnt in submap.get(cid, []))
        gtm = max(st["gtm"], key=st["gtm"].get) if st["gtm"] else "n/a"
        cards += f"""
<div class="card">
 <div class="cardhead"><span class="fit f{'hi' if s['vendor_fit']>=70 else 'md' if s['vendor_fit']>=45 else 'lo'}">{s['vendor_fit']}</span>
 <h3>{e(s['name'])}</h3><span class="n">{st['n']:,} companies</span></div>
 <p>{e(s['description'])}</p>
 <p class="traits">{' · '.join(e(t) for t in s['defining_traits'])}</p>
 <p><b>Angle:</b> {e(s['opening_angle'])}</p>
 <p class="risk"><b>Fit risk:</b> {e(s['fit_risk'])}</p>
 <p class="meta">dominant motion: {e(gtm)} · median founded: {st['founded_median']} · e.g. {e(', '.join(st['exemplar_names'][:4]))}</p>
 <p class="meta">contains (fine level): {subs}</p>
</div>"""
    queues = ""
    for act, label, desc in [
        ("route_now", "Route to campaign now", "high confidence, high fit — write the sequence"),
        ("nurture", "Nurture", "clearly segmented, moderate fit — cheap-touch track"),
        ("enrich_next", "Buy data on these next", "ambiguous placement — the next enrichment dollar changes a decision here"),
        ("suppress", "Suppress", "confidently placed in low-fit segments — don't spend here"),
        ("low_info", "Low information", "not enough signal to place — sample manually or ignore"),
    ]:
        sample = accounts[accounts["action"] == act].head(5)
        names = ", ".join(e(x) for x in sample["name"])
        queues += f"""<tr><td><b>{label}</b><br><span class="meta">{desc}</span></td>
<td class="num">{q.get(act, 0):,}</td><td class="meta">{names}</td></tr>"""
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>TAMage spike report — Meterly</title><style>
 body{{background:#0d0f12;color:#e6e8eb;font:14px/1.55 ui-monospace,Menlo,monospace;max-width:1000px;margin:0 auto;padding:32px 20px}}
 h1{{font-size:20px;letter-spacing:.08em}} h2{{font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:#9aa3ad;margin:34px 0 12px;border-bottom:1px solid #232830;padding-bottom:6px}}
 .sub{{color:#6b7480}} img{{max-width:100%;border-radius:10px;border:1px solid #232830}}
 .tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:18px 0}}
 .tile{{background:#14171c;border:1px solid #232830;border-radius:8px;padding:10px 12px}}
 .tile b{{font-size:22px;display:block}} .tile span{{font-size:11px;color:#9aa3ad;text-transform:uppercase;letter-spacing:.06em}}
 .card{{background:#14171c;border:1px solid #232830;border-radius:10px;padding:16px 18px;margin:12px 0}}
 .cardhead{{display:flex;gap:10px;align-items:baseline}} .cardhead h3{{font-size:15px}}
 .cardhead .n{{margin-left:auto;color:#6b7480;font-size:12px}}
 .fit{{font-weight:700;border-radius:6px;padding:2px 8px;font-size:13px}}
 .fhi{{background:#173321;color:#7fc98f}} .fmd{{background:#2e2a17;color:#d9b96a}} .flo{{background:#331b18;color:#e07a6a}}
 .traits{{color:#9aa3ad;font-size:12.5px}} .risk{{color:#d9b96a}} .meta{{color:#6b7480;font-size:12px}}
 table{{width:100%;border-collapse:collapse;margin:8px 0}} td{{padding:8px 6px;border-bottom:1px solid #1c2129;vertical-align:top}}
 td.num{{font-size:20px;font-weight:700;white-space:nowrap}}
</style></head><body>
<h1>TAMage — segmentation spike report</h1>
<p class="sub">Vendor persona: Meterly (usage-based billing infrastructure) · dataset: 55,360 curated B2B software companies → 25,169 enriched & embedded · one-shot capability run, 2026-07-08</p>
<div class="tiles">
 <div class="tile"><b>22</b><span>segments (coarse)</span></div>
 <div class="tile"><b>37</b><span>segments (fine)</span></div>
 <div class="tile"><b>{q.get('route_now',0):,}</b><span>route now</span></div>
 <div class="tile"><b>{q.get('enrich_next',0):,}</b><span>enrich next</span></div>
 <div class="tile"><b>{q.get('suppress',0):,}</b><span>suppress</span></div>
 <div class="tile"><b>{q.get('low_info',0):,}</b><span>low info</span></div>
</div>
<h2>The map</h2>
<img src="data:image/png;base64,{map_b64}" alt="TAM map">
<h2>Granularity — the one decision the tool won't make for you</h2>
<p>The same landscape resolves cleanly at two zoom levels: <b>22 segments</b> (write ~15-20 sequences; broad audiences) or <b>37 segments</b> (tighter audiences, roughly one per niche). Segment count is a strategy decision — it equals the number of distinct messages your team will write and maintain. Each coarse card below lists the fine segments it contains, so you can see exactly what splitting buys you.</p>
<h2>Action queues</h2>
<table>{queues}</table>
<h2>Segment cards — sorted by Meterly fit</h2>
{cards}
<h2>Method & honest caveats</h2>
<p class="meta">Pipeline: company descriptions (AI-summarized from scraped sites) + GTM motion serialized to text → gemini-embedding-001 (768d, clustering mode) → UMAP → HDBSCAN (config chosen by structural metrics + blind LLM coherence panel across 12 candidates) → per-segment stats → LLM segment cards (prompt selected by blind A/B tournament) → deterministic queue thresholds (documented in report.py). Caveats: single embedding model, single run — cross-run stability not yet verified; queue thresholds are first-pass judgment calls; fit scores are LLM-estimated from segment profiles, not validated against outcomes; ~29% of companies land outside any dense segment, which is honest — they feed the enrichment and low-info queues rather than being force-assigned.</p>
</body></html>"""
    path = os.path.join(OUT, "report.html")
    with open(path, "w") as f:
        f.write(doc)
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    d, L, seg_c, seg_f, stats, df, xy = load()
    accounts = build_accounts(d, L, seg_c, df)
    accounts.to_csv(os.path.join(OUT, "segments.csv"), index=False)
    print("queues:", accounts["action"].value_counts().to_dict())
    map_path = draw_map(xy, L, seg_c)
    submap = subsegment_table(L, seg_c, seg_f)
    path = render_report(accounts, seg_c, stats, map_path, submap, df)
    print("report:", path)


if __name__ == "__main__":
    main()
