#!/usr/bin/env python3
"""Live dashboard for the Lane B enrichment sweep.

Serves progress + quality metrics from the dataset layer files:
  scraped.jsonl  (Lane A, complete — parsed once at startup)
  enriched.jsonl (Lane B, growing — re-read on each poll)

Usage: python dataset/dashboard.py [--port 8766]
"""

import argparse
import json
import os
import threading
import time
from collections import Counter, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
SCRAPED = os.path.join(HERE, "scraped.jsonl")
ENRICHED = os.path.join(HERE, "enriched.jsonl")
TAM_TOTAL = 55_360
COST_PER_CALLED_ROW = 0.0006  # pilot-derived: tokens/row x flash pricing

_history = deque(maxlen=60)   # (t, done) samples for rate calc
_lock = threading.Lock()


def lane_a_stats():
    ok = 0
    errs, pages, tech = Counter(), Counter(), Counter()
    with open(SCRAPED, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ok"):
                ok += 1
                for k in (r.get("pages") or {}):
                    pages[k] += 1
                for t in (r.get("tech_hints") or []):
                    tech[t] += 1
            else:
                errs[(r.get("error") or "?").split(":")[0]] += 1
    return {
        "reachable": ok, "failed": sum(errs.values()),
        "errors": errs.most_common(6),
        "subpages": sorted(pages.items(), key=lambda x: -x[1]),
        "tech": tech.most_common(10),
    }


LANE_A = None  # filled at startup


def lane_b_stats():
    done = summarized = skipped = errors = 0
    gtm = Counter()
    recent = deque(maxlen=4)
    if os.path.exists(ENRICHED):
        with open(ENRICHED, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # partial last line while writer is mid-append
                done += 1
                if r.get("skipped"):
                    skipped += 1
                elif r.get("error"):
                    errors += 1
                if r.get("company_description"):
                    summarized += 1
                    recent.append({"domain": r.get("domain", ""),
                                   "desc": r["company_description"][:110]})
                if r.get("gtm_motion_hint"):
                    gtm[r["gtm_motion_hint"]] += 1
    now = time.time()
    with _lock:
        _history.append((now, done))
        rate = eta_min = None
        if len(_history) >= 2:
            (t0, n0), (t1, n1) = _history[0], _history[-1]
            if t1 > t0 and n1 > n0:
                rate = (n1 - n0) / (t1 - t0) * 60          # rows/min
                eta_min = (LANE_A["reachable"] - n1) / max(rate, 1e-9)
    called = done - skipped
    return {
        "target": LANE_A["reachable"], "done": done, "summarized": summarized,
        "skipped": skipped, "errors": errors, "gtm": gtm.most_common(),
        "rate_per_min": round(rate, 1) if rate else None,
        "eta_min": round(eta_min) if eta_min and eta_min > 0 else None,
        "est_cost": round(called * COST_PER_CALLED_ROW, 2),
        "recent": list(recent), "running": done < LANE_A["reachable"],
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/stats":
            self._send(200, {"lane_a": LANE_A, "lane_b": lane_b_stats()})
        else:
            self._send(404, {"error": "not found"})


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>TAMage Sweep Dashboard</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing:border-box; margin:0; }
  body { background:#0d0f12; color:#e6e8eb; font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; padding:24px; max-width:1100px; margin:0 auto; }
  h1 { font-size:15px; letter-spacing:.12em; color:#9aa3ad; }
  .sub { color:#6b7480; font-size:12px; margin:2px 0 18px; }
  .bar { height:14px; background:#1a1e25; border-radius:7px; overflow:hidden; margin:10px 0 6px; }
  .bar>div { height:100%; background:#6c9ef8; border-radius:7px; transition:width .8s; }
  .barlabel { display:flex; justify-content:space-between; color:#9aa3ad; font-size:12px; margin-bottom:20px; }
  .tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin-bottom:20px; }
  .tile { background:#14171c; border:1px solid #232830; border-radius:8px; padding:12px 14px; }
  .tile .v { font-size:26px; font-weight:700; font-variant-numeric:tabular-nums; }
  .tile .k { font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:#9aa3ad; margin-top:2px; }
  .tile.err .v { color:#e07a6a; } .tile.ok .v { color:#7fc98f; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:12px; margin-bottom:12px; }
  .panel { background:#14171c; border:1px solid #232830; border-radius:8px; padding:14px; }
  .panel h2 { font-size:11px; letter-spacing:.1em; text-transform:uppercase; color:#9aa3ad; margin-bottom:10px; }
  table { width:100%; border-collapse:collapse; }
  td { padding:3px 4px; border-bottom:1px solid #1c2129; vertical-align:middle; }
  td.n { text-align:right; color:#9aa3ad; font-variant-numeric:tabular-nums; white-space:nowrap; }
  .rowbar { height:6px; background:#6c9ef8; border-radius:3px; opacity:.75; min-width:2px; }
  .ticker div { padding:5px 0; border-bottom:1px solid #1c2129; color:#c9ced4; }
  .ticker b { color:#e6e8eb; font-weight:600; }
  .done-banner { background:#12291a; border:1px solid #245c33; color:#7fc98f; padding:10px 14px; border-radius:8px; margin-bottom:16px; display:none; }
</style></head><body>
<h1>TAMAGE · LANE B SWEEP</h1>
<div class="sub">summarizing scraped sites → structured AI fields · polls every 5s</div>
<div class="done-banner" id="doneBanner">Sweep complete — canonical dataset ready for assembly.</div>
<div class="bar"><div id="fill" style="width:0%"></div></div>
<div class="barlabel"><span id="pct">…</span><span id="counts">…</span></div>
<div class="tiles">
  <div class="tile"><div class="v" id="rate">…</div><div class="k">rows / min</div></div>
  <div class="tile"><div class="v" id="eta">…</div><div class="k">est. remaining</div></div>
  <div class="tile ok"><div class="v" id="summ">…</div><div class="k">summarized</div></div>
  <div class="tile"><div class="v" id="skip">…</div><div class="k">skipped (thin text)</div></div>
  <div class="tile err"><div class="v" id="errs">…</div><div class="k">api errors</div></div>
  <div class="tile"><div class="v" id="cost">…</div><div class="k">est. spend</div></div>
</div>
<div class="grid">
  <div class="panel"><h2>Funnel</h2><table id="funnel"></table></div>
  <div class="panel"><h2>GTM motion (AI-classified)</h2><table id="gtm"></table></div>
  <div class="panel"><h2>Latest enrichments</h2><div class="ticker" id="ticker"></div></div>
</div>
<div class="grid">
  <div class="panel"><h2>Lane A · unreachable breakdown</h2><table id="laneErr"></table></div>
  <div class="panel"><h2>Lane A · subpages found</h2><table id="subp"></table></div>
  <div class="panel"><h2>Lane A · top tech fingerprints</h2><table id="tech"></table></div>
</div>
<script>
const $ = id => document.getElementById(id);
const fmt = n => n == null ? '—' : n.toLocaleString();

function rows(el, items, max) {
  el.innerHTML = items.map(([k, v]) =>
    `<tr><td>${k}</td><td style="width:40%"><div class="rowbar" style="width:${(v / max * 100).toFixed(1)}%"></div></td><td class="n">${fmt(v)}</td></tr>`
  ).join('');
}

async function refresh() {
  try {
    const d = await (await fetch('/api/stats')).json();
    const a = d.lane_a, b = d.lane_b;
    const pct = (b.done / b.target * 100);
    $('fill').style.width = pct.toFixed(1) + '%';
    $('pct').textContent = pct.toFixed(1) + '% of reachable companies processed';
    $('counts').textContent = `${fmt(b.done)} / ${fmt(b.target)}`;
    $('rate').textContent = fmt(b.rate_per_min);
    $('eta').textContent = b.eta_min == null ? '—' : (b.eta_min > 90 ? (b.eta_min/60).toFixed(1) + ' h' : b.eta_min + ' min');
    $('summ').textContent = fmt(b.summarized);
    $('skip').textContent = fmt(b.skipped);
    $('errs').textContent = fmt(b.errors);
    $('cost').textContent = '$' + (b.est_cost ?? 0).toFixed(2);
    $('doneBanner').style.display = b.running ? 'none' : 'block';
    $('funnel').innerHTML = [
      ['curated TAM (PDL slice)', 55360],
      ['reachable (Lane A)', a.reachable],
      ['processed (Lane B)', b.done],
      ['fully summarized', b.summarized],
    ].map(([k,v]) => `<tr><td>${k}</td><td class="n">${fmt(v)}</td></tr>`).join('');
    rows($('gtm'), b.gtm, Math.max(1, ...b.gtm.map(g => g[1])));
    $('ticker').innerHTML = b.recent.map(r => `<div><b>${r.domain}</b> — ${r.desc}…</div>`).join('') || '<div>waiting…</div>';
    rows($('laneErr'), a.errors, Math.max(...a.errors.map(e => e[1])));
    rows($('subp'), a.subpages, Math.max(...a.subpages.map(e => e[1])));
    rows($('tech'), a.tech, Math.max(...a.tech.map(e => e[1])));
  } catch (e) { /* server briefly busy; next poll catches up */ }
}
refresh(); setInterval(refresh, 5000);
</script>
</body></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()
    print("parsing Lane A stats (one-time)…")
    LANE_A = lane_a_stats()
    print(f"dashboard on http://localhost:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
