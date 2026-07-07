#!/usr/bin/env python3
"""TAM Curator — local mini-app for carving an ICP slice out of the PDL Free Company Dataset.

Usage:  python dataset/tam_curator.py [--port 8765]
Reads:  dataset/companies.parquet  (built by the one-time JSONL->Parquet conversion)
Writes: dataset/exports/tam_export_<timestamp>.csv on export
"""

import argparse
import datetime
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
PARQUET = os.path.join(HERE, "companies.parquet")
EXPORT_DIR = os.path.join(HERE, "exports")

SIZE_ORDER = ["1-10", "11-50", "51-200", "201-500", "501-1000",
              "1001-5000", "5001-10000", "10001+"]

_con = duckdb.connect()
_lock = threading.Lock()  # duckdb connections are not thread-safe across requests
_facets_cache = None


def q(sql, params=None):
    with _lock:
        return _con.execute(sql, params or []).fetchall()


def build_where(f):
    """Filters dict -> (where_sql, params). Every clause is parameterized."""
    clauses, params = [], []
    if f.get("industries"):
        clauses.append(f"industry IN ({','.join('?' * len(f['industries']))})")
        params += f["industries"]
    if f.get("sizes"):
        clauses.append(f"size IN ({','.join('?' * len(f['sizes']))})")
        params += f["sizes"]
    if f.get("countries"):
        clauses.append(f"country IN ({','.join('?' * len(f['countries']))})")
        params += f["countries"]
    if f.get("region_contains"):
        clauses.append("(region ILIKE ? OR locality ILIKE ?)")
        params += [f"%{f['region_contains']}%"] * 2
    if f.get("name_contains"):
        clauses.append("name ILIKE ?")
        params.append(f"%{f['name_contains']}%")
    if f.get("founded_min"):
        clauses.append("founded >= ?")
        params.append(int(f["founded_min"]))
    if f.get("founded_max"):
        clauses.append("founded <= ?")
        params.append(int(f["founded_max"]))
    if f.get("require_website"):
        clauses.append("website IS NOT NULL AND website != ''")
    if f.get("require_linkedin"):
        clauses.append("linkedin_url IS NOT NULL AND linkedin_url != ''")
    if f.get("require_industry"):
        clauses.append("industry IS NOT NULL")
    where = " AND ".join(clauses) if clauses else "TRUE"
    return where, params


def facets():
    global _facets_cache
    if _facets_cache is None:
        inds = q("SELECT industry, count(*) n FROM read_parquet(?) "
                 "WHERE industry IS NOT NULL GROUP BY 1 ORDER BY n DESC", [PARQUET])
        ctys = q("SELECT country, count(*) n FROM read_parquet(?) "
                 "WHERE country IS NOT NULL GROUP BY 1 ORDER BY n DESC", [PARQUET])
        total = q("SELECT count(*) FROM read_parquet(?)", [PARQUET])[0][0]
        _facets_cache = {
            "total": total,
            "industries": [{"v": r[0], "n": r[1]} for r in inds],
            "countries": [{"v": r[0], "n": r[1]} for r in ctys],
            "sizes": SIZE_ORDER,
        }
    return _facets_cache


def query(f):
    where, params = build_where(f)
    base = f"FROM read_parquet('{PARQUET}') WHERE {where}"
    count = q(f"SELECT count(*) {base}", params)[0][0]
    top_ind = q(f"SELECT coalesce(industry,'(none)'), count(*) n {base} "
                "GROUP BY 1 ORDER BY n DESC LIMIT 12", params)
    top_cty = q(f"SELECT coalesce(country,'(none)'), count(*) n {base} "
                "GROUP BY 1 ORDER BY n DESC LIMIT 12", params)
    sizes = dict(q(f"SELECT coalesce(size,'(none)'), count(*) {base} GROUP BY 1", params))
    return {
        "count": count,
        "top_industries": [{"v": r[0], "n": r[1]} for r in top_ind],
        "top_countries": [{"v": r[0], "n": r[1]} for r in top_cty],
        "sizes": [{"v": s, "n": sizes.get(s, 0)} for s in SIZE_ORDER + ["(none)"]],
    }


def export(f):
    where, params = build_where(f)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(EXPORT_DIR, f"tam_export_{stamp}.csv")
    with _lock:
        _con.execute(
            f"COPY (SELECT id, name, website, founded, size, locality, region, "
            f"country, industry, linkedin_url FROM read_parquet('{PARQUET}') "
            f"WHERE {where}) TO '{path}' (FORMAT CSV, HEADER)", params)
        n = _con.execute(f"SELECT count(*) FROM read_parquet('{PARQUET}') WHERE {where}",
                         params).fetchone()[0]
    return {"path": path, "rows": n}


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
        elif self.path == "/api/facets":
            self._send(200, facets())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        f = json.loads(self.rfile.read(n) or b"{}")
        try:
            if self.path == "/api/query":
                self._send(200, query(f))
            elif self.path == "/api/export":
                self._send(200, export(f))
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:  # surface errors to the UI instead of a hung request
            self._send(500, {"error": str(e)})


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>TAM Curator</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; margin: 0; }
  body { background:#0d0f12; color:#e6e8eb; font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; padding:24px; }
  h1 { font-size:16px; letter-spacing:.12em; text-transform:uppercase; color:#9aa3ad; }
  #count { font-size:44px; font-weight:700; margin:8px 0 2px; font-variant-numeric:tabular-nums; }
  #countSub { color:#9aa3ad; margin-bottom:20px; }
  .grid { display:grid; grid-template-columns:280px 280px 1fr; gap:16px; align-items:start; }
  .panel { background:#14171c; border:1px solid #232830; border-radius:8px; padding:14px; }
  .panel h2 { font-size:11px; letter-spacing:.1em; text-transform:uppercase; color:#9aa3ad; margin-bottom:10px; }
  .list { max-height:280px; overflow-y:auto; }
  label.row { display:flex; gap:8px; align-items:center; padding:2px 0; cursor:pointer; color:#c9ced4; }
  label.row .n { margin-left:auto; color:#6b7480; font-size:12px; }
  input[type=text], input[type=number] { width:100%; background:#0d0f12; border:1px solid #2a303a; border-radius:5px; color:#e6e8eb; padding:6px 8px; margin-bottom:8px; font:inherit; }
  .inline { display:flex; gap:8px; }
  table { width:100%; border-collapse:collapse; margin-bottom:14px; }
  td { padding:2px 4px; border-bottom:1px solid #1c2129; }
  td.n { text-align:right; color:#9aa3ad; font-variant-numeric:tabular-nums; }
  button { background:#e6e8eb; color:#0d0f12; border:0; border-radius:6px; padding:10px 16px; font:inherit; font-weight:700; cursor:pointer; }
  button:disabled { opacity:.4; cursor:default; }
  #exportResult { margin-top:10px; color:#8fd18f; word-break:break-all; }
  .muted { color:#6b7480; font-size:12px; }
  .chips { margin:10px 0 0; }
  .chip { display:inline-block; background:#1d232c; border:1px solid #2a303a; border-radius:20px; padding:2px 10px; margin:2px 4px 2px 0; font-size:12px; color:#c9ced4; }
</style></head><body>
<h1>TAMage · TAM Curator</h1>
<div id="count">…</div>
<div id="countSub">companies match the current filters</div>
<div class="grid">
  <div class="panel">
    <h2>Industry</h2>
    <input type="text" id="indSearch" placeholder="search industries…">
    <div class="list" id="indList"></div>
  </div>
  <div class="panel">
    <h2>Country</h2>
    <input type="text" id="ctySearch" placeholder="search countries…">
    <div class="list" id="ctyList"></div>
    <h2 style="margin-top:14px">Region / locality contains</h2>
    <input type="text" id="regionContains" placeholder="e.g. california">
  </div>
  <div class="panel">
    <h2>Size</h2><div id="sizeList" style="margin-bottom:12px"></div>
    <h2>Founded</h2>
    <div class="inline">
      <input type="number" id="foundedMin" placeholder="min year">
      <input type="number" id="foundedMax" placeholder="max year">
    </div>
    <h2>Data presence</h2>
    <label class="row"><input type="checkbox" id="reqWebsite" checked> has website domain</label>
    <label class="row"><input type="checkbox" id="reqLinkedin"> has LinkedIn</label>
    <label class="row"><input type="checkbox" id="reqIndustry"> has industry</label>
    <div style="margin-top:16px">
      <button id="exportBtn">Export CSV</button>
      <div id="exportResult"></div>
      <div class="muted" style="margin-top:6px">Exports to dataset/exports/</div>
    </div>
    <div class="chips" id="chips"></div>
  </div>
</div>
<div class="grid" style="margin-top:16px; grid-template-columns:1fr 1fr 1fr;">
  <div class="panel"><h2>Top industries in slice</h2><table id="tblInd"></table></div>
  <div class="panel"><h2>Top countries in slice</h2><table id="tblCty"></table></div>
  <div class="panel"><h2>Size breakdown in slice</h2><table id="tblSize"></table></div>
</div>
<script>
const sel = { industries:new Set(), countries:new Set(), sizes:new Set() };
let FACETS = null, timer = null;

const fmt = n => n.toLocaleString();
const el = id => document.getElementById(id);

function filters() {
  return {
    industries:[...sel.industries], countries:[...sel.countries], sizes:[...sel.sizes],
    region_contains: el('regionContains').value.trim(),
    founded_min: el('foundedMin').value, founded_max: el('foundedMax').value,
    require_website: el('reqWebsite').checked,
    require_linkedin: el('reqLinkedin').checked,
    require_industry: el('reqIndustry').checked,
  };
}

function renderChecklist(container, items, set, filterText) {
  const ft = (filterText||'').toLowerCase();
  container.innerHTML = '';
  let shown = 0;
  for (const it of items) {
    if (ft && !it.v.toLowerCase().includes(ft)) continue;
    if (++shown > 400) break;
    const l = document.createElement('label'); l.className = 'row';
    const c = document.createElement('input'); c.type = 'checkbox'; c.checked = set.has(it.v);
    c.onchange = () => { c.checked ? set.add(it.v) : set.delete(it.v); refresh(); };
    const s = document.createElement('span'); s.textContent = it.v;
    const n = document.createElement('span'); n.className = 'n'; n.textContent = fmt(it.n ?? 0);
    l.append(c, s, n); container.append(l);
  }
}

function renderTable(t, rows) {
  t.innerHTML = rows.map(r =>
    `<tr><td>${r.v}</td><td class="n">${fmt(r.n)}</td></tr>`).join('');
}

function renderChips() {
  const parts = [...sel.industries, ...sel.countries, ...sel.sizes];
  el('chips').innerHTML = parts.map(p => `<span class="chip">${p}</span>`).join('');
}

async function refresh() {
  renderChips();
  clearTimeout(timer);
  timer = setTimeout(async () => {
    el('count').style.opacity = .4;
    const r = await fetch('/api/query', {method:'POST', body:JSON.stringify(filters())});
    const d = await r.json();
    el('count').textContent = fmt(d.count);
    el('count').style.opacity = 1;
    renderTable(el('tblInd'), d.top_industries);
    renderTable(el('tblCty'), d.top_countries);
    renderTable(el('tblSize'), d.sizes);
  }, 250);
}

el('exportBtn').onclick = async () => {
  el('exportBtn').disabled = true;
  el('exportResult').textContent = 'exporting…';
  const r = await fetch('/api/export', {method:'POST', body:JSON.stringify(filters())});
  const d = await r.json();
  el('exportResult').textContent = d.error ? ('error: ' + d.error)
    : `${fmt(d.rows)} rows → ${d.path}`;
  el('exportBtn').disabled = false;
};

['regionContains','foundedMin','foundedMax'].forEach(id => el(id).oninput = refresh);
['reqWebsite','reqLinkedin','reqIndustry'].forEach(id => el(id).onchange = refresh);
el('indSearch').oninput = () => renderChecklist(el('indList'), FACETS.industries, sel.industries, el('indSearch').value);
el('ctySearch').oninput = () => renderChecklist(el('ctyList'), FACETS.countries, sel.countries, el('ctySearch').value);

(async () => {
  FACETS = await (await fetch('/api/facets')).json();
  renderChecklist(el('indList'), FACETS.industries, sel.industries);
  renderChecklist(el('ctyList'), FACETS.countries, sel.countries);
  el('sizeList').innerHTML = '';
  renderChecklist(el('sizeList'), FACETS.sizes.map(s => ({v:s, n:null})), sel.sizes);
  refresh();
})();
</script>
</body></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if not os.path.exists(PARQUET):
        raise SystemExit(f"missing {PARQUET} — run the JSONL->Parquet conversion first")
    facets()  # warm the cache before serving
    print(f"TAM Curator on http://localhost:{args.port} — {facets()['total']:,} companies")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
