"""HTML + CSV review report from a Store."""

from __future__ import annotations

import csv
import html
import json
from collections import Counter
from pathlib import Path

from vqs.store import PAIR_COLUMNS, VIDEO_COLUMNS, Store


def write_report(store: Store, report_dir) -> Path:
    """Write ``index.html``, ``videos.csv``, and ``pairs.csv``. Returns index path."""
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)

    videos = store.list_videos()
    pairs = store.list_pairs()

    videos_csv = out / "videos.csv"
    pairs_csv = out / "pairs.csv"
    _write_csv(videos_csv, videos, VIDEO_COLUMNS)
    _write_csv(pairs_csv, pairs, PAIR_COLUMNS)

    index = out / "index.html"
    index.write_text(_render_html(videos, pairs, out), encoding="utf-8")
    return index


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in columns})


def _csv_cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    return value


def _rel_sheet(path_value, report_dir: Path) -> str:
    if not path_value:
        return ""
    try:
        sheet = Path(path_value)
        if sheet.is_absolute():
            return str(sheet.relative_to(report_dir.resolve())).replace("\\", "/")
        return str(sheet).replace("\\", "/")
    except (ValueError, OSError):
        return str(path_value).replace("\\", "/")


def _fmt(value, digits=2):
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return ""
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.{digits}f}"


def _esc(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _video_payload(row: dict, report_dir: Path) -> dict:
    return {
        "path": row.get("path") or "",
        "filename": row.get("filename") or "",
        "folder": row.get("folder") or "",
        "decision": row.get("decision") or "",
        "confidence": row.get("confidence"),
        "reason": row.get("reason") or "",
        "health_status": row.get("health_status") or "",
        "health_first_error": row.get("health_first_error") or "",
        "width": row.get("width"),
        "height": row.get("height"),
        "avg_blur": row.get("avg_blur"),
        "avg_block": row.get("avg_block"),
        "avg_noise": row.get("avg_noise"),
        "nr_iqa_median": row.get("nr_iqa_median"),
        "issue_flags": row.get("issue_flags") or "",
        "contact_sheet": _rel_sheet(row.get("contact_sheet"), report_dir),
        "error": row.get("error") or "",
        "duration_hms": row.get("duration_hms") or "",
        "codec": row.get("codec") or "",
    }


def _pair_payload(row: dict, report_dir: Path) -> dict:
    return {
        "original_filename": row.get("original_filename") or "",
        "other_filename": row.get("other_filename") or "",
        "folder": row.get("folder") or "",
        "verdict": row.get("verdict") or "",
        "action": row.get("action") or "",
        "confidence": row.get("confidence"),
        "blur_vs_base": row.get("blur_vs_base"),
        "ssim": row.get("ssim"),
        "lpips_native": row.get("lpips_native"),
        "nr_uplift": row.get("nr_uplift"),
        "note": row.get("note") or "",
        "contact_sheet": _rel_sheet(row.get("contact_sheet"), report_dir),
        "original_path": row.get("original_path") or "",
        "other_path": row.get("other_path") or "",
    }


def _render_html(videos: list[dict], pairs: list[dict], report_dir: Path) -> str:
    decisions = Counter(v.get("decision") or "" for v in videos)
    health = Counter(v.get("health_status") or "" for v in videos)
    actions = Counter(p.get("action") or "" for p in pairs)

    video_rows = [_video_payload(row, report_dir) for row in videos]
    pair_rows = [_pair_payload(row, report_dir) for row in pairs]
    payload = json.dumps({"videos": video_rows, "pairs": pair_rows}, ensure_ascii=False)

    summary_bits = [
        f"<span><strong>{len(videos)}</strong> videos</span>",
        f"<span><strong>{len(pairs)}</strong> pairs</span>",
        f"<span>topaz_candidate: <strong>{decisions.get('topaz_candidate', 0)}</strong></span>",
        f"<span>health failed: <strong>{health.get('failed', 0)}</strong></span>",
    ]
    for key, count in actions.most_common():
        if key:
            summary_bits.append(f"<span>{_esc(key)}: <strong>{count}</strong></span>")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Video Quality Scanner report</title>
<style>
:root {{
  --bg: #12151a;
  --panel: #1a1f27;
  --ink: #e8ecf1;
  --muted: #9aa3af;
  --line: #2a3140;
  --accent: #3d9a78;
  --warn: #c9a227;
  --bad: #c45c5c;
  --ok: #5a9fd4;
  --mono: "Cascadia Mono", "Consolas", "Segoe UI Mono", monospace;
  --sans: "Segoe UI", "Helvetica Neue", sans-serif;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: var(--sans);
  background: radial-gradient(1200px 600px at 10% -10%, #1e2a24 0%, var(--bg) 55%);
  color: var(--ink);
  min-height: 100vh;
}}
header {{
  padding: 1.25rem 1.5rem 0.75rem;
  border-bottom: 1px solid var(--line);
}}
header h1 {{
  margin: 0 0 0.35rem;
  font-size: 1.35rem;
  font-weight: 650;
  letter-spacing: 0.01em;
}}
.summary {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem 1.25rem;
  color: var(--muted);
  font-size: 0.9rem;
}}
.tabs {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
  padding: 0.85rem 1.5rem 0;
}}
.tabs button {{
  appearance: none;
  border: 1px solid var(--line);
  background: var(--panel);
  color: var(--ink);
  padding: 0.45rem 0.85rem;
  border-radius: 6px;
  cursor: pointer;
  font: inherit;
}}
.tabs button.active {{
  border-color: var(--accent);
  background: #1f332c;
}}
.toolbar {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
  align-items: center;
  padding: 0.85rem 1.5rem;
}}
.toolbar input[type="search"] {{
  flex: 1 1 220px;
  min-width: 180px;
  background: var(--panel);
  border: 1px solid var(--line);
  color: var(--ink);
  border-radius: 6px;
  padding: 0.5rem 0.7rem;
  font: inherit;
}}
.toolbar button, .toolbar a.button {{
  appearance: none;
  border: 1px solid var(--line);
  background: #243049;
  color: var(--ink);
  padding: 0.45rem 0.8rem;
  border-radius: 6px;
  cursor: pointer;
  text-decoration: none;
  font: inherit;
}}
main {{ padding: 0 1.5rem 2rem; }}
table {{
  width: 100%;
  border-collapse: collapse;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
  font-size: 0.9rem;
}}
th, td {{
  padding: 0.55rem 0.65rem;
  border-bottom: 1px solid var(--line);
  text-align: left;
  vertical-align: top;
}}
th {{
  color: var(--muted);
  font-weight: 600;
  background: #151a22;
  position: sticky;
  top: 0;
  cursor: pointer;
  user-select: none;
}}
tr:hover td {{ background: #202733; }}
.badge {{
  display: inline-block;
  padding: 0.1rem 0.45rem;
  border-radius: 999px;
  font-size: 0.78rem;
  border: 1px solid var(--line);
}}
.badge.topaz_candidate, .badge.keep_processed {{ border-color: var(--accent); color: #9fe0c5; }}
.badge.manual_review, .badge.keep_original {{ border-color: var(--warn); color: #f0d989; }}
.badge.repair_or_remux, .badge.failed {{ border-color: var(--bad); color: #f0a8a8; }}
.badge.ok, .badge.skip_or_ok {{ border-color: var(--ok); color: #a8d4f0; }}
.path {{ font-family: var(--mono); font-size: 0.78rem; color: var(--muted); word-break: break-all; }}
.thumb {{ max-width: 160px; max-height: 90px; border-radius: 4px; border: 1px solid var(--line); }}
.empty {{ color: var(--muted); padding: 1.5rem; }}
.panel {{ display: none; }}
.panel.active {{ display: block; }}
@media (max-width: 720px) {{
  th:nth-child(n+6), td:nth-child(n+6) {{ display: none; }}
}}
</style>
</head>
<body>
<header>
  <h1>Video Quality Scanner</h1>
  <div class="summary">{"".join(summary_bits)}</div>
</header>
<nav class="tabs" id="tabs">
  <button type="button" data-tab="topaz" class="active">Topaz candidates</button>
  <button type="button" data-tab="pairs">Keep processed / Keep original</button>
  <button type="button" data-tab="corrupt">Corrupt / repair</button>
  <button type="button" data-tab="all">All videos</button>
  <button type="button" data-tab="export">Export filtered CSV</button>
</nav>
<div class="toolbar">
  <input id="filter" type="search" placeholder="Filter by path, decision, reason…" />
  <a class="button" href="videos.csv">videos.csv</a>
  <a class="button" href="pairs.csv">pairs.csv</a>
  <button type="button" id="download-filtered">Download filtered CSV</button>
</div>
<main>
  <section class="panel active" id="panel-topaz"></section>
  <section class="panel" id="panel-pairs"></section>
  <section class="panel" id="panel-corrupt"></section>
  <section class="panel" id="panel-all"></section>
  <section class="panel" id="panel-export">
    <p class="empty">Use the filter box, then <strong>Download filtered CSV</strong> to export the visible Topaz / All / Corrupt rows. Full dumps: <a href="videos.csv">videos.csv</a>, <a href="pairs.csv">pairs.csv</a>.</p>
  </section>
</main>
<script id="vqs-data" type="application/json">{_esc(payload)}</script>
<script>
const DATA = JSON.parse(document.getElementById("vqs-data").textContent);
let activeTab = "topaz";
let sortKey = "confidence";
let sortDir = -1;

function fmt(v, digits=2) {{
  if (v === null || v === undefined || v === "") return "";
  const n = Number(v);
  if (!Number.isFinite(n)) return String(v);
  return Number.isInteger(n) ? String(n) : n.toFixed(digits);
}}

function badge(text, cls) {{
  if (!text) return "";
  return `<span class="badge ${{cls || text}}">${{escapeHtml(text)}}</span>`;
}}

function escapeHtml(s) {{
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}}

function sheetCell(path) {{
  if (!path) return "";
  return `<a href="${{escapeHtml(path)}}" target="_blank" rel="noopener"><img class="thumb" src="${{escapeHtml(path)}}" alt="sheet"/></a>`;
}}

function matchesFilter(row, q) {{
  if (!q) return true;
  const hay = Object.values(row).join(" ").toLowerCase();
  return hay.includes(q);
}}

function sortRows(rows) {{
  const key = sortKey;
  const dir = sortDir;
  return rows.slice().sort((a, b) => {{
    const av = a[key], bv = b[key];
    const an = Number(av), bn = Number(bv);
    if (Number.isFinite(an) && Number.isFinite(bn)) return (an - bn) * dir;
    return String(av ?? "").localeCompare(String(bv ?? "")) * dir;
  }});
}}

function videoTable(rows) {{
  if (!rows.length) return `<p class="empty">No matching rows.</p>`;
  const head = ["confidence","decision","filename","health_status","avg_blur","avg_block","nr_iqa_median","reason","contact_sheet","path"];
  const labels = {{confidence:"Conf",decision:"Decision",filename:"File",health_status:"Health",avg_blur:"Blur",avg_block:"Block",nr_iqa_median:"NR-IQA",reason:"Reason",contact_sheet:"Sheet",path:"Path"}};
  let html = "<table><thead><tr>";
  for (const k of head) html += `<th data-sort="${{k}}">${{labels[k]}}</th>`;
  html += "</tr></thead><tbody>";
  for (const r of rows) {{
    html += `<tr>
      <td>${{fmt(r.confidence)}}</td>
      <td>${{badge(r.decision)}}</td>
      <td>${{escapeHtml(r.filename)}}<div class="path">${{escapeHtml(r.folder)}}</div></td>
      <td>${{badge(r.health_status)}}</td>
      <td>${{fmt(r.avg_blur,1)}}</td>
      <td>${{fmt(r.avg_block,2)}}</td>
      <td>${{fmt(r.nr_iqa_median,3)}}</td>
      <td>${{escapeHtml(r.reason)}}</td>
      <td>${{sheetCell(r.contact_sheet)}}</td>
      <td class="path">${{escapeHtml(r.path)}}</td>
    </tr>`;
  }}
  html += "</tbody></table>";
  return html;
}}

function pairTable(rows) {{
  if (!rows.length) return `<p class="empty">No matching pairs.</p>`;
  let html = `<table><thead><tr>
    <th data-sort="confidence">Conf</th>
    <th data-sort="action">Action</th>
    <th data-sort="verdict">Verdict</th>
    <th>Original</th>
    <th>Other</th>
    <th data-sort="ssim">SSIM</th>
    <th data-sort="blur_vs_base">Blur↑</th>
    <th data-sort="lpips_native">LPIPS</th>
    <th data-sort="nr_uplift">NR↑</th>
    <th>Note</th>
    <th>Sheet</th>
  </tr></thead><tbody>`;
  for (const r of rows) {{
    html += `<tr>
      <td>${{fmt(r.confidence)}}</td>
      <td>${{badge(r.action)}}</td>
      <td>${{escapeHtml(r.verdict)}}</td>
      <td>${{escapeHtml(r.original_filename)}}</td>
      <td>${{escapeHtml(r.other_filename)}}</td>
      <td>${{fmt(r.ssim,3)}}</td>
      <td>${{fmt(r.blur_vs_base,2)}}</td>
      <td>${{fmt(r.lpips_native,3)}}</td>
      <td>${{fmt(r.nr_uplift,3)}}</td>
      <td>${{escapeHtml(r.note)}}</td>
      <td>${{sheetCell(r.contact_sheet)}}</td>
    </tr>`;
  }}
  html += "</tbody></table>";
  return html;
}}

function currentVideoRows() {{
  const q = document.getElementById("filter").value.trim().toLowerCase();
  let rows = DATA.videos;
  if (activeTab === "topaz") rows = rows.filter(r => r.decision === "topaz_candidate");
  else if (activeTab === "corrupt") rows = rows.filter(r =>
    r.health_status === "failed" || r.decision === "repair_or_remux" || r.health_status === "errors");
  rows = rows.filter(r => matchesFilter(r, q));
  return sortRows(rows);
}}

function currentPairRows() {{
  const q = document.getElementById("filter").value.trim().toLowerCase();
  let rows = DATA.pairs.filter(r =>
    !r.action || r.action === "keep_processed" || r.action === "keep_original" || r.action === "manual_review");
  rows = rows.filter(r => matchesFilter(r, q));
  return sortRows(rows);
}}

function render() {{
  document.getElementById("panel-topaz").innerHTML = videoTable(
    sortRows(DATA.videos.filter(r => r.decision === "topaz_candidate").filter(r => matchesFilter(r, document.getElementById("filter").value.trim().toLowerCase())))
  );
  document.getElementById("panel-pairs").innerHTML = pairTable(currentPairRows());
  document.getElementById("panel-corrupt").innerHTML = videoTable(
    sortRows(DATA.videos.filter(r =>
      r.health_status === "failed" || r.decision === "repair_or_remux" || r.health_status === "errors"
    ).filter(r => matchesFilter(r, document.getElementById("filter").value.trim().toLowerCase())))
  );
  document.getElementById("panel-all").innerHTML = videoTable(
    sortRows(DATA.videos.filter(r => matchesFilter(r, document.getElementById("filter").value.trim().toLowerCase())))
  );
  document.querySelectorAll("th[data-sort]").forEach(th => {{
    th.addEventListener("click", () => {{
      const key = th.getAttribute("data-sort");
      if (sortKey === key) sortDir *= -1;
      else {{ sortKey = key; sortDir = key === "filename" || key === "path" ? 1 : -1; }}
      render();
    }});
  }});
}}

document.getElementById("tabs").addEventListener("click", (ev) => {{
  const btn = ev.target.closest("button[data-tab]");
  if (!btn) return;
  activeTab = btn.getAttribute("data-tab");
  document.querySelectorAll(".tabs button").forEach(b => b.classList.toggle("active", b === btn));
  document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
  document.getElementById("panel-" + activeTab).classList.add("active");
}});

document.getElementById("filter").addEventListener("input", render);

document.getElementById("download-filtered").addEventListener("click", () => {{
  const rows = activeTab === "pairs" ? currentPairRows() : currentVideoRows();
  if (!rows.length) return;
  const keys = Object.keys(rows[0]);
  const lines = [keys.join(",")].concat(rows.map(r => keys.map(k => {{
    const v = r[k] == null ? "" : String(r[k]);
    return /["\\n,]/.test(v) ? `"${{v.replace(/"/g, '""')}}"` : v;
  }}).join(",")));
  const blob = new Blob([lines.join("\\n")], {{type: "text/csv;charset=utf-8"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = activeTab === "pairs" ? "pairs-filtered.csv" : "videos-filtered.csv";
  a.click();
  URL.revokeObjectURL(a.href);
}});

render();
</script>
</body>
</html>
"""
