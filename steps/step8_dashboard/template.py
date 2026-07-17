"""step8_dashboard/template.py - the complete, self-contained dashboard document.

`TEMPLATE` is one HTML string with the literal token ``__PAYLOAD_JSON__`` where the
minified payload JSON is spliced in by ``render.render_dashboard``. Every section is
built client-side by the embedded vanilla JS from that single JSON blob - the Python
side builds nothing per-cluster, which keeps the output byte-for-byte deterministic
and the file free of any network dependency.

Design notes (dataviz + artifact-design skills):
- Colour roles are CSS custom properties in :root, re-stated for prefers-color-scheme
  dark; the chart body is written against roles, never raw hex (palette.md). Neutrals
  are a deliberate warm-paper ground under a cool analytical blue accent (a finance-
  report pairing, not the cream+terracotta AI default); status red/green come straight
  from palette.md's fixed status slots.
- Typography is a real modular scale on the mandated system sans (dataviz forbids a
  display/serif face, and the offline no-external-requests guarantee forbids a webfont):
  an uppercase micro-label channel, balanced headings, proportional hero figures, and
  tabular-nums reserved for aligned columns.
- Categorical cluster colour: the reference palette tops out at 8 hues and forbids
  cycling, but this universe has ~40 peer groups. Per the task's explicit mandate we
  generate a fixed HSL wheel keyed on cluster_id (golden-angle spacing, saturation/
  lightness tuned inside the skill's per-mode bands and kept legible on both grounds).
  Colour is never the sole identity channel: the scatter has a name/ticker hover
  tooltip and the Clusters index table is the legend (a swatch per row beside the short
  title). See the report for this documented deviation.
- Marks: surface-ring on scatter dots over a recessive gridded frame with dimensionless
  captions (the embedding carries no units, so no numeric ticks); top-holdings bars are
  one-hue (slot-1 blue) with rounded data-ends; the out-of-time per-quarter strip
  diverges from a 0.50 coin-flip baseline, below-0.50 bars in the critical status colour
  with an explicit label (never colour alone); text wears ink tokens only.
- Interaction: dense scatter uses a single nearest-point hover layer (not thousands of
  hit targets); tables are generic click-to-sort, numeric-aware, nulls last both ways.
  Transitions are CSS-only and disabled under prefers-reduced-motion.
"""

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fund Peer-Group Dashboard</title>
<style>
  :root {
    color-scheme: light dark;
    /* warm-paper ground under a cool analytical accent (palette.md surfaces/inks) */
    --surface-1: #fcfcfb;
    --page: #f5f5f2;
    --card: #ffffff;
    --ink-1: #0b0b0b;
    --ink-2: #52514e;
    --ink-muted: #898781;
    --grid: #e6e5de;
    --axis: #c3c2b7;
    --border: rgba(11,11,11,0.10);
    --border-strong: rgba(11,11,11,0.16);
    --series-1: #2a78d6;
    --series-1-wash: #d7e6fb;
    --accent-tint: #f0f5fd;
    --good: #006300;
    --critical: #c23a35;
    --shadow-1: 0 1px 2px rgba(11,11,11,.05), 0 1px 1px rgba(11,11,11,.03);
    --shadow-2: 0 6px 22px rgba(11,11,11,.10);
    --shadow-pop: 0 8px 28px rgba(11,11,11,.16);
    /* type scale (~1.2 modular) */
    --fs-eyebrow: .72rem;
    --fs-body: 15px;
    --fs-h3: 1.06rem;
    --fs-h2: 1.34rem;
    --fs-h1: 1.72rem;
    --fs-hero: 3rem;
    --radius: 14px;
    --radius-sm: 9px;
    --nav-h: 58px;
    --foot-h: 68px;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --surface-1: #1a1a19;
      --page: #0d0d0d;
      --card: #1c1c1b;
      --ink-1: #ffffff;
      --ink-2: #c3c2b7;
      --ink-muted: #8f8d87;
      --grid: #2c2c2a;
      --axis: #3d3d3a;
      --border: rgba(255,255,255,0.11);
      --border-strong: rgba(255,255,255,0.18);
      --series-1: #3987e5;
      --series-1-wash: #1c4a86;
      --accent-tint: #14243a;
      --good: #0ca30c;
      --critical: #e06b6b;
      --shadow-1: 0 1px 2px rgba(0,0,0,.4);
      --shadow-2: 0 8px 26px rgba(0,0,0,.5);
      --shadow-pop: 0 10px 30px rgba(0,0,0,.6);
    }
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body {
    margin: 0;
    font: var(--fs-body)/1.6 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    background: var(--page);
    color: var(--ink-1);
    padding-bottom: calc(var(--foot-h) + 1.25rem);
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
  }
  a { color: var(--series-1); }
  .eyebrow {
    font-size: var(--fs-eyebrow); font-weight: 600; letter-spacing: .07em;
    text-transform: uppercase; color: var(--ink-muted); margin: 0 0 .35rem;
  }

  /* ---- top nav + search ---- */
  nav {
    position: sticky; top: 0; z-index: 20;
    display: flex; align-items: center; gap: .3rem; flex-wrap: wrap;
    height: var(--nav-h); padding: 0 1.15rem;
    background: color-mix(in srgb, var(--surface-1) 88%, transparent);
    backdrop-filter: saturate(1.4) blur(8px);
    border-bottom: 1px solid var(--border);
  }
  nav .brand {
    font-weight: 700; margin-right: 1rem; white-space: nowrap;
    letter-spacing: -.01em; display: inline-flex; align-items: center; gap: .5rem;
  }
  nav .brand::before {
    content: ""; width: .7rem; height: .7rem; border-radius: 3px;
    background: var(--series-1); box-shadow: 0 0 0 3px var(--accent-tint);
  }
  nav a.navlink {
    text-decoration: none; color: var(--ink-2); font-size: .92rem;
    padding: .35rem .65rem; border-radius: 8px; white-space: nowrap;
    transition: background .14s ease, color .14s ease;
  }
  nav a.navlink:hover { background: var(--page); color: var(--ink-1); }
  nav a.navlink.active { color: var(--ink-1); background: var(--accent-tint); font-weight: 600; }
  .search-wrap { position: relative; margin-left: auto; }
  #search {
    font: inherit; font-size: .92rem; padding: .4rem .65rem; width: 15rem; max-width: 45vw;
    border: 1px solid var(--border-strong); border-radius: 9px;
    background: var(--card); color: var(--ink-1);
    transition: border-color .14s ease, box-shadow .14s ease;
  }
  #search:focus {
    outline: none; border-color: var(--series-1);
    box-shadow: 0 0 0 3px var(--accent-tint);
  }
  #results {
    position: absolute; right: 0; top: 2.6rem; z-index: 30;
    width: 22rem; max-width: 80vw; max-height: 60vh; overflow-y: auto;
    background: var(--card); border: 1px solid var(--border-strong); border-radius: 11px;
    box-shadow: var(--shadow-pop); padding: .35rem;
  }
  #results button {
    display: block; width: 100%; text-align: left; font: inherit; cursor: pointer;
    background: none; border: 0; color: var(--ink-1);
    padding: .45rem .55rem; border-radius: 7px; transition: background .12s ease;
  }
  #results button:hover, #results button:focus { background: var(--accent-tint); outline: none; }
  #results .r-sub { color: var(--ink-2); font-size: .82rem; }
  #results .r-empty { padding: .55rem; color: var(--ink-2); }

  /* ---- layout & type ---- */
  main { max-width: 1040px; margin: 0 auto; padding: 2rem 1.15rem 2.5rem; }
  h1 { font-size: var(--fs-h1); font-weight: 700; letter-spacing: -.02em;
    line-height: 1.15; text-wrap: balance; margin: 0 0 .5rem;
    display: flex; align-items: center; gap: .55rem; }
  h2 { font-size: var(--fs-h2); font-weight: 650; letter-spacing: -.01em;
    text-wrap: balance; margin: 2.4rem 0 .85rem; }
  h3 { font-size: var(--fs-h3); font-weight: 650; margin: 1.6rem 0 .55rem; }
  p { margin: .55rem 0; }
  p.lead { color: var(--ink-2); max-width: 65ch; }
  .muted { color: var(--ink-2); }

  .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.35rem 1.5rem; margin: 1.25rem 0;
    box-shadow: var(--shadow-1);
  }

  /* ---- stat tiles ---- */
  .tiles { display: flex; flex-wrap: wrap; gap: .8rem; margin: 1.15rem 0; }
  .tile {
    flex: 1 1 8.5rem; min-width: 8.5rem;
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: .9rem 1rem;
    box-shadow: var(--shadow-1);
    transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease;
  }
  .tile:hover { transform: translateY(-2px); box-shadow: var(--shadow-2);
    border-color: var(--border-strong); }
  .tile .label {
    color: var(--ink-muted); font-size: var(--fs-eyebrow); font-weight: 600;
    letter-spacing: .05em; text-transform: uppercase;
  }
  .tile .value { font-size: 1.55rem; font-weight: 650; margin-top: .3rem;
    line-height: 1.1; letter-spacing: -.01em; }
  .tile.accent { border-color: color-mix(in srgb, var(--series-1) 35%, var(--border)); }
  .tile.accent .value { color: var(--series-1); }
  .hero { font-size: var(--fs-hero); font-weight: 700; line-height: 1.05;
    letter-spacing: -.03em; margin: .2rem 0 .1rem; }

  /* ---- tables ---- */
  .tbl-scroll { overflow-x: auto; border: 1px solid var(--border);
    border-radius: var(--radius-sm); box-shadow: var(--shadow-1); }
  table { border-collapse: collapse; width: 100%; font-size: .9rem; }
  th, td { padding: .55rem .75rem; text-align: left;
    border-bottom: 1px solid var(--border); }
  tbody tr:last-child td { border-bottom: 0; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  thead th {
    color: var(--ink-muted); font-weight: 600; white-space: nowrap;
    font-size: var(--fs-eyebrow); letter-spacing: .04em; text-transform: uppercase;
    background: var(--surface-1); position: sticky; top: var(--nav-h); z-index: 1;
  }
  table.sortable thead th.sortable-th { cursor: pointer; user-select: none;
    transition: color .12s ease; }
  table.sortable thead th.sortable-th:hover { color: var(--ink-1); }
  th[aria-sort="ascending"]::after { content: " \2191"; color: var(--series-1); }
  th[aria-sort="descending"]::after { content: " \2193"; color: var(--series-1); }
  tbody tr { transition: background .12s ease; }
  tbody tr:hover { background: var(--accent-tint); }
  tbody tr.hit { background: var(--series-1-wash); }
  .swatch {
    display: inline-block; width: .8rem; height: .8rem; border-radius: 3px;
    margin-right: .5rem; vertical-align: -1px;
    box-shadow: 0 0 0 2px var(--surface-1), 0 0 0 3px var(--border);
  }
  a.rowlink { text-decoration: none; color: inherit; font-weight: 550; }
  a.rowlink:hover { text-decoration: underline; text-decoration-color: var(--series-1); }
  sup.foot a { text-decoration: none; color: var(--series-1); font-weight: 600; }

  /* ---- scatter ---- */
  .scatter-wrap { position: relative; margin: 1rem 0; }
  svg.scatter { width: 100%; height: auto; display: block;
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: var(--radius); box-shadow: var(--shadow-1); }
  svg.scatter .grid-line { stroke: var(--grid); stroke-width: 1; }
  svg.scatter .frame { fill: none; stroke: var(--axis); stroke-width: 1; }
  svg.scatter .axis-cap { fill: var(--ink-muted); font-size: 13px;
    font-family: system-ui, sans-serif; letter-spacing: .04em; }
  .tooltip {
    position: absolute; pointer-events: none; z-index: 5;
    background: var(--card); color: var(--ink-1);
    border: 1px solid var(--border-strong); border-radius: 9px;
    padding: .45rem .6rem; font-size: .82rem; max-width: 16rem;
    box-shadow: var(--shadow-pop); opacity: 0; transition: opacity .09s ease;
  }
  .tooltip .t-val { font-weight: 650; }
  .tooltip .t-sub { color: var(--ink-2); margin-top: .1rem; }

  /* ---- top-holdings bars ---- */
  .bars { margin: .6rem 0; display: flex; flex-direction: column; gap: .45rem; }
  .bar-row { display: grid; grid-template-columns: 12rem 1fr auto; gap: .7rem;
    align-items: center; }
  .bar-row .issuer { color: var(--ink-1); overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap; font-size: .9rem; }
  .bar-track { background: var(--page); border-radius: 6px; height: 16px; overflow: hidden;
    box-shadow: inset 0 0 0 1px var(--border); }
  .bar-fill { height: 100%; background: var(--series-1);
    border-radius: 0 5px 5px 0; transition: width .3s ease; }
  .bar-row .wt { color: var(--ink-2); font-variant-numeric: tabular-nums;
    font-size: .85rem; font-weight: 550; }
  @media (max-width: 620px) { .bar-row { grid-template-columns: 8rem 1fr auto; } }

  .narrative { max-width: 68ch; }
  .identity-meta { color: var(--ink-2); margin: .2rem 0 0; }

  /* ---- out-of-time reality panel ---- */
  .oot {
    background: linear-gradient(180deg, var(--accent-tint), var(--card) 62%);
    border: 1px solid color-mix(in srgb, var(--series-1) 28%, var(--border));
    box-shadow: var(--shadow-2);
  }
  .oot .eyebrow { color: var(--series-1); }
  .qstrip { width: 100%; height: auto; display: block; margin: .35rem 0 .2rem; }
  .qstrip .base-line { stroke: var(--axis); stroke-width: 1.5; stroke-dasharray: 4 3; }
  .qstrip .base-cap, .qstrip .q-lab { fill: var(--ink-muted); font-size: 12px;
    font-family: system-ui, sans-serif; }
  .qstrip .v-lab { fill: var(--ink-2); font-size: 12px; font-weight: 600;
    font-family: system-ui, sans-serif; font-variant-numeric: tabular-nums; }
  .oot-note { color: var(--ink-2); font-size: .88rem; margin: .5rem 0 0;
    display: flex; align-items: flex-start; gap: .45rem; }
  .oot-note .flag { color: var(--critical); font-weight: 700; }
  .oot-retired {
    border: 1px solid var(--border); border-radius: var(--radius-sm);
    padding: .65rem .8rem; margin: 0 0 .85rem; color: var(--ink-2);
  }
  .oot-retired .oot-retired-head { margin: 0 0 .3rem; font-weight: 700; color: var(--ink-2); }
  .oot-retired p { margin: 0 0 .3rem; }
  .oot-retired p:last-child { margin-bottom: 0; }

  /* ---- footer ---- */
  footer#disclaimer {
    position: fixed; left: 0; right: 0; bottom: 0; z-index: 15;
    background: color-mix(in srgb, var(--surface-1) 92%, transparent);
    backdrop-filter: blur(6px);
    border-top: 1px solid var(--border);
    color: var(--ink-2); font-size: .78rem; line-height: 1.4;
    padding: .6rem 1.15rem; text-align: center;
    max-height: var(--foot-h); overflow-y: auto;
  }
  @media (prefers-reduced-motion: reduce) {
    html { scroll-behavior: auto; }
    * { transition: none !important; }
  }
</style>
</head>
<body>
<nav>
  <span class="brand">Fund Peer Groups</span>
  <a class="navlink" href="#overview">Overview</a>
  <a class="navlink" href="#clusters">Clusters</a>
  <a class="navlink" href="#allocation">Target-Date &amp; Allocation</a>
  <span class="search-wrap">
    <input id="search" type="search" placeholder="Search funds by name or ticker&hellip;"
           autocomplete="off" aria-label="Search funds by name or ticker">
    <div id="results" hidden></div>
  </span>
</nav>
<main id="app"></main>
<footer id="disclaimer"></footer>
<script id="payload" type="application/json">__PAYLOAD_JSON__</script>
<script>
"use strict";
const DATA = JSON.parse(document.getElementById("payload").textContent);
const SVGNS = "http://www.w3.org/2000/svg";

/* ---------- formatting ---------- */
function fmtPct(x) { return (x === null || x === undefined) ? "—" : (x * 100).toFixed(1) + "%"; }
function fmtSharpe(x) { return (x === null || x === undefined) ? "—" : Number(x).toFixed(2); }
function fmtMoney(x) {
  return (x === null || x === undefined)
    ? "—" : Number(x).toLocaleString("en-US", { maximumFractionDigits: 0 });
}
function fmtScore(x) { return (x === null || x === undefined) ? "—" : Number(x).toFixed(3); }

/* ---------- categorical cluster colour (fixed HSL wheel keyed on cluster_id) ---------- */
const DARK = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
function clusterColor(id) {
  const hue = (id * 137.508) % 360;                 // golden-angle spacing, fixed by id
  // Saturation/lightness tuned per mode: muted enough to read as one disciplined system
  // across ~40 hues, light enough on dark ground and dark enough on light ground to stay
  // legible. Identity never rests on colour alone (legend swatch + hover tooltip).
  const s = DARK ? 58 : 55;
  const l = DARK ? 62 : 45;
  return "hsl(" + hue.toFixed(1) + " " + s + "% " + l + "%)";
}

/* ---------- small DOM helpers ---------- */
function el(tag, attrs, children) {
  const n = document.createElement(tag);
  if (attrs) for (const k in attrs) {
    if (k === "class") n.className = attrs[k];
    else if (k === "text") n.textContent = attrs[k];
    else if (k === "html") { /* intentionally unused - we never inject untrusted HTML */ }
    else n.setAttribute(k, attrs[k]);
  }
  if (children) for (const c of children) if (c) n.appendChild(c);
  return n;
}
function txt(tag, s, cls) { const n = el(tag, cls ? { class: cls } : null); n.textContent = s; return n; }

/* ---------- lookups ---------- */
const SID_INFO = {};       // sid -> {name, ticker}
const CLUSTER_TITLE = {};  // cluster_id -> short_title
const SID_CLUSTER = {};    // sid -> cluster_id (from coords)
for (const c of DATA.clusters) {
  CLUSTER_TITLE[c.cluster_id] = c.short_title;
  for (const m of c.members) SID_INFO[m.sid] = { name: m.name, ticker: m.ticker };
}
for (const v of DATA.allocation) for (const m of v.members) SID_INFO[m.sid] = { name: m.name, ticker: m.ticker };
for (const p of DATA.coords) SID_CLUSTER[p.sid] = p.cluster;

/* ---------- generic sortable table ---------- */
function makeSortable(table) {
  const ths = table.tHead.rows[0].cells;
  for (let idx = 0; idx < ths.length; idx++) {
    const th = ths[idx];
    if (th.dataset.noSort !== undefined) continue;
    th.classList.add("sortable-th");
    th.setAttribute("aria-sort", "none");
    th.addEventListener("click", () => {
      const asc = th.dataset.dir !== "asc";
      for (const h of ths) { delete h.dataset.dir; h.setAttribute("aria-sort", "none"); }
      th.dataset.dir = asc ? "asc" : "desc";
      th.setAttribute("aria-sort", asc ? "ascending" : "descending");
      const tbody = table.tBodies[0];
      const rows = Array.from(tbody.rows);
      rows.sort((ra, rb) => {
        const a = ra.cells[idx].dataset.sort, b = rb.cells[idx].dataset.sort;
        const aNull = (a === undefined || a === ""), bNull = (b === undefined || b === "");
        if (aNull && bNull) return 0;
        if (aNull) return 1;               // nulls last - checked BEFORE the asc/desc flip
        if (bNull) return -1;
        const na = parseFloat(a), nb = parseFloat(b);
        let cmp;
        if (!isNaN(na) && !isNaN(nb)) cmp = na - nb;
        else cmp = String(a).localeCompare(String(b));
        return asc ? cmp : -cmp;
      });
      for (const r of rows) tbody.appendChild(r);
    });
  }
}

/* ---------- table-cell builders ---------- */
function numCell(value, formatter) {
  const td = el("td", { class: "num" });
  if (value === null || value === undefined) { td.textContent = "—"; td.dataset.sort = ""; }
  else { td.textContent = formatter(value); td.dataset.sort = String(value); }
  return td;
}
function textCell(value) {
  const td = el("td");
  const s = (value === null || value === undefined) ? "" : String(value);
  td.textContent = s || "—";
  td.dataset.sort = s.toLowerCase();
  return td;
}

/* ---------- member table (clusters & allocation share this) ---------- */
function memberTable(members, withProb) {
  const table = el("table", { class: "sortable" });
  const thead = el("thead");
  const htr = el("tr");
  const cols = [
    ["Fund", false], ["Ticker", false], ["Net assets", true], ["Sharpe", true],
    ["Volatility", true], ["Max drawdown", true], ["Cum. return", true],
  ];
  for (const [label, num] of cols) {
    const th = el("th", num ? { class: "num" } : null);
    th.textContent = label;
    htr.appendChild(th);
  }
  if (withProb) {
    const th = el("th", { class: "num" });
    th.textContent = "Lag prob. ";
    th.title = "Estimated probability that this fund's total return over the NEXT QUARTER "
             + "(the quarter after " + DATA.universe.latest_quarter + ") finishes below the "
             + "median of its 10 most-similar peer funds. One quarter only - says nothing "
             + "about longer horizons or absolute returns. A statistical estimate from past "
             + "data - not advice. See disclaimer.";
    const sup = el("sup", { class: "foot" });
    const a = el("a", { href: "#disclaimer", "aria-label": "See disclaimer" });
    a.textContent = "†";
    a.addEventListener("click", (e) => {
      e.preventDefault();
      document.getElementById("disclaimer").scrollIntoView({ block: "nearest" });
    });
    sup.appendChild(a);
    th.appendChild(sup);
    htr.appendChild(th);
  }
  thead.appendChild(htr);
  table.appendChild(thead);

  const tbody = el("tbody");
  for (const m of members) {
    const tr = el("tr");
    tr.dataset.sid = m.sid;
    tr.appendChild(textCell(m.name));
    tr.appendChild(textCell(m.ticker));
    tr.appendChild(numCell(m.net_assets, fmtMoney));
    tr.appendChild(numCell(m.sharpe, fmtSharpe));
    tr.appendChild(numCell(m.volatility, fmtPct));
    tr.appendChild(numCell(m.max_drawdown, fmtPct));
    tr.appendChild(numCell(m.cumulative_return, fmtPct));
    if (withProb) tr.appendChild(numCell(m.probability === undefined ? null : m.probability, fmtPct));
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  const wrap = el("div", { class: "tbl-scroll" }, [table]);
  makeSortable(table);
  return wrap;
}

/* ---------- OVERVIEW ---------- */
function renderOverview() {
  const u = DATA.universe;
  const frag = document.createDocumentFragment();
  frag.appendChild(txt("h1", "Fund peer-group universe"));
  frag.appendChild(txt("p", u.n_funds.toLocaleString("en-US") + " funds", "hero"));

  const tiles = el("div", { class: "tiles" }, [
    tile("Strategy funds", u.n_strategy.toLocaleString("en-US")),
    tile("Allocation / target-date", u.n_allocation.toLocaleString("en-US")),
    tile("Quarters covered", String(u.quarters.length)),
    tile("Latest quarter", u.latest_quarter),
  ]);
  frag.appendChild(tiles);

  // "how to read this"
  const how = el("div", { class: "card" });
  how.appendChild(txt("h3", "How to read this dashboard"));
  how.appendChild(txt("p",
    "Every fund here is sorted into a peer group of funds that invest alike, based on "
    + "what they actually hold. For each group you can see its typical risk and return, "
    + "its largest shared holdings, and a plain-English summary. Where a fund carries a "
    + "“lag probability,” that number is the model's estimated chance that the fund's "
    + "total return over the next quarter (the quarter after " + DATA.universe.latest_quarter
    + ", the latest filings here) finishes below the median of its 10 most-similar peer "
    + "funds. It covers that single quarter only — not longer horizons, and not absolute "
    + "returns — and it is a statistical estimate from past data, not investment advice; "
    + "it can be wrong for any individual fund. Use it to ask questions, not to make "
    + "decisions.", "lead"));
  frag.appendChild(how);

  frag.appendChild(renderScorecard());

  // scatter
  frag.appendChild(txt("h2", "The map of funds"));
  frag.appendChild(txt("p",
    "Each dot is a fund; funds that hold similar things sit near each other, and colour "
    + "marks the peer group. Hover a dot for its name and group. Coordinates come from a "
    + "similarity embedding and carry no units of their own.", "lead"));
  frag.appendChild(renderScatter());

  return frag;
}

function tile(label, value) {
  return el("div", { class: "tile" }, [txt("div", label, "label"), txt("div", value, "value")]);
}
function tileAccent(label, value) {
  return el("div", { class: "tile accent" }, [txt("div", label, "label"), txt("div", value, "value")]);
}

/* ---------- out-of-time per-quarter strip (diverges from a 0.50 baseline) ---------- */
function barPath(x, w, top, h, r, down) {
  r = Math.max(0, Math.min(r, w / 2, h));
  if (down) {                                            // square at baseline (top), round at cap (bottom)
    return "M" + x + " " + top + " h" + w + " v" + (h - r)
      + " a" + r + " " + r + " 0 0 1 " + (-r) + " " + r
      + " h" + (-(w - 2 * r))
      + " a" + r + " " + r + " 0 0 1 " + (-r) + " " + (-r) + " z";
  }
  return "M" + x + " " + (top + h) + " v" + (-(h - r))   // square at baseline (bottom), round at cap (top)
    + " a" + r + " " + r + " 0 0 1 " + r + " " + (-r)
    + " h" + (w - 2 * r)
    + " a" + r + " " + r + " 0 0 1 " + r + " " + r
    + " v" + (h - r) + " z";
}

function renderQuarterStrip(pq) {
  const W = 720, H = 190, padT = 24, padB = 30, padL = 10, padR = 10;
  const y0 = padT, y1 = H - padB;
  let lo = 0.5, hi = 0.5;
  for (const q of pq) {
    const v = q.auc;
    if (v === null || v === undefined) continue;
    if (v < lo) lo = v; if (v > hi) hi = v;
  }
  const pad = Math.max(0.03, (hi - lo) * 0.18);
  lo -= pad; hi += pad;
  if (hi - lo < 0.12) { const mid = (hi + lo) / 2; lo = mid - 0.06; hi = mid + 0.06; }
  const sy = v => y1 - (v - lo) / (hi - lo) * (y1 - y0);
  const yBase = sy(0.5);
  const n = pq.length;
  const slot = (W - padL - padR) / n;
  const bw = Math.min(24, slot * 0.5);

  const svg = document.createElementNS(SVGNS, "svg");
  svg.setAttribute("class", "qstrip");
  svg.setAttribute("viewBox", "0 0 " + W + " " + H);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Frozen-model AUC by quarter, relative to a 0.50 coin-flip baseline");

  function sEl(tag, attrs, text) {
    const e = document.createElementNS(SVGNS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    return e;
  }

  svg.appendChild(sEl("line", { class: "base-line",
    x1: padL, y1: yBase.toFixed(1), x2: W - padR, y2: yBase.toFixed(1) }));
  svg.appendChild(sEl("text", { class: "base-cap", x: padL, y: 14,
    "text-anchor": "start" }, "0.50 — coin flip"));

  for (let i = 0; i < n; i++) {
    const q = pq[i];
    const cx = padL + slot * (i + 0.5);
    svg.appendChild(sEl("text", { class: "q-lab", x: cx.toFixed(1), y: (H - 10),
      "text-anchor": "middle" }, q.quarter));
    if (q.auc === null || q.auc === undefined) continue;
    const yv = sy(q.auc);
    const below = q.auc < 0.5;
    const top = Math.min(yv, yBase), bot = Math.max(yv, yBase);
    const h = Math.max(1.5, bot - top);
    svg.appendChild(sEl("path", { d: barPath(cx - bw / 2, bw, top, h, 4, below),
      fill: below ? "var(--critical)" : "var(--series-1)" }));
    svg.appendChild(sEl("text", { class: "v-lab", x: cx.toFixed(1),
      y: (below ? bot + 14 : top - 6).toFixed(1), "text-anchor": "middle" }, fmtScore(q.auc)));
  }
  return svg;
}

/* ---------- out-of-time reality panel (leads the scorecard; branches on the data) ---------- */
function renderOOTPanel(s) {
  const pub = s.oot_published_auc;
  const hasPub = (pub !== null && pub !== undefined);
  const pq = Array.isArray(s.oot_frozen_per_quarter) ? s.oot_frozen_per_quarter : [];
  const hasStrip = pq.length > 0;
  const retirement = s.retirement;
  if (!hasPub && !hasStrip && !retirement) return null;   // absent -> panel simply not rendered

  const card = el("div", { class: "card oot" });
  if (retirement) {
    const banner = el("div", { class: "oot-retired" });
    banner.appendChild(txt("p", "✕ MODEL RETIRED as of " + retirement.as_of,
      "oot-retired-head"));
    banner.appendChild(txt("p", retirement.statement));
    banner.appendChild(txt("p",
      "Two scorers, disclosed: the statement's trigger numbers are the deployed " +
      "retrained model's per-quarter scores; the frozen-model strip below shows 0.428 " +
      "and 0.418 for the same quarters. Both scorers were below the 0.5 coin-flip in " +
      "both quarters."));
    card.appendChild(banner);
  }
  card.appendChild(txt("p", "Out-of-time reality check", "eyebrow"));
  card.appendChild(txt("h2", "How the model held up on data it had never seen"));
  card.appendChild(txt("p",
    "The scorecard below is a backtest — measured on history the model was fitted and "
    + "tuned against. This is the harder test: predictions frozen before the outcomes "
    + "existed, then graded once the returns actually arrived.", "lead"));

  if (hasPub) {
    const backtest = s.auc;
    const frozenPooled = s.oot_frozen_pooled_auc;
    const tiles = [
      tileAccent("Published-forward AUC", fmtScore(pub)),
      tile("Backtest AUC (for contrast)", fmtScore(backtest)),
    ];
    if (frozenPooled !== null && frozenPooled !== undefined) {
      tiles.push(tile("Frozen model, rolled forward", fmtScore(frozenPooled)));
    }
    card.appendChild(el("div", { class: "tiles" }, tiles));

    // graded-against detail (only the fields that are present)
    let nInfo = "";
    const ns = s.oot_published_n_scored, br = s.oot_published_base_rate;
    if (ns !== null && ns !== undefined) {
      nInfo = " It was graded against " + Number(ns).toLocaleString("en-US")
        + " funds whose next-quarter returns have since been realized"
        + ((br !== null && br !== undefined)
            ? " (a " + fmtPct(br) + " base rate of lagging peers)." : ".");
    }

    // published vs the 0.50 coin-flip line
    const p1 = el("p", { class: "lead" });
    if (pub < 0.5) {
      p1.textContent = "On genuinely unseen filings the model scored " + fmtScore(pub)
        + " — below 0.50, i.e. worse than a coin flip at ranking which funds went on to "
        + "lag their peers." + nInfo;
    } else if (pub < 0.52) {
      p1.textContent = "On genuinely unseen filings the model scored " + fmtScore(pub)
        + " — essentially a coin flip (0.50) at ranking which funds went on to lag their "
        + "peers." + nInfo;
    } else {
      p1.textContent = "On genuinely unseen filings the model scored " + fmtScore(pub)
        + ", above the 0.50 coin-flip line." + nInfo;
    }
    card.appendChild(p1);

    // published vs the backtest (different model/run - branch on the sign, never assume)
    if (backtest !== null && backtest !== undefined) {
      const diff = pub - backtest;
      const p2 = el("p", { class: "lead" });
      if (diff <= -0.03) {
        p2.textContent = "That is " + fmtScore(-diff) + " below the backtest's "
          + fmtScore(backtest) + ": the backtest was optimistic about what this approach "
          + "delivers once the future is genuinely unknown.";
      } else if (diff >= 0.03) {
        p2.textContent = "That is " + fmtScore(diff) + " above the backtest's "
          + fmtScore(backtest) + " — on this slice the frozen predictions held up at least "
          + "as well as the backtest implied.";
      } else {
        p2.textContent = "That lands within " + fmtScore(Math.abs(diff)) + " of the "
          + "backtest's " + fmtScore(backtest) + " — roughly in line, on this slice, with "
          + "what the backtest implied.";
      }
      card.appendChild(p2);
    }
  }

  if (hasStrip) {
    card.appendChild(txt("h3", "Frozen model, quarter by quarter"));
    card.appendChild(renderQuarterStrip(pq));
    const below = pq.filter(q => q.auc !== null && q.auc !== undefined && q.auc < 0.5);
    const cap = el("p", { class: "oot-note" });
    if (below.length) {
      cap.appendChild(txt("span", "▼", "flag"));
      cap.appendChild(document.createTextNode(
        "Each bar is one quarter's AUC for the frozen model rolled forward; the dashed line "
        + "is 0.50 (a coin flip). " + below.length + " of " + pq.length + " quarters came in "
        + "below it (" + below.map(q => q.quarter).join(", ") + ") — quarters the model did "
        + "no better than chance. A single pooled number hides that."));
    } else {
      cap.appendChild(document.createTextNode(
        "Each bar is one quarter's AUC for the frozen model rolled forward; the dashed line "
        + "is 0.50 (a coin flip). Every quarter here cleared it, but the spread is wide — "
        + "read any single quarter with caution."));
    }
    card.appendChild(cap);
  }
  return card;
}

/* ---------- scorecard (honest about persistence < 0.5) ---------- */
function renderScorecard() {
  const s = DATA.scorecard;
  const wrap = document.createDocumentFragment();
  const oot = renderOOTPanel(s);
  if (oot) wrap.appendChild(oot);
  const auc = s.auc, lo = s.auc_ci ? s.auc_ci[0] : null, hi = s.auc_ci ? s.auc_ci[1] : null;
  const persist = s.persistence_auc;
  const reversed = (persist === null || persist === undefined) ? null : 1 - persist;
  const meanReverting = (persist !== null && persist !== undefined && persist < 0.5);
  const baseline = (persist === null || persist === undefined)
    ? null : Math.max(persist, reversed);

  const card = el("div", { class: "card" });
  card.appendChild(txt("p", "Backtest scorecard", "eyebrow"));
  card.appendChild(txt("h2", "Does the model actually have an edge?"));

  card.appendChild(el("div", { class: "tiles" }, [
    tile("Model AUC", fmtScore(auc)),
    tile("95% confidence interval",
      (lo === null || hi === null) ? "—" : fmtScore(lo) + " – " + fmtScore(hi)),
    tile(meanReverting ? "Reversed-persistence baseline" : "Persistence baseline",
      fmtScore(meanReverting ? reversed : persist)),
  ]));

  // AUC plain-English
  card.appendChild(txt("p",
    "AUC is a ranking score: given two funds, it is the probability the model ranks the "
    + "one that went on to lag its peer group above the one that did not. 0.50 is a coin "
    + "flip; 1.00 is perfect. It is measured on past data.", "lead"));

  // baseline / persistence honesty
  const bp = el("p", { class: "lead" });
  if (persist === null || persist === undefined) {
    bp.textContent = "No persistence baseline is available for this run.";
  } else if (meanReverting) {
    bp.textContent =
      "The naive “past laggards keep lagging” rule scores only "
      + fmtScore(persist) + ". Because that is below 0.50, peer-relative returns actually "
      + "mean-revert here — recent laggards tend to bounce back rather than keep "
      + "trailing. The honest bar to clear is therefore the reversed rule (bet that recent "
      + "leaders lag next), which scores " + fmtScore(reversed) + ". A model earns its keep "
      + "only by beating that " + fmtScore(baseline) + ".";
  } else {
    bp.textContent =
      "The naive “past laggards keep lagging” rule scores " + fmtScore(persist)
      + ". A model earns its keep only by beating that baseline of " + fmtScore(baseline) + ".";
  }
  card.appendChild(bp);

  // significance from p_edge_le_zero
  const p = s.p_edge_le_zero;
  const sp = el("p", { class: "lead" });
  if (p === null || p === undefined) {
    sp.textContent = "A significance estimate for the model's edge is not available for this run.";
  } else {
    sp.textContent =
      "Estimated probability that the model has no real edge over the baseline (its true "
      + "advantage is zero or negative): " + fmtPct(p) + ". Lower is stronger evidence.";
  }
  card.appendChild(sp);

  // per-quarter range
  if (Array.isArray(s.per_quarter) && s.per_quarter.length) {
    const vals = s.per_quarter.map(q => q.auc).filter(v => v !== null && v !== undefined);
    if (vals.length) {
      const mn = Math.min.apply(null, vals), mx = Math.max.apply(null, vals);
      card.appendChild(txt("p",
        "Quarter by quarter the AUC ranged from " + fmtScore(mn) + " to " + fmtScore(mx)
        + " across " + s.per_quarter.length + " quarters — a single pooled number "
        + "hides that spread.", "lead"));
    }
  }

  // flip rate
  const fr = s.mean_flip_rate;
  if (fr !== null && fr !== undefined) {
    const stability = fr < 0.15 ? "fairly stable" : (fr < 0.35 ? "moderately stable" : "unstable");
    card.appendChild(txt("p",
      "On average " + fmtPct(fr) + " of funds changed peer group from one quarter to the "
      + "next, so cluster membership is " + stability + ". Read the groups as approximate, "
      + "not fixed.", "lead"));
  }

  // what this is and isn't
  const wi = el("p", { class: "lead" });
  wi.appendChild(txt("strong", "What this number is, and isn't. "));
  wi.appendChild(document.createTextNode(
    "The AUC is a backward-looking ranking score, not a forecast of returns, a price "
    + "target, or a buy/sell signal. It was measured on history and can be wrong on any "
    + "individual fund. Treat it as one weak, past-tense input among many — never as advice."));
  card.appendChild(wi);

  wrap.appendChild(card);
  return wrap;
}

/* ---------- scatter (SVG, nearest-point hover) ---------- */
function renderScatter() {
  const W = 920, H = 540;
  const padL = 46, padR = 20, padT = 20, padB = 44;   // room for dimensionless captions
  const x0 = padL, x1 = W - padR, y0 = padT, y1 = H - padB;
  const pts = DATA.coords;
  const wrap = el("div", { class: "scatter-wrap" });
  const svg = document.createElementNS(SVGNS, "svg");
  svg.setAttribute("class", "scatter");
  svg.setAttribute("viewBox", "0 0 " + W + " " + H);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Similarity map of funds, coloured by peer group");

  if (!pts.length) {
    wrap.appendChild(svg);
    wrap.appendChild(txt("p", "No coordinates available.", "muted"));
    return wrap;
  }

  function svgEl(tag, attrs) {
    const n = document.createElementNS(SVGNS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    return n;
  }

  // recessive gridlines behind the marks (spatial reference only - the embedding is
  // unitless, so no numeric ticks: they would imply a false precision).
  const NV = 6, NH = 4;
  for (let i = 1; i < NV; i++) {
    const gx = x0 + (x1 - x0) * i / NV;
    svg.appendChild(svgEl("line", { class: "grid-line",
      x1: gx.toFixed(1), y1: y0, x2: gx.toFixed(1), y2: y1 }));
  }
  for (let i = 1; i < NH; i++) {
    const gy = y0 + (y1 - y0) * i / NH;
    svg.appendChild(svgEl("line", { class: "grid-line",
      x1: x0, y1: gy.toFixed(1), x2: x1, y2: gy.toFixed(1) }));
  }
  svg.appendChild(svgEl("rect", { class: "frame",
    x: x0, y: y0, width: (x1 - x0), height: (y1 - y0), rx: 6 }));

  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const p of pts) {
    if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
  }
  const spanX = (maxX - minX) || 1, spanY = (maxY - minY) || 1;
  const inset = 14;
  const sx = x => x0 + inset + (x - minX) / spanX * (x1 - x0 - 2 * inset);
  const sy = y => y1 - inset - (y - minY) / spanY * (y1 - y0 - 2 * inset);

  const placed = [];
  for (const p of pts) {
    const cx = sx(p.x), cy = sy(p.y);
    const dot = document.createElementNS(SVGNS, "circle");
    dot.setAttribute("cx", cx.toFixed(1));
    dot.setAttribute("cy", cy.toFixed(1));
    dot.setAttribute("r", "3.6");
    dot.setAttribute("fill", clusterColor(p.cluster));
    dot.setAttribute("stroke", "var(--surface-1)");   // surface ring for overlap legibility
    dot.setAttribute("stroke-width", "1.4");
    svg.appendChild(dot);
    placed.push({ cx, cy, p });
  }

  // dimensionless axis captions (the embedding carries no units of its own)
  const capX = svgEl("text", { class: "axis-cap", x: ((x0 + x1) / 2).toFixed(0),
    y: (H - 12), "text-anchor": "middle" });
  capX.textContent = "Similarity dimension 1 →";
  svg.appendChild(capX);
  const capY = svgEl("text", { class: "axis-cap", x: 16, y: ((y0 + y1) / 2).toFixed(0),
    "text-anchor": "middle", transform: "rotate(-90 16 " + ((y0 + y1) / 2).toFixed(0) + ")" });
  capY.textContent = "Similarity dimension 2 →";
  svg.appendChild(capY);
  wrap.appendChild(svg);

  const tip = el("div", { class: "tooltip" });
  wrap.appendChild(tip);

  function onMove(evt) {
    const rect = svg.getBoundingClientRect();
    const scaleX = W / rect.width, scaleY = H / rect.height;
    const mx = (evt.clientX - rect.left) * scaleX;
    const my = (evt.clientY - rect.top) * scaleY;
    let best = null, bestD = Infinity;
    for (const q of placed) {
      const d = (q.cx - mx) * (q.cx - mx) + (q.cy - my) * (q.cy - my);
      if (d < bestD) { bestD = d; best = q; }
    }
    if (!best || bestD > 900) { tip.style.opacity = 0; return; }  // ~30px in svg units
    const info = SID_INFO[best.p.sid] || { name: best.p.sid, ticker: "" };
    tip.textContent = "";
    tip.appendChild(txt("div", info.name || best.p.sid, "t-val"));
    const sub = (info.ticker ? info.ticker + " · " : "")
      + (CLUSTER_TITLE[best.p.cluster] || ("Cluster " + best.p.cluster));
    tip.appendChild(txt("div", sub, "t-sub"));
    const left = best.cx / scaleX + 12, top = best.cy / scaleY + 12;
    tip.style.left = Math.min(left, rect.width - 180) + "px";
    tip.style.top = top + "px";
    tip.style.opacity = 1;
  }
  svg.addEventListener("pointermove", onMove);
  svg.addEventListener("pointerleave", () => { tip.style.opacity = 0; });
  return wrap;
}

/* ---------- CLUSTER INDEX (also the colour legend) ---------- */
function renderClusterIndex() {
  const frag = document.createDocumentFragment();
  frag.appendChild(txt("h1", "Peer groups"));
  frag.appendChild(txt("p",
    "One row per peer group; the swatch is the colour used on the map. Click any column "
    + "header to sort, or a group name to open it.", "lead"));

  const table = el("table", { class: "sortable" });
  const thead = el("thead");
  const htr = el("tr");
  const cols = [
    ["Group", false], ["Dominant category", false], ["Share", true], ["Funds", true],
    ["Avg Sharpe", true], ["Avg volatility", true], ["Avg max drawdown", true],
    ["Median net assets", true],
  ];
  for (const [label, num] of cols) { const th = el("th", num ? { class: "num" } : null); th.textContent = label; htr.appendChild(th); }
  thead.appendChild(htr);
  table.appendChild(thead);

  const tbody = el("tbody");
  for (const c of DATA.clusters) {
    const tr = el("tr");
    // Group (swatch + link)
    const tdName = el("td");
    const sw = el("span", { class: "swatch" });
    sw.style.background = clusterColor(c.cluster_id);
    const a = el("a", { class: "rowlink", href: "#cluster-" + c.cluster_id });
    a.textContent = c.short_title;
    tdName.appendChild(sw); tdName.appendChild(a);
    tdName.dataset.sort = String(c.short_title).toLowerCase();
    tr.appendChild(tdName);

    tr.appendChild(textCell(c.dominant_category));
    tr.appendChild(numCell(c.dominant_share, fmtPct));
    tr.appendChild(numCell(c.member_count, v => v.toLocaleString("en-US")));
    tr.appendChild(numCell(c.avg_sharpe, fmtSharpe));
    tr.appendChild(numCell(c.avg_volatility, fmtPct));
    tr.appendChild(numCell(c.avg_max_drawdown, fmtPct));
    tr.appendChild(numCell(c.median_net_assets, fmtMoney));
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  makeSortable(table);
  frag.appendChild(el("div", { class: "tbl-scroll" }, [table]));
  return frag;
}

/* ---------- ONE CLUSTER SECTION ---------- */
function renderClusterSection(c) {
  const frag = document.createDocumentFragment();

  const head = el("div");
  const h1 = el("h1");
  const sw = el("span", { class: "swatch" });
  sw.style.background = clusterColor(c.cluster_id);
  sw.style.width = "1rem"; sw.style.height = "1rem";
  h1.appendChild(sw);
  h1.appendChild(document.createTextNode(c.short_title));
  head.appendChild(h1);
  head.appendChild(txt("p",
    "Peer group " + c.cluster_id + " · " + c.member_count + " funds · dominant category "
    + c.dominant_category + " (" + fmtPct(c.dominant_share) + " of the group)", "identity-meta"));
  frag.appendChild(head);

  frag.appendChild(el("div", { class: "tiles" }, [
    tile("Avg Sharpe", fmtSharpe(c.avg_sharpe)),
    tile("Avg volatility", fmtPct(c.avg_volatility)),
    tile("Avg max drawdown", fmtPct(c.avg_max_drawdown)),
    tile("Median net assets", fmtMoney(c.median_net_assets)),
  ]));

  // top holdings bars
  frag.appendChild(txt("h3", "Largest shared holdings"));
  if (c.top_holdings && c.top_holdings.length) {
    const maxW = c.top_holdings.reduce((m, h) => Math.max(m, h.weight), 0) || 1;
    const bars = el("div", { class: "bars" });
    for (const h of c.top_holdings) {
      const track = el("div", { class: "bar-track" });
      const fill = el("div", { class: "bar-fill" });
      fill.style.width = (h.weight / maxW * 100).toFixed(1) + "%";
      track.appendChild(fill);
      bars.appendChild(el("div", { class: "bar-row" }, [
        txt("span", h.issuer, "issuer"), track, txt("span", fmtPct(h.weight), "wt"),
      ]));
    }
    frag.appendChild(bars);
  } else {
    frag.appendChild(txt("p", "No holdings breakdown available for this group.", "muted"));
  }

  // narrative
  frag.appendChild(txt("h3", "Summary"));
  const nar = el("p", { class: "narrative" });
  if (c.narrative && c.narrative.trim()) nar.textContent = c.narrative;
  else nar.appendChild(el("em", { text: "narrative not generated" }));
  frag.appendChild(nar);

  // members
  frag.appendChild(txt("h3", "Funds in this group"));
  frag.appendChild(memberTable(c.members, true));
  return frag;
}

/* ---------- ALLOCATION ---------- */
function renderAllocation() {
  const frag = document.createDocumentFragment();
  frag.appendChild(txt("h1", "Target-date & allocation funds"));
  frag.appendChild(txt("p",
    "These multi-asset funds are grouped by vintage. They are shown descriptively — the "
    + "peer-group model does not score them.", "lead"));
  if (!DATA.allocation.length) {
    frag.appendChild(txt("p", "No allocation funds in this universe.", "muted"));
    return frag;
  }
  for (const v of DATA.allocation) {
    frag.appendChild(txt("h3", v.vintage));
    frag.appendChild(memberTable(v.members, false));
  }
  return frag;
}

/* ---------- search ---------- */
const searchEl = document.getElementById("search");
const resultsEl = document.getElementById("results");

function runSearch() {
  const q = searchEl.value.trim().toLowerCase();
  resultsEl.textContent = "";
  if (!q) { resultsEl.hidden = true; return; }
  const hits = [];
  for (const c of DATA.clusters) {
    for (const m of c.members) {
      const name = (m.name || "").toLowerCase(), tick = (m.ticker || "").toLowerCase();
      if (name.includes(q) || tick.includes(q)) hits.push({ c, m });
      if (hits.length >= 20) break;
    }
    if (hits.length >= 20) break;
  }
  if (!hits.length) {
    resultsEl.appendChild(txt("div", "No funds match “" + searchEl.value + "”.", "r-empty"));
    resultsEl.hidden = false;
    return;
  }
  for (const h of hits) {
    const btn = el("button", { type: "button" });
    btn.appendChild(txt("div", h.m.name || h.m.sid));
    btn.appendChild(txt("div",
      (h.m.ticker ? h.m.ticker + " · " : "") + h.c.short_title, "r-sub"));
    btn.addEventListener("click", () => {
      resultsEl.hidden = true;
      searchEl.value = "";
      goToMember(h.c.cluster_id, h.m.sid);
    });
    resultsEl.appendChild(btn);
  }
  resultsEl.hidden = false;
}
searchEl.addEventListener("input", runSearch);
document.addEventListener("click", (e) => {
  if (!e.target.closest(".search-wrap")) resultsEl.hidden = true;
});

let pendingHighlight = null;                              // sid to flag after the next render
function goToMember(cid, sid) {
  pendingHighlight = sid;
  const target = "#cluster-" + cid;
  if (location.hash === target) route();                  // hash unchanged -> render + highlight now
  else location.hash = target;                            // else hashchange -> route() does both
}
function applyHighlight(sid) {
  const row = document.querySelector('#app tr[data-sid="' + cssEscape(sid) + '"]');
  if (row) {
    row.scrollIntoView({ block: "center" });
    row.classList.add("hit");
    setTimeout(() => row.classList.remove("hit"), 2600);
  }
}
function cssEscape(s) {
  return (window.CSS && CSS.escape) ? CSS.escape(s) : String(s).replace(/["\\]/g, "\\$&");
}

/* ---------- router ---------- */
const VIEWS = { overview: renderOverview, clusters: renderClusterIndex, allocation: renderAllocation };
function setActiveNav(hash) {
  const key = hash.indexOf("cluster-") === 0 ? "clusters" : hash;
  for (const a of document.querySelectorAll("nav a.navlink")) {
    a.classList.toggle("active", a.getAttribute("href") === "#" + key);
  }
}
function route() {
  const hash = (location.hash || "#overview").slice(1);
  const app = document.getElementById("app");
  app.textContent = "";
  setActiveNav(hash);
  window.scrollTo(0, 0);
  let rendered = false;
  if (hash.indexOf("cluster-") === 0) {
    const id = parseInt(hash.slice("cluster-".length), 10);
    const c = DATA.clusters.find(x => x.cluster_id === id);
    if (c) { app.appendChild(renderClusterSection(c)); rendered = true; }
  }
  if (!rendered) app.appendChild((VIEWS[hash] || renderOverview)());
  if (pendingHighlight) { const s = pendingHighlight; pendingHighlight = null; applyHighlight(s); }
}
window.addEventListener("hashchange", route);

/* ---------- boot ---------- */
document.getElementById("disclaimer").textContent = DATA.disclaimer;
route();   // cold load - reads location.hash, defaults to #overview
</script>
</body>
</html>"""
