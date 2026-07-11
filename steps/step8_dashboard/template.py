"""step8_dashboard/template.py - the complete, self-contained dashboard document.

`TEMPLATE` is one HTML string with the literal token ``__PAYLOAD_JSON__`` where the
minified payload JSON is spliced in by ``render.render_dashboard``. Every section is
built client-side by the embedded vanilla JS from that single JSON blob - the Python
side builds nothing per-cluster, which keeps the output byte-for-byte deterministic
and the file free of any network dependency.

Design notes (dataviz skill):
- Colour roles are CSS custom properties in :root, re-stated for prefers-color-scheme
  dark; the chart body is written against roles, never raw hex (palette.md).
- Categorical cluster colour: the reference palette tops out at 8 hues and forbids
  cycling, but this universe has ~30 peer groups. Per the task's explicit mandate we
  generate a fixed HSL wheel keyed on cluster_id (golden-angle spacing, lightness/
  chroma inside the skill's per-mode bands). Colour is never the sole identity channel:
  the scatter has a name/ticker hover tooltip and the Clusters index table is the
  legend (a swatch per row beside the short title). See the report for this deviation.
- Marks: 2px surface ring on scatter dots; top-holdings bars are one-hue (slot-1 blue)
  with 4px rounded data-ends; recessive hairline chrome; text wears ink tokens only.
- Interaction: dense scatter uses a single nearest-point hover layer (not 2,086 hit
  targets); tables are generic click-to-sort, numeric-aware, nulls last both ways.
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
    --surface-1: #fcfcfb;
    --page: #f9f9f7;
    --card: #ffffff;
    --ink-1: #0b0b0b;
    --ink-2: #52514e;
    --ink-muted: #898781;
    --grid: #e1e0d9;
    --axis: #c3c2b7;
    --border: rgba(11,11,11,0.10);
    --series-1: #2a78d6;
    --series-1-wash: #cde2fb;
    --good: #006300;
    --nav-h: 56px;
    --foot-h: 68px;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --surface-1: #1a1a19;
      --page: #0d0d0d;
      --card: #1a1a19;
      --ink-1: #ffffff;
      --ink-2: #c3c2b7;
      --ink-muted: #898781;
      --grid: #2c2c2a;
      --axis: #383835;
      --border: rgba(255,255,255,0.10);
      --series-1: #3987e5;
      --series-1-wash: #184f95;
      --good: #0ca30c;
    }
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body {
    margin: 0;
    font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
    background: var(--page);
    color: var(--ink-1);
    padding-bottom: calc(var(--foot-h) + 1rem);
  }
  a { color: var(--series-1); }

  /* ---- top nav + search ---- */
  nav {
    position: sticky; top: 0; z-index: 20;
    display: flex; align-items: center; gap: .35rem; flex-wrap: wrap;
    height: var(--nav-h); padding: 0 1rem;
    background: var(--surface-1);
    border-bottom: 1px solid var(--border);
  }
  nav .brand { font-weight: 600; margin-right: .75rem; white-space: nowrap; }
  nav a.navlink {
    text-decoration: none; color: var(--ink-2);
    padding: .3rem .6rem; border-radius: 7px; white-space: nowrap;
  }
  nav a.navlink:hover { background: var(--page); }
  nav a.navlink.active { color: var(--ink-1); background: var(--page); font-weight: 600; }
  .search-wrap { position: relative; margin-left: auto; }
  #search {
    font: inherit; padding: .35rem .6rem; width: 15rem; max-width: 45vw;
    border: 1px solid var(--border); border-radius: 7px;
    background: var(--card); color: var(--ink-1);
  }
  #results {
    position: absolute; right: 0; top: 2.4rem; z-index: 30;
    width: 22rem; max-width: 80vw; max-height: 60vh; overflow-y: auto;
    background: var(--card); border: 1px solid var(--border); border-radius: 9px;
    box-shadow: 0 8px 28px rgba(0,0,0,.18); padding: .3rem;
  }
  #results button {
    display: block; width: 100%; text-align: left; font: inherit; cursor: pointer;
    background: none; border: 0; color: var(--ink-1);
    padding: .4rem .5rem; border-radius: 6px;
  }
  #results button:hover, #results button:focus { background: var(--page); }
  #results .r-sub { color: var(--ink-2); font-size: .82rem; }
  #results .r-empty { padding: .5rem; color: var(--ink-2); }

  /* ---- layout ---- */
  main { max-width: 1000px; margin: 0 auto; padding: 1.5rem 1rem 2rem; }
  h1 { font-size: 1.5rem; margin: 0 0 .4rem; }
  h2 { font-size: 1.2rem; margin: 2rem 0 .8rem; }
  h3 { font-size: 1.02rem; margin: 1.4rem 0 .5rem; }
  p.lead { color: var(--ink-2); max-width: 62ch; }
  .muted { color: var(--ink-2); }

  .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.1rem 1.25rem; margin: 1rem 0;
  }

  /* ---- stat tiles ---- */
  .tiles { display: flex; flex-wrap: wrap; gap: .75rem; margin: 1rem 0; }
  .tile {
    flex: 1 1 8rem; min-width: 8rem;
    background: var(--card); border: 1px solid var(--border);
    border-radius: 11px; padding: .8rem .9rem;
  }
  .tile .label { color: var(--ink-2); font-size: .8rem; }
  .tile .value { font-size: 1.5rem; font-weight: 600; margin-top: .15rem; }
  .hero { font-size: 2.6rem; font-weight: 600; line-height: 1.1; }

  /* ---- tables ---- */
  .tbl-scroll { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; font-size: .9rem; }
  th, td { padding: .45rem .6rem; text-align: left; border-bottom: 1px solid var(--border); }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  thead th { color: var(--ink-2); font-weight: 600; white-space: nowrap; }
  table.sortable thead th.sortable-th { cursor: pointer; user-select: none; }
  table.sortable thead th.sortable-th:hover { color: var(--ink-1); }
  th[aria-sort="ascending"]::after { content: " \2191"; color: var(--ink-muted); }
  th[aria-sort="descending"]::after { content: " \2193"; color: var(--ink-muted); }
  tbody tr:hover { background: var(--page); }
  tbody tr.hit { background: var(--series-1-wash); }
  .swatch {
    display: inline-block; width: .8rem; height: .8rem; border-radius: 3px;
    margin-right: .45rem; vertical-align: -1px;
    box-shadow: 0 0 0 2px var(--surface-1);
  }
  a.rowlink { text-decoration: none; color: inherit; }
  a.rowlink:hover { text-decoration: underline; }
  sup.foot a { text-decoration: none; color: var(--series-1); font-weight: 600; }

  /* ---- scatter ---- */
  .scatter-wrap { position: relative; }
  svg.scatter { width: 100%; height: auto; display: block;
    background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; }
  .tooltip {
    position: absolute; pointer-events: none; z-index: 5;
    background: var(--card); color: var(--ink-1);
    border: 1px solid var(--border); border-radius: 8px;
    padding: .4rem .55rem; font-size: .82rem; max-width: 16rem;
    box-shadow: 0 6px 20px rgba(0,0,0,.2); opacity: 0; transition: opacity .08s;
  }
  .tooltip .t-val { font-weight: 600; }
  .tooltip .t-sub { color: var(--ink-2); }

  /* ---- top-holdings bars ---- */
  .bars { margin: .4rem 0; }
  .bar-row { display: grid; grid-template-columns: 12rem 1fr auto; gap: .6rem;
    align-items: center; margin: .3rem 0; }
  .bar-row .issuer { color: var(--ink-1); overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap; }
  .bar-track { background: var(--page); border-radius: 5px; height: 14px; overflow: hidden; }
  .bar-fill { height: 100%; background: var(--series-1);
    border-radius: 0 4px 4px 0; }
  .bar-row .wt { color: var(--ink-2); font-variant-numeric: tabular-nums; font-size: .85rem; }
  @media (max-width: 620px) { .bar-row { grid-template-columns: 8rem 1fr auto; } }

  .narrative { max-width: 68ch; }
  .identity-meta { color: var(--ink-2); margin: .2rem 0 0; }

  /* ---- footer ---- */
  footer#disclaimer {
    position: fixed; left: 0; right: 0; bottom: 0; z-index: 15;
    background: var(--surface-1); border-top: 1px solid var(--border);
    color: var(--ink-2); font-size: .78rem; line-height: 1.4;
    padding: .55rem 1rem; text-align: center;
    max-height: var(--foot-h); overflow-y: auto;
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
  const s = DARK ? 62 : 58;                          // chroma inside the skill's band
  const l = DARK ? 60 : 46;                          // lightness inside the per-mode band
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

/* ---------- scorecard (honest about persistence < 0.5) ---------- */
function renderScorecard() {
  const s = DATA.scorecard;
  const auc = s.auc, lo = s.auc_ci ? s.auc_ci[0] : null, hi = s.auc_ci ? s.auc_ci[1] : null;
  const persist = s.persistence_auc;
  const reversed = (persist === null || persist === undefined) ? null : 1 - persist;
  const meanReverting = (persist !== null && persist !== undefined && persist < 0.5);
  const baseline = (persist === null || persist === undefined)
    ? null : Math.max(persist, reversed);

  const card = el("div", { class: "card" });
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

  return card;
}

/* ---------- scatter (SVG, nearest-point hover) ---------- */
function renderScatter() {
  const W = 920, H = 520, PAD = 18;
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

  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const p of pts) {
    if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
  }
  const spanX = (maxX - minX) || 1, spanY = (maxY - minY) || 1;
  const sx = x => PAD + (x - minX) / spanX * (W - 2 * PAD);
  const sy = y => H - PAD - (y - minY) / spanY * (H - 2 * PAD);

  const placed = [];
  for (const p of pts) {
    const cx = sx(p.x), cy = sy(p.y);
    const dot = document.createElementNS(SVGNS, "circle");
    dot.setAttribute("cx", cx.toFixed(1));
    dot.setAttribute("cy", cy.toFixed(1));
    dot.setAttribute("r", "3.4");
    dot.setAttribute("fill", clusterColor(p.cluster));
    dot.setAttribute("stroke", "var(--surface-1)");   // 2px surface ring for overlap
    dot.setAttribute("stroke-width", "1");
    svg.appendChild(dot);
    placed.push({ cx, cy, p });
  }
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
