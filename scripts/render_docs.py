"""Render project docs to self-contained, offline HTML.

Reads three small data files and writes two double-clickable HTML pages with
NO external dependencies (inline CSS, inline SVG — no CDN, no network):

  techniques.json +
  decisions.json  ->  reflection.html   (why-these-techniques section, then the
                                         decision log, one card per entry)
  workflow.json   ->  workflow.html     (vertical step-flow diagram)

The agent edits the JSON; this script owns the markup. Same input -> same
output, every run.

Usage:
  python render_docs.py [--decisions decisions.json] [--workflow workflow.json]
                        [--techniques techniques.json] [--out-dir .]

decisions.json:  [{"date": "2026-07-07", "decision": "...", "rationale": "..."}]
techniques.json: [{"technique": "...", "why": "..."}]  (optional file; section
                 is omitted from reflection.html when absent)
workflow.json :  {"steps": [{"name": "step0_setup", "status": "done"}, ...]}
                 status is one of: done | in_progress | pending
"""
import argparse
import html
import json
from pathlib import Path

# Status -> (fill, border) colours. Chosen to read in both light and dark.
STATUS_COLORS = {
    "done": ("#1a7f37", "#0f5323"),
    "in_progress": ("#9a6700", "#7a5200"),
    "pending": ("#57606a", "#424a53"),
}

_PAGE_CSS = """
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 2rem 1rem; font: 16px/1.5 system-ui, sans-serif;
         background: #f6f8fa; color: #1f2328; }
  main { max-width: 820px; margin: 0 auto; }
  h1 { font-size: 1.5rem; margin: 0 0 1.5rem; }
  @media (prefers-color-scheme: dark) {
    body { background: #0d1117; color: #e6edf3; }
    .card { background: #161b22 !important; border-color: #30363d !important; }
  }
"""

_CARD_CSS = """
  .card { background: #fff; border: 1px solid #d0d7de; border-radius: 10px;
          padding: 1rem 1.25rem; margin-bottom: 1rem; }
  .decision { font-weight: 600; margin: 0 0 .35rem; }
  .rationale { margin: 0; }
  .rationale::before { content: "Why: "; font-weight: 600; opacity: .8; }
  .date { float: right; font-size: .8rem; opacity: .6; }
  h2 { font-size: 1.15rem; margin: 2rem 0 1rem; }
  h2:first-of-type { margin-top: 0; }
"""


def _page(title: str, body: str, extra_css: str = "") -> str:
    """Wrap body in a minimal, self-contained HTML document."""
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{html.escape(title)}</title>\n<style>{_PAGE_CSS}{extra_css}</style>\n"
        f"</head>\n<body>\n<main>\n<h1>{html.escape(title)}</h1>\n{body}\n"
        "</main>\n</body>\n</html>\n"
    )


def render_reflection(decisions: list, techniques: list = None) -> str:
    """Optional why-these-techniques primer, then one card per decision."""
    sections = []
    if techniques:
        cards = []
        for t in techniques:
            technique = html.escape(str(t.get("technique", "")))
            why = html.escape(str(t.get("why", "")))
            cards.append(
                f'<div class="card"><p class="decision">{technique}</p>'
                f'<p class="rationale">{why}</p></div>'
            )
        sections.append("<h2>Why these techniques</h2>\n" + "\n".join(cards))

    cards = []
    for d in decisions:
        date = html.escape(str(d.get("date", "")))
        decision = html.escape(str(d.get("decision", "")))
        rationale = html.escape(str(d.get("rationale", "")))
        date_html = f'<span class="date">{date}</span>' if date else ""
        cards.append(
            f'<div class="card">{date_html}'
            f'<p class="decision">{decision}</p>'
            f'<p class="rationale">{rationale}</p></div>'
        )
    log = "\n".join(cards) if cards else "<p>No decisions logged yet.</p>"
    sections.append(("<h2>Decision log</h2>\n" if techniques else "") + log)
    return _page("Reflection — decision log", "\n".join(sections), _CARD_CSS)


def render_workflow(steps: list) -> str:
    """Vertical flow of step boxes joined by arrows, as inline SVG."""
    box_w, box_h, gap = 300, 54, 34
    width = box_w + 40
    height = len(steps) * (box_h + gap) + gap if steps else 120
    parts = [
        f'<svg width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        'style="max-width:100%;height:auto">',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" '
        'refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#8b949e"/>'
        "</marker></defs>",
    ]
    x = 20
    for i, step in enumerate(steps):
        name = html.escape(str(step.get("name", "")))
        status = step.get("status", "pending")
        fill, border = STATUS_COLORS.get(status, STATUS_COLORS["pending"])
        y = gap + i * (box_h + gap)
        parts.append(
            f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="10" '
            f'fill="{fill}" stroke="{border}" stroke-width="1.5"/>'
        )
        parts.append(
            f'<text x="{x + box_w / 2}" y="{y + box_h / 2 + 5}" fill="#fff" '
            f'font-family="system-ui,sans-serif" font-size="16" font-weight="600" '
            f'text-anchor="middle">{name}</text>'
        )
        if i < len(steps) - 1:  # arrow down to the next box
            cx = x + box_w / 2
            parts.append(
                f'<line x1="{cx}" y1="{y + box_h}" x2="{cx}" y2="{y + box_h + gap}" '
                'stroke="#8b949e" stroke-width="2" marker-end="url(#arrow)"/>'
            )
    parts.append("</svg>")
    return _page("Workflow — step flow", "".join(parts))


def _load(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decisions", default="decisions.json")
    ap.add_argument("--workflow", default="workflow.json")
    ap.add_argument("--techniques", default="techniques.json")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    decisions = _load(Path(args.decisions), [])
    workflow = _load(Path(args.workflow), {"steps": []})
    techniques = _load(Path(args.techniques), [])

    (out / "reflection.html").write_text(
        render_reflection(decisions, techniques), encoding="utf-8"
    )
    (out / "workflow.html").write_text(
        render_workflow(workflow.get("steps", [])), encoding="utf-8"
    )
    print(f"wrote {out/'reflection.html'} and {out/'workflow.html'}")


if __name__ == "__main__":
    main()
