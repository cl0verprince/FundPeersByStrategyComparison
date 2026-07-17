"""The honesty pattern components. Placement rule enforced by construction:
probability_card is the ONLY probability renderer and it embeds status_chip.
All colors come from webapp.theme.TOKENS; mode defaults to "light" until the
live mode is wired in (Task 7)."""
from nicegui import ui

from webapp.theme import DISCLAIMER, PROBABILITY_SENTENCE, STATUS, TOKENS


def status_chip(health: dict, mode: str = "light") -> None:
    icon, label, token = STATUS[health["health_state"]]
    color = TOKENS[mode][token]
    with ui.link(target="/model").classes("no-underline"):
        ui.label(f"{icon} {label}").classes(
            "px-2 py-0.5 rounded text-xs font-semibold border"
        ).style(f"color:{color}; border-color:{color}")


def freshness_stamp(provenance: dict, fund_last_quarter: str = None,
                    mode: str = "light") -> None:
    t = TOKENS[mode]
    parts = [f"Data as of {provenance['last_quarter']}",
             "SEC filings publish with a ~60-day lag",
             f"extract built {str(provenance.get('refreshed_at', ''))[:10]}"]
    if fund_last_quarter and fund_last_quarter != provenance["last_quarter"]:
        parts.append(f"this fund last filed {fund_last_quarter}")
    ui.label(" · ".join(parts)).classes("text-xs").style(f"color:{t['muted']}")


def disclaimer_line(mode: str = "light") -> None:
    t = TOKENS[mode]
    ui.label(DISCLAIMER).classes("text-xs").style(f"color:{t['muted']}")


def probability_card(pred: dict, health: dict, reason: str = None,
                     mode: str = "light") -> None:
    """The only legal rendering of a lag probability (design.md 'uncertain_probability')."""
    t = TOKENS[mode]
    with ui.card().classes("w-full max-w-sm p-4"):
        ui.label("Next-quarter outlook").classes("text-sm font-semibold")
        if pred is None:
            ui.label("No prediction this quarter").classes("text-lg")
            ui.label(reason or "This fund is not in the current forward book."
                     ).classes("text-sm").style(f"color:{t['muted']}")
            status_chip(health, mode)
            return
        p = float(pred["predicted_probability"])
        pct = round(p * 100)
        interval = ""
        if pred.get("ci_low") is not None and not _isna(pred.get("ci_low")):
            half = (float(pred["ci_high"]) - float(pred["ci_low"])) / 2 * 100
            interval = f" ±{round(half)} pts"
        ui.label(f"{pct}%{interval}").classes("text-4xl font-semibold")
        # the meter: seq-ramp fill keyed to the value, hairline tick at 50%
        fill = t["seq"][min(6, 2 + int(p * 5))]
        ui.html(
            f'<div style="position:relative;height:10px;border-radius:5px;'
            f'background:{t["seq"][0]}">'
            f'<div style="width:{pct}%;height:10px;border-radius:5px;background:{fill}"></div>'
            f'<div style="position:absolute;left:50%;top:-2px;width:1px;height:14px;'
            f'background:{t["muted"]}"></div></div>')
        ui.label(f"{PROBABILITY_SENTENCE} ({pred['target_quarter']})"
                 ).classes("text-sm").style(f"color:{t['ink2']}")
        if pred.get("flip_rate") is not None and not _isna(pred.get("flip_rate")):
            ui.label(f"The outcome itself is noisy: this fund's lag/lead label flips in "
                     f"{float(pred['flip_rate']):.0%} of peer-set draws."
                     ).classes("text-xs").style(f"color:{t['muted']}")
        status_chip(health, mode)
        ui.label("A probability from a model whose live scorecard is public."
                 ).classes("text-xs").style(f"color:{t['muted']}")


def retirement_card(health: dict, mode: str = "light") -> None:
    """The fund-page Zone A tile when the model is retired: a fact, not an alarm.
    Never renders a probability or meter (design: no live probabilities anywhere)."""
    t = TOKENS[mode]
    with ui.card().classes("w-full max-w-sm p-4"):
        ui.label(f"Model retired ({health.get('retired_as_of', '')})").classes(
            "text-sm font-semibold")
        ui.label(str(health.get("retirement_statement") or "")).classes(
            "text-sm").style(f"color:{t['ink2']}")
        ui.link("The full record →", "/model").classes("text-sm")
        status_chip(health, mode)


def _isna(v) -> bool:
    return v != v  # NaN check without importing pandas here
