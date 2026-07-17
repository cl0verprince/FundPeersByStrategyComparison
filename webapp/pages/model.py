"""Model health - the page the product's credibility rests on. Verdict first, then evidence.

Deviation from the plan (documented, minimal, proven by a failing test): the plan's verdict
card rendered ``label.upper()`` (e.g. "SIGNAL DEGRADED"). ``STATUS`` labels are mixed-case
("Signal degraded") and the honesty-pattern chip (`honesty.status_chip`) never uppercases
them either, so upper-casing here silently broke exact-text matching against that shared
vocabulary (the test `test_model_page_leads_with_verdict_and_shows_coinflip` asserts the
mixed-case string "Signal degraded" is visible). Fixed by dropping `.upper()` so the verdict
card echoes the same casing as everywhere else in the app.
"""
from nicegui import ui

from webapp.components import charts, honesty
from webapp.theme import STATUS, TOKENS


def render_model(store) -> None:
    health = store.model_health()
    prov = store.provenance()
    t = TOKENS["light"]

    ui.label("Can you trust the lag-probability signal right now?").classes(
        "text-2xl font-semibold")
    honesty.freshness_stamp(prov)

    retired = health["health_state"] == "retired"
    icon, label, token = STATUS[health["health_state"]]
    with ui.card().classes("w-full p-4 border-2").style(f"border-color:{t[token]}"):
        if retired:
            # Muted/ink treatment (STATUS["retired"] token is "muted", not "critical") -
            # a fact, not an alarm; never the red border-2 the degraded/weak states get.
            ui.label(f"{icon} RETIRED as of {health.get('retired_as_of', '')}").classes(
                "text-xl font-semibold").style(f"color:{t[token]}")
            ui.label(str(health.get("retirement_statement") or health["rule_text"])
                     ).classes("text-sm")
        else:
            ui.label(f"{icon} {label}").classes("text-xl font-semibold").style(
                f"color:{t[token]}")
            ui.label(health["rule_text"]).classes("text-sm")
            ui.label(f"The rule: {health['rule_text']}").classes("text-xs text-gray-500")

    if retired:
        ui.label("Since retirement").classes("text-lg font-semibold mt-4")
        record = store.retirement_record()
        if len(record):
            ui.table(rows=record.round(3).to_dict("records"))
        else:
            ui.label(
                "First post-retirement score expected ~Nov 2026 (when the next N-PORT "
                "data set publishes). The frozen model's predictions are scored against "
                "reality every quarter; this record is what would justify — or refute — "
                "this retirement."
            ).classes("text-sm text-gray-700 max-w-2xl")
        ui.label(
            "Two scorers, disclosed: the retirement statement's trigger numbers (0.457, "
            "0.427) are the deployed retrained model's per-quarter scores; this standing "
            "record grades the frozen step7 model, whose same-quarter scores were 0.428 "
            "and 0.418. Both scorers were below the 0.5 coin-flip in both quarters."
        ).classes("text-xs max-w-2xl").style(f"color:{t['ink2']}")

    ui.label("Realized AUC by quarter vs baselines").classes("text-lg font-semibold mt-4")
    dfq = store.model_quarters()
    ui.echart(charts.auc_by_quarter_option("light", dfq)).classes("w-full h-72")
    with ui.expansion("table view"):
        ui.table(rows=dfq.round(3).to_dict("records"))

    # Plain-English AUC explainer, inline
    live = health.get("pooled_live_auc")
    if live:
        n = round(live * 100)
        ui.label(
            f"What AUC means: pick one fund that went on to lag its peers and one that "
            f"didn't. AUC {live:.3f} means the model gave the lagger the higher warning "
            f"about {n} times out of 100. A coin flip gets 50; recently this model has "
            f"been below 50.").classes("text-sm text-gray-700 max-w-2xl")

    ui.label("Backtest vs reality").classes("text-lg font-semibold mt-4")
    rows = [("backtest (hindsight)", health["backtest_auc"], health["backtest_auc"]),
            ("committed forward → reality", health["backtest_auc"],
             health["pooled_live_auc"]),
            ("last realized quarter", health["pooled_live_auc"], health["auc_last"])]
    rows = [(a, float(b), float(c)) for a, b, c in rows if b is not None and c is not None]
    ui.echart(charts.dumbbell_option("light", rows)).classes("w-full h-48")
    ui.label("The backtest is measured on the past with hindsight of the modeling "
             "choices — shown for contrast, never as the headline."
             ).classes("text-xs text-gray-500")

    ui.label("Calibration (held-out test quarters)").classes("text-lg font-semibold mt-4")
    calib = store.calibration()
    if len(calib):
        ui.echart(charts.calibration_option(
            "light", calib, float(health["base_rate"] or 0.5))).classes("w-full h-64")
        ui.label('Reading: when the model said "70%", how often did funds actually lag?'
                 ).classes("text-xs text-gray-500")

    if health.get("label_noise_floor") is not None:
        ui.label(f"Label noise floor: ~{float(health['label_noise_floor']):.0%} of "
                 "lag/lead labels flip under peer-set perturbation — a ceiling on any "
                 "model of this target.").classes("text-sm text-gray-700")

    # Forward book: what the model is saying right now, with nothing to compare it to yet.
    ui.label("Forward book").classes("text-lg font-semibold mt-4")
    if retired:
        ui.label("No live forward book — the model is retired; no new predictions are "
                 "generated.").classes("text-sm text-gray-700")
    else:
        probs = store.forward_probabilities()
        n = len(probs)
        ui.echart(charts.histogram_option("light", list(probs))).classes("w-full h-56")
        ui.label(
            f"Distribution of {n} live, uncommitted-outcome "
            f"probabilit{'y' if n == 1 else 'ies'} in the current forward book. None of "
            "these outcomes are known yet: they will be scored against actual returns "
            "once next quarter's N-PORT filings post, and folded into the calibration "
            "and prediction-history charts above."
        ).classes("text-xs text-gray-500")

    if retired:
        ui.label("Resolved 2026-07-17: retired.").classes("text-sm font-semibold mt-2")
    else:
        ui.label("Open question, published: two consecutive below-chance quarters is under "
                 "investigation; the signal may be retired or retrained."
                 ).classes("text-sm font-semibold mt-2")
    honesty.disclaimer_line()
