"""step5_narrate — retrieve everything computed about a fund and have phi-4 (via LM Studio)
explain it in plain English. RAG-style: the LLM narrates retrieved facts, it never invents
numbers or makes predictions itself.

See steps/step5_narrate/design.md for the retrieval contract and why this is the one
deliberately non-deterministic step in the pipeline.
"""
import logging

import pandas as pd
from openai import OpenAI

from fundspeers.config import load_env
from fundspeers.io import load_table

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a financial-data narrator. You are given retrieved facts about one mutual fund "
    "for one quarter, already computed by a data pipeline. Explain those facts in plain, "
    "accessible English for someone reviewing the fund.\n\n"
    "Rules:\n"
    "- Use ONLY the numbers and facts given to you below. Never invent a number, a peer fund, "
    "or a prediction.\n"
    "- If a piece of data is marked as unavailable, say so plainly instead of guessing.\n"
    "- This is a descriptive summary, not investment advice - do not recommend buying, "
    "selling, or holding.\n"
    "- Keep it to a short paragraph or two."
)


def _get_client(cfg: dict) -> OpenAI:
    env = load_env()
    return OpenAI(base_url=env["LM_STUDIO_BASE_URL"], api_key=env["LM_STUDIO_API_KEY"])


def _first_or_none(df: pd.DataFrame, col: str):
    return df[col].iloc[0] if not df.empty else None


def build_context(series_id: str, quarter: str, cfg: dict) -> dict:
    """Retrieve every fact this step is allowed to narrate about one (series_id, quarter)."""
    funds = load_table("funds", cfg)
    fund_row = funds[(funds["series_id"] == series_id) & (funds["quarter"] == quarter)]
    if fund_row.empty:
        raise ValueError(f"no fund found for series_id={series_id!r}, quarter={quarter!r}")
    fund_row = fund_row.iloc[0]

    context = {
        "series_id": series_id,
        "quarter": quarter,
        "series_name": fund_row["series_name"],
        "ticker": fund_row["ticker"],
        "yahoo_category": fund_row["yahoo_category"],
        "net_assets": fund_row["net_assets"],
    }

    clusters = load_table("fund_clusters", cfg)
    cluster_row = clusters[(clusters["series_id"] == series_id) & (clusters["quarter"] == quarter)]
    context["cluster_id"] = _first_or_none(cluster_row, "cluster_id")

    validation = load_table("cluster_validation", cfg)
    val_row = validation[validation["quarter"] == quarter]
    context["cluster_purity"] = _first_or_none(val_row, "purity")
    context["cluster_ari"] = _first_or_none(val_row, "adjusted_rand_index")

    top_n = cfg["llm"]["top_peers_to_narrate"]
    peers = load_table("fund_peers", cfg)
    fund_peers = (
        peers[(peers["series_id"] == series_id) & (peers["quarter"] == quarter)]
        .sort_values("peer_rank").head(top_n)
    )
    peer_info = []
    for _, row in fund_peers.iterrows():
        peer_fund = funds[(funds["series_id"] == row["peer_series_id"]) & (funds["quarter"] == quarter)]
        peer_info.append({
            "rank": int(row["peer_rank"]),
            "name": _first_or_none(peer_fund, "series_name") or row["peer_series_id"],
            "category": _first_or_none(peer_fund, "yahoo_category"),
            "similarity": row["cosine_similarity"],
        })
    context["peers"] = peer_info

    overall = load_table("fund_metrics_overall", cfg)
    overall_row = overall[overall["series_id"] == series_id]
    for col in ["cumulative_return", "annualized_volatility", "sharpe_ratio", "max_drawdown"]:
        context[col] = _first_or_none(overall_row, col)

    quarterly = load_table("fund_metrics_quarterly", cfg)
    q_row = quarterly[(quarterly["series_id"] == series_id) & (quarterly["quarter"] == quarter)]
    for col in ["quarterly_return", "return_vs_cluster_median"]:
        context[col] = _first_or_none(q_row, col)

    predictions = load_table("fund_predictions", cfg)
    pred_row = predictions[(predictions["series_id"] == series_id) & (predictions["quarter"] == quarter)]
    context["predicted_probability"] = _first_or_none(pred_row, "predicted_probability")
    context["actual_label"] = _first_or_none(pred_row, "actual_label")

    return context


def _fmt_pct(x) -> str:
    return "unavailable" if x is None or pd.isna(x) else f"{x * 100:.1f}%"


def _fmt_num(x, digits=2) -> str:
    return "unavailable" if x is None or pd.isna(x) else f"{x:.{digits}f}"


def format_context_as_text(context: dict) -> str:
    """Serialize the retrieved facts as clearly-labeled plain text - easier for a small
    local model to ground on cleanly than a raw JSON dump."""
    lines = [
        f"Fund: {context['series_name']} (ticker {context['ticker']}, category "
        f"{context['yahoo_category']}), quarter {context['quarter']}.",
        f"Net assets: {_fmt_num(context['net_assets'], 0)}.",
        "",
        f"Whole-history performance: cumulative return {_fmt_pct(context['cumulative_return'])}, "
        f"annualized volatility {_fmt_pct(context['annualized_volatility'])}, "
        f"Sharpe ratio {_fmt_num(context['sharpe_ratio'])}, "
        f"max drawdown {_fmt_pct(context['max_drawdown'])}.",
        f"This quarter's return: {_fmt_pct(context['quarterly_return'])} "
        f"({_fmt_pct(context['return_vs_cluster_median'])} vs. its peer cluster's median).",
        "",
    ]

    if context["cluster_id"] is None or pd.isna(context["cluster_id"]):
        lines.append("Peer cluster: unavailable this quarter (fund had no equity holdings on file).")
    else:
        lines.append(
            f"Peer cluster: cluster {int(context['cluster_id'])} this quarter "
            f"(cluster-assignment quality this quarter: purity "
            f"{_fmt_num(context['cluster_purity'])}, adjusted Rand index "
            f"{_fmt_num(context['cluster_ari'])} vs. Morningstar-style categories)."
        )
        lines.append("Nearest peers by holdings similarity:")
        for peer in context["peers"]:
            lines.append(
                f"  {peer['rank']}. {peer['name']} ({peer['category']}), "
                f"similarity {_fmt_num(peer['similarity'])}"
            )

    lines.append("")
    if context["predicted_probability"] is None or pd.isna(context["predicted_probability"]):
        lines.append(
            "Next-quarter prediction: unavailable (this fund-quarter is outside the model's "
            "evaluated panel)."
        )
    else:
        lines.append(
            f"Model's predicted probability of underperforming its peer cluster next quarter: "
            f"{_fmt_pct(context['predicted_probability'])}."
        )
        if context["actual_label"] is not None and not pd.isna(context["actual_label"]):
            actual = "did underperform" if context["actual_label"] == 1 else "did NOT underperform"
            lines.append(f"What actually happened next quarter: the fund {actual} its peer cluster.")

    return "\n".join(lines)


def narrate_fund(series_id: str, quarter: str, cfg: dict) -> str:
    context = build_context(series_id, quarter, cfg)
    context_text = format_context_as_text(context)
    client = _get_client(cfg)
    response = client.chat.completions.create(
        model=cfg["llm"]["model_name"],
        temperature=cfg["llm"]["temperature"],
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Here are the retrieved facts:\n\n{context_text}"},
        ],
    )
    return response.choices[0].message.content


def run(cfg: dict) -> None:
    predictions = load_table("fund_predictions", cfg)
    test_predictions = predictions[predictions["split"] == "test"]
    if test_predictions.empty:
        log.warning("no test-split predictions found, skipping example narration")
        return

    top = test_predictions.sort_values(
        ["predicted_probability", "series_id"], ascending=[False, True]
    ).iloc[0]
    series_id, quarter = top["series_id"], top["quarter"]
    log.info(f"narrating example fund {series_id} ({quarter}), "
             f"predicted_probability={top['predicted_probability']:.3f}")

    narration = narrate_fund(series_id, quarter, cfg)
    log.info(f"narration:\n{narration}")
