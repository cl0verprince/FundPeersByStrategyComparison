"""step8_dashboard/narratives.py - per-cluster phi-4 paragraphs, cached in a table.

Same RAG contract as step5: the prompt contains every fact the model may use; it invents
nothing. Cached so dashboard builds are deterministic and LM-Studio-free after the first
generation (design.md: build determinism must not depend on LLM output stability).
"""
import logging

import pandas as pd

from fundspeers.io import load_table, save_table, table_exists

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a financial-data narrator writing for a financial analyst or advisor. You are "
    "given computed facts about ONE peer group (cluster) of US equity mutual funds, grouped "
    "by what they actually hold. Explain the group in one plain-English paragraph.\n\n"
    "Rules:\n"
    "- Use ONLY the facts given. Never invent a number, holding, or fund.\n"
    "- Descriptive only - no investment advice, no buy/sell/hold language.\n"
    "- One paragraph, no lists.")


def build_cluster_prompt(cluster: dict) -> str:
    holdings = ", ".join(f"{h['issuer']} ({h['weight']:.1%})"
                         for h in cluster["top_holdings"][:10]) or "unavailable"
    return (
        f"Cluster name: {cluster['short_title']}.\n"
        f"Members: {cluster['member_count']} funds; dominant category "
        f"{cluster['dominant_category']} ({cluster['dominant_share']:.0%} of members).\n"
        f"Average annualized volatility: {cluster['avg_volatility']:.1%}. "
        f"Average Sharpe ratio: {cluster['avg_sharpe']:.2f}. "
        f"Average max drawdown: {cluster['avg_max_drawdown']:.1%}.\n"
        f"Most-held stocks across members (average weight): {holdings}.")


def _get_client(cfg: dict):
    from openai import OpenAI

    from fundspeers.config import load_env
    env = load_env()
    return OpenAI(base_url=env["LM_STUDIO_BASE_URL"], api_key=env["LM_STUDIO_API_KEY"])


def generate_one(client, cfg: dict, cluster: dict) -> str:
    response = client.chat.completions.create(
        model=cfg["llm"]["model_name"], temperature=cfg["llm"]["temperature"],
        messages=[{"role": "system", "content": _SYSTEM_PROMPT},
                  {"role": "user", "content": build_cluster_prompt(cluster)}])
    return response.choices[0].message.content


def get_narratives(cfg: dict, payload_clusters: list, mode: str = "cached",
                   quarter: str = "2024q4") -> dict:
    if mode == "skip":
        return {}
    cached = {}
    if mode != "regenerate" and table_exists("dashboard_narratives", cfg):
        tbl = load_table("dashboard_narratives", cfg)
        cached = {int(r["cluster_id"]): r["narrative"]
                  for _, r in tbl[tbl["quarter"] == quarter].iterrows()}

    missing = [c for c in payload_clusters if int(c["cluster_id"]) not in cached]
    if missing:
        log.info(f"generating {len(missing)} narrative(s) via LM Studio "
                 f"({'regenerate' if mode == 'regenerate' else 'cache misses'})")
        client = _get_client(cfg)
        for cluster in missing:
            cached[int(cluster["cluster_id"])] = generate_one(client, cfg, cluster)

        updated = pd.DataFrame(
            [{"cluster_id": cid, "quarter": quarter, "narrative": text}
             for cid, text in sorted(cached.items())])
        if table_exists("dashboard_narratives", cfg):
            existing = load_table("dashboard_narratives", cfg)
            existing = existing[existing["quarter"] != quarter]
            updated = pd.concat([existing, updated], ignore_index=True)
        updated = updated.sort_values(["quarter", "cluster_id"]).reset_index(drop=True)
        save_table(updated, "dashboard_narratives", cfg)
    return cached
