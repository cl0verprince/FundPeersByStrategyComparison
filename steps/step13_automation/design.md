# step13_automation — design

## Purpose
Quarterly refresh without a human driving it: when the SEC publishes a new N-PORT (and RR)
quarterly data set, ingest it, advance the pipeline, rebuild the dashboard, and commit —
**scheduled** (primary) and **on demand** (same entry point, run manually).

## Reality constraints (learned the hard way this week)
- **Publication lag:** N-PORT filings are due ~60 days after quarter end; the bulk data set
  lands after that. "End of quarter" in practice = poll from ~2.5 months after quarter end.
- **Local machines sleep:** two long runs died to machine sleep / process reaping. The
  scheduled vehicle is a **cloud routine** (user's choice), not a local timer. On-demand
  local runs are foreground or short-lived detached with a completion sentinel.
- **Single-writer DuckDB:** the refresh must be the DB's only client while it runs.
- **Dormant config trap (final-review finding):** `full.n_clusters: 40` is consumed by the
  orchestrator, not by `similarity.run` defaults — the advance-quarter path MUST pass it
  explicitly (this design makes that a checklist item, and the runbook encodes it).

## The advance-quarter mode (on-demand entry point)
`python -m steps.step13_automation.advance` — deterministic conductor for one refresh:
1. **Probe**: is a NEW N-PORT quarter published beyond `data.quarters`' last entry?
   (HEAD request on the next quarter's URL, N-PORT pattern; likewise the RR quarter.)
   If not: log and exit 0 — the routine treats "nothing new" as success.
2. **Extend config**: append the new quarter to `data.quarters` (and the RR year if new).
3. **Re-ingest `_full`** with the step10 parameters (relaxed pool, metadata reuse now
   INCLUDING `funds_full` itself as a reuse source, uncapped). ZIP cache makes this
   incremental in practice; Yahoo lookups only for genuinely new candidates.
4. **Re-cluster + metrics**: `similarity.run(_full, n_clusters=cfg["full"]["n_clusters"],
   top_n_peers=15, require_segment="strategy", save_coords=True)`; `metrics.run(_full)`.
5. **Fees refresh**: step9's acquire (new RR quarter) → parse → point-in-time join.
6. **Re-evaluate**: step10's build (validation-first out-of-time scoring of the previous
   official model on the newly-realized quarter, then retrain) + step9's evaluation.
7. **Dashboard rebuild** (step8/12 build, narratives cache reused; new clusters get
   narratives generated if LM Studio is up, placeholders otherwise — never a failure).
8. **Docs**: append the quarter's numbers to a rolling `refresh_log` table + decisions.json
   entry; render docs; commit everything with a standard message. Never push without the
   configured flag (`--push` off by default).

## The scheduled vehicle (primary) — REVISED at the gate (2026-07-16)
As designed, the routine would have run the full refresh in the cloud. The first live run
exposed the data-locality constraint: a cloud sandbox gets only the git checkout, but the
5.6 GB DuckDB, the cached N-PORT ZIPs, and the ~137k resolved Yahoo lookups are local-only
(gitignored) — a from-scratch cloud rebuild is hours of rate-limited re-resolution, and a
commit-without-push dies with the sandbox.

The scheduled vehicle is therefore a **probe-and-notify cloud routine**
(`trig_014DJXd45LMNXuwhUTzoMR6i`, monthly, 1st at 06:00 UTC): it HEAD-probes the next
N-PORT quarter's URL (probing forward from the last quarter known at creation; its report
is advisory) and reports "published — run the advance locally" or "nothing new". The
refresh itself is the **local on-demand run** of `advance.py`, where the data lives. LM
Studio narrative behavior on refresh: new clusters get real narratives when phi-4 is up,
placeholders otherwise — never a failure (enforced in narratives.py since commit 971590d;
placeholders are not cached, so a later healthy build regenerates them).

## Out of scope
- Auto-push (human reviews the refresh commit before publishing).
- Model re-promotion decisions (the refresh reports; promotion stays a human call).
- Backfilling quarters older than the configured window.

## UAT (acceptance)
- `advance` on the current state (no new quarter) exits 0 with "nothing new" logged.
- A dry-run flag (`--dry-run`) prints the plan (which quarter, which steps) without writing.
- The runbook checklist in this design is encoded in the module docstring.
- A scheduled routine exists, monthly (probe-and-notify form; see the revised section
  above). MET 2026-07-16. The full refresh's own first live run happened early — 2026q2
  published ahead of the expected ~2026-09 window and was refreshed successfully after
  two live-run fixes (commits 834156c, 971590d).
