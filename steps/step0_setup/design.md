# step0_setup — design

## Purpose (traces to Required Output)
Stand up the project skeleton once so every later step plugs in and the whole pipeline is
reproducible from **one command**. This step produces no analysis; it delivers the deterministic
conductor, secret-safe setup, shared config/IO, step stubs, and live browser docs that the Required
Output's "runnable from one deterministic conductor entry point" depends on.

## Deliverables
1. **Secret-safe setup** (before first commit)
   - Merge the existing `.gitignore` with the methodology template (adds `.env`, key/cert patterns,
     `dist/`, `build/`, logs). Keep the existing Python/IDE entries.
   - `.env.example` with **key names only, no values**:
     - `SEC_USER_AGENT` — SEC requires a descriptive UA with contact email (e.g. `FundsPeers you@email`).
     - `LM_STUDIO_BASE_URL` — default `http://localhost:1234/v1`.
     - `LM_STUDIO_API_KEY` — LM Studio accepts any string (e.g. `lm-studio`).
   - Real values live only in `.env` (git-ignored).

2. **Dependencies** — `requirements.txt`, versions pinned:
   `pandas`, `pyarrow`, `duckdb`, `scikit-learn`, `numpy`, `requests`, `yfinance`, `matplotlib`,
   `tqdm`, `openai`, `python-dotenv`, `pytest`, `jupyter`.

3. **Config** — `config.json` holding all knobs (no hardcoded paths / seeds):
   - `seed` (int), `data.quarters` (e.g. `["2022q1", ... "2024q4"]`), `paths.raw`/`paths.processed`,
   - placeholders for later steps: `similarity.n_clusters`, `similarity.top_n_peers`,
     `metrics.risk_free_annual`, `model.rf` params. Later steps read these; step0 just defines them.

4. **Shared library** `fundspeers/`
   - `config.py` — `load_config()` reads `config.json`; `load_env()` reads `.env` via `python-dotenv`.
   - `io.py` — resolve data paths from config; `save_table()`/`load_table()` parquet helpers.
   - `seeding.py` — `seed_everything(seed)` seeds `random` and `numpy` (extend later).

5. **Step stubs** `steps/stepN_name/` — each a package with `__init__.py`, its `design.md` (written
   when that step starts), and a module exposing `def run(cfg): ...`. For step0, steps 1–5 are
   **no-op stubs** that log `"<step>: pending"` and return — so the conductor runs end-to-end.

6. **Deterministic conductor** `conductor.py` (from template, adapted)
   - Loads `config.json`, calls `seed_everything(cfg["seed"])`.
   - `build_pipeline()` returns ordered `(label, run)` for step1..step5.
   - Runs steps under a **tqdm** in-place progress bar (percentage, bar, elapsed).
   - **Final step regenerates the browser docs** (`scripts/render_docs.py`) so they never drift.

7. **Browser-readable docs**
   - Copy `scripts/render_docs.py` verbatim from the skill.
   - `decisions.json` seeded with the key locked decisions (data source, universe, Yahoo role,
     label definition, LLM role) — each with a rationale.
   - `workflow.json` with the 6 steps; step0 `in_progress` → `done` at gate, rest `pending`.

8. **Smoke test** `tests/test_smoke.py` — imports the conductor and runs `main()`; asserts it
   completes without error (the no-op pipeline) and that `render_docs` produces both HTML files.

## Determinism
Single `seed` from `config.json` applied via `seed_everything`; no wall-clock in outputs; paths from
config; versions pinned. The LM Studio call (step5) is the one documented non-deterministic boundary.

## UAT (acceptance for this step)
- `pip install -r requirements.txt` succeeds in the venv.
- `python conductor.py` runs end-to-end, shows a live progress bar, prints each step's "pending",
  and finishes at 100% with no error.
- `python scripts/render_docs.py ...` (also invoked by the conductor) writes `reflection.html` and
  `workflow.html`; both open in a browser and show the decisions + the 6-step flow.
- `pytest` passes the smoke test.
- `git status` shows `.env` is ignored; `.env.example` is tracked; no secrets staged.

## Out of scope for step0
No data download, no analysis, no real `run()` logic — those belong to steps 1–5.
