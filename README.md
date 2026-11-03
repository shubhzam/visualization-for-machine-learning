# ChartBreak

A Python experiment backbone for Team 11's proposal. Use the notebook to run experiments,
compare prompts, and inspect generated charts. The original proposal files are preserved.

## Start here

Open `notebooks/01_chartbreak.ipynb` with the project's `.venv` Python interpreter.
Run all cells for an offline demo. Demo outcomes and charts are **simulated**, not research results.

Clone the repository, then install the notebook environment:

```bash
git clone https://github.com/shubhzam/visualization-for-machine-learning.git
cd visualization-for-machine-learning
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[notebook]'
jupyter lab notebooks/01_chartbreak.ipynb
```

The CLI demo needs only Python 3.10+:

```bash
python3 -m chartbreak run --config configs/demo.json --limit 12
```

## Connect a real model

1. Start Docker Desktop and build the isolated worker once:

   ```bash
   docker build -t chartbreak-worker:0.2 sandbox
   ```

2. Run `cp .env.example .env`, then fill in `CHARTBREAK_API_KEY`,
   `CHARTBREAK_MODEL`, and `CHARTBREAK_BASE_URL`. The model must support text
   Chat Completions at an OpenAI-compatible endpoint. Keys stay in `.env`, which is ignored
   by Git. Model ID is explicit because availability depends on your provider/account.

3. Download the official VisEval dataset (the dataset is excluded from Git):

   ```bash
   python scripts/prepare_data.py
   ```

4. Run a small real experiment:

   ```bash
   python -m chartbreak run --config configs/live.json --limit 5
   ```

   Or set `LIVE = True` and `LIMIT = 5` in the notebook. This makes paid API calls.
   Five queries across four strategies require 30–35 calls with the default repair cap.
   Provider failures are recorded separately and excluded from correctness rates.
   Runs use new directories, so running again repeats model calls.

Add additional entries to `configs/live.json` for the two open models (Qwen2.5-Coder-7B-Instruct
and Llama-3.1-8B-Instruct), served with vLLM on Colab GPUs. Each model can use
its own `model_env`, `base_url_env`, and `api_key_env`. A vLLM server can use the same
client through an accessible HTTPS endpoint or a localhost SSH tunnel. Set `temperature`
to `null` only if the provider does not accept it; record that deviation in the report.
Do not put actual keys directly in JSON configuration.

## Implemented

- **S1:** question, table columns, five preview rows, full CSVs available to generated code.
- **S2:** three examples retrieved by word overlap from development databases. Examples
  use benchmark SQL and chart templates for simple bar, pie, line, and scatter charts.
  `scripts/verify_s2_examples.py` runs every candidate through the sandbox and the official
  checkers; S2 retrieves only those that pass (`configs/s2_verified_examples.json`).
  Stacked and grouped chart examples are not yet available.
- **S3:** a separate planning call followed by a code-generation call.
- **S4:** S1 followed by up to two self-check/repair calls. Only execution errors and the
  model's own code are returned; ground-truth legality feedback is never sent to it.
- Frozen database split in `splits/split_v1.json` (made by `scripts/02_make_split.py`):
  29 development databases (523 queries) and 117 evaluation databases (2,001 queries).
- Optional reproducible stratified sample (`sample_per_chart_type` in the config): a fixed
  number of evaluation charts per chart type, one paraphrase each. Without it, `--limit`
  takes the first queries in file order, which come from very few charts and databases.
- Real Matplotlib execution in containers with networking disabled, read-only inputs,
  no API keys, CPU/memory/process limits, and execution timeouts.
- Official VisEval surface-form, deconstruction, chart-type, data, and order checks.
  VisEval is pinned to commit `55162fac5765d8873180d962cce7e20e75b2c092`.
- First-failure labels and raw checker messages. `unparseable` is kept separately so
  checker failures remain visible. Readability is explicitly `not_evaluated`.
- Persisted prompts, responses, usage, code, execution attempts, SVG/PNG charts,
  configuration, model IDs, dataset fingerprint, and database split.
- Counts, chart-clustered 95% bootstrap intervals, paired strategy transitions, and
  feature slices. Small chart groups are flagged; one-chart intervals are omitted.
- Notebook comparison views and case inspection with code diffs.
- Optional baseline/logistic/gradient-boosted failure predictors with database-grouped
  five-fold evaluation. Synthetic demo results are rejected by this evaluation.

## Project layout

```text
chartbreak/
  data.py          VisEval loader, database split, development examples
  prompts.py       Prompt construction and code extraction
  providers.py     API client and environment configuration
  execution.py     Isolated execution and official checker adapter
  runner.py        Model × strategy × query experiment loop
  analysis.py      Summaries, uncertainty, paired transitions, exports
  predictor.py     Optional failure prediction
  demo.py          Clearly labeled synthetic fixtures
configs/           Demo and live run settings
notebooks/         Main runnable notebook
sandbox/           Container and execution/checker workers
scripts/           Dataset setup and integration smoke test
tests/             Experiment logic tests
outputs/<run-id>/  Results and per-case artifacts (ignored by Git)
```

Each run contains `manifest.json`, `results.jsonl`, `results.csv`, `summary.json`, and
`dashboard.json`. The last file supplies summaries, slices, transitions, and case details
for a future D3 frontend. A result is keyed by query ID, model, and strategy. All strategies
receive the same selected evaluation queries; paraphrases share their chart ID.

The proposal's full research deliverables still require a linked D3 frontend,
CoML4VIS reference runs, few-shot examples for stacked and grouped charts, readability
sampling, paired significance tests with multiple-comparison correction, rerun churn, a
checker audit, and usage scenarios with a notebook baseline plus formative expert feedback
(replacing the controlled user study). The failure predictor and SHAP are stretch goals.
The backend is a runnable foundation, not a claim that those studies are complete.

## Verification

Prepare the dataset first if you have not downloaded it yet.

```bash
python scripts/prepare_data.py
python -m unittest discover -s tests -v
python scripts/smoke_worker.py
```

The first checks dataset shape, split isolation, prompt boundaries, self-repair, failure
labels, statistical summaries, and saved demo outputs. The second needs the built Docker
image and checks a real Matplotlib chart against VisEval, plus an intentional code error.
Live-provider calls require your credentials and have not been run during setup.

The ground-truth pilot (no LLM) runs inside the sandbox image and writes `results/pilot_*.json`:

```bash
docker run --rm --network none -v "$PWD:/repo:ro" -v "$PWD/results:/repo/results" -w /repo chartbreak-worker:0.2 python scripts/01_pilot.py
```

To try the whole pipeline without an API key, `python -m chartbreak run --config configs/mock.json --limit 70`
uses a fake offline model with the real data, sandbox and checkers. Its numbers mean nothing.

Ground truth is only available to a separate checker container after generation finishes.
Container/environment failures stop the run and leave its manifest marked interrupted;
completed result rows remain available. Automatic resume and distributed execution are
future additions. Inspect partial runs with `python -m chartbreak analyze outputs/<run-id>`.

Data/toolkit source: [Microsoft VisEval](https://github.com/microsoft/VisEval) (MIT).
The downloaded data's source URL and checksum are in `data/VISEVAL_SOURCE.txt`, with its
license in `data/VISEVAL_LICENSE`. Generated artifacts may contain the benchmark data and
model responses; `.env` values are not serialized.
