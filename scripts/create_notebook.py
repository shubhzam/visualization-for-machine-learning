"""Regenerate the checked-in notebook from readable cell definitions."""
import json
from pathlib import Path
cells=[]
def md(text): cells.append(dict(cell_type='markdown',metadata={},source=text.splitlines(True)))
def code(text): cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=text.splitlines(True)))

md('''# ChartBreak • experiment notebook
Run the cells in order. The default is an **offline, simulated demo** with no API calls.
Live mode uses the real VisEval dataset and official chart checkers. Add your provider key,
model ID, and endpoint to `.env`, and build the Docker worker once (see README).
''')
code('''from pathlib import Path
import os, sys, json
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "chartbreak").is_dir() and (p / "configs").is_dir())
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image, SVG, Code
from chartbreak.runner import run
from chartbreak.analysis import load_results, summary, flips
print("Project:", ROOT)
''')
md('''## 1. Choose a run
Keep `LIVE = False` to explore synthetic fixtures. For a small real experiment, change it to
`True` after setup. `LIMIT` is the number of queries **per model and strategy**. With four
strategies, 5 queries need at least 25 API calls, plus S4 self-check/repair calls.
To inspect an existing run without making calls, set `EXISTING_RUN` to its folder.
''')
code('''LIVE = False
LIMIT = 12
EXISTING_RUN = None  # e.g. ROOT / "outputs" / "20261103T..."
run_dir = Path(EXISTING_RUN) if EXISTING_RUN else run("configs/live.json" if LIVE else "configs/demo.json", limit=LIMIT)
records = load_results(run_dir)
manifest = json.loads((run_dir / "manifest.json").read_text())
print("Mode:", manifest["mode"], "| Status:", manifest["status"], "| Run:", run_dir.name)
if manifest["mode"] == "demo":
    print("SIMULATED DATA: these outcomes do not measure any AI model.")
''')
md('''## 2. Model × prompting strategy
Intervals resample whole charts, keeping paraphrases together. Cells with fewer than ten
charts are flagged. Provider errors are excluded from scored rates and shown in the counts.
''')
code('''stats = pd.DataFrame(summary(records))
display(stats.drop(columns=["outcomes"]))
matrix = stats.pivot(index="model", columns="strategy", values="pass_rate")
fig, ax = plt.subplots(figsize=(8, max(2.5, len(matrix) * .8)))
im = ax.imshow(matrix.to_numpy(dtype=float), vmin=0, vmax=1, cmap="Blues", aspect="auto")
ax.set_xticks(range(len(matrix.columns)), matrix.columns)
ax.set_yticks(range(len(matrix.index)), matrix.index)
for i in range(len(matrix)):
    for j in range(len(matrix.columns)):
        ax.text(j, i, f"{matrix.iloc[i,j]:.0%}", ha="center", va="center", color="black")
ax.set_title(f"Pass rate • {manifest['mode'].upper()}")
fig.colorbar(im, ax=ax)
plt.show()
''')
md('''## 3. Find difficult query groups
Choose a model, strategy, and feature below, then rerun this cell.
''')
code('''MODEL = records[0]["model"]
STRATEGY = "S1"
FEATURE = "chart_type"  # hardness, join, sort, binning, three_channels, num_tables
subset = [r for r in records if r["model"] == MODEL and r["strategy"] == STRATEGY]
slices = pd.DataFrame(summary(subset, (FEATURE,)))
display(slices)
if not slices.empty:
    slices.set_index(FEATURE)["pass_rate"].plot.bar(ylim=(0,1), ylabel="Pass rate", title=f"{MODEL} / {STRATEGY} • {manifest['mode'].upper()}")
    plt.tight_layout(); plt.show()
''')
md('''## 4. What did the prompt change fix or break?
Pairs use the same query and model. Choose two strategies. The table includes query IDs
so you can inspect any transition below.
''')
code('''STRATEGY_A, STRATEGY_B = "S1", "S4"
flow_rows = flips(records, MODEL, STRATEGY_A, STRATEGY_B)
display(pd.DataFrame(flow_rows))
print("Fixed:", sum(r["count"] for r in flow_rows if r["source"] != "pass" and r["target"] == "pass"))
print("Broken:", sum(r["count"] for r in flow_rows if r["source"] == "pass" and r["target"] != "pass"))
''')
md('''## 5. Inspect one case
Set `QUERY_ID` to an ID from the flow table. Each strategy shows its generated chart,
final code, checker messages, and repair count. Ground-truth reference rendering and the
linked D3 frontend are future extensions; the backend already exports `dashboard.json`.
''')
code('''import difflib
QUERY_ID = next((r["query_ids"][0] for r in flow_rows if r["source"] != "pass" and r["target"] == "pass"), records[0]["query_id"])
case_rows = [r for r in records if r["query_id"] == QUERY_ID and r["model"] == MODEL and r["strategy"] in {STRATEGY_A,STRATEGY_B}]
for row in case_rows:
    print(row["query"], "\\n", row["strategy"], "→", row["outcome"])
    if row.get("png") and Path(row["png"]).exists():
        display(Image(filename=row["png"]))
    elif row["mode"] == "demo":
        display(SVG(filename=row["svg"]))
    display(pd.DataFrame(row["checks"]))
    if row.get("attempts"):
        display(Code(row["attempts"][-1]["code"], language="python"))
        print("Execution attempts:", len(row["attempts"]))
if len(case_rows) == 2 and all(r.get("attempts") for r in case_rows):
    print("\\n".join(difflib.unified_diff(case_rows[0]["attempts"][-1]["code"].splitlines(), case_rows[1]["attempts"][-1]["code"].splitlines(), fromfile=STRATEGY_A, tofile=STRATEGY_B)))
''')
md('''## 6. Optional failure predictor
Run only after collecting real results across at least five databases. This compares a
chart-type/hardness baseline, logistic regression, and gradient boosting. Database groups
stay together across folds. SHAP explanations are a later extension.
''')
code('''RUN_PREDICTOR = False
if RUN_PREDICTOR:
    from chartbreak.predictor import evaluate_predictors
    display(pd.DataFrame(evaluate_predictors(records)))
print("Saved files:", [p.name for p in run_dir.iterdir() if p.is_file()])
''')
notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3 (ipykernel)',language='python',name='python3'),language_info=dict(name='python',version='3.13')),nbformat=4,nbformat_minor=5)
for i,c in enumerate(cells):c['id']=f'chartbreak-{i:02d}'
Path('notebooks/01_chartbreak.ipynb').write_text(json.dumps(notebook,indent=2))
