"""Separate process: generated code never gets ground truth or checker output."""
import json
from pathlib import Path
from viseval import Evaluator

evaluator = Evaluator()
context = {'library': 'matplotlib', 'svg_string': Path('/input/chart.svg').read_text()}
ground_truth = json.loads(Path('/truth/ground_truth.json').read_text())
checks = [evaluator.surface_form_check(context)]
if all(x.answer for x in checks):
    checks += evaluator.legality_check(context, ground_truth)
Path('/output/checks.json').write_text(json.dumps([x.get_json() for x in checks], default=str))
