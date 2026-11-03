"""Explicit synthetic fixtures: no LLM calls and no execution of generated code."""
import json
from pathlib import Path


def cases():
    return [dict(query_id=f'demo-{i}:0', chart_id=f'demo-{i}', db_id=f'demo-db-{i}',
                 query=f'DEMO: Show counts by category, example {i}.', tables=[],
                 features=dict(chart_type=['Bar','Pie','Line'][i % 3], hardness=['Easy','Medium'][i % 2],
                 join=i % 2 == 0, sort=i % 3 == 0, binning=False, three_channels=False,
                 num_tables=1 + i % 2, query_length=8)) for i in range(12)]


def record(case, strategy, model, folder):
    i = int(case['chart_id'].split('-')[-1])
    outcomes = {
        'S1': ['pass','invalid','illegal_chart','illegal_data','illegal_order','pass'],
        'S2': ['pass','invalid','pass','illegal_data','illegal_order','illegal_chart'],
        'S3': ['pass','invalid','pass','pass','pass','invalid'],
        'S4': ['pass','pass','illegal_chart','illegal_data','illegal_order','illegal_data']}
    outcome = outcomes[strategy][i % 6]
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="520" height="240"><rect width="520" height="240" fill="#f5f7fb"/><text x="24" y="32" font-family="sans-serif">SIMULATED DEMO — not an LLM chart</text>'
    for j, h in enumerate([65,110,80]):
        svg += f'<rect x="{55+j*145}" y="{205-h}" width="70" height="{h}" fill="#5276cc"/>'
    svg += '</svg>'
    folder.mkdir(parents=True)
    (folder / 'chart.svg').write_text(svg)
    code = '# SIMULATED demo fixture; no model generated or executed this code.\n'
    (folder / 'generated.py').write_text(code)
    return dict(**case, model=model['name'], model_id=model['model'], strategy=strategy,
        mode='demo', outcome=outcome, checks=[dict(aspect='simulated fixture', answer=outcome=='pass', rationale='Synthetic outcome for testing the interface only.')],
        attempts=[dict(code=code, usage={}, execution=dict(ok=outcome!='invalid', error='Simulated outcome'))],
        svg=str(folder/'chart.svg'), png=None, code_path=str(folder/'generated.py'),
        readability='not_evaluated', example_ids=[], elapsed_seconds=0)
