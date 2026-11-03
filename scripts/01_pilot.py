"""Pilot: draw every ground-truth chart with a simple template and run VisEval's checks on it (no LLM).

Needs VisEval, so run it inside the sandbox image:
    docker run --rm --network none -v "$PWD:/repo:ro" -v "$PWD/results:/repo/results" -w /repo chartbreak-worker:0.2 python scripts/01_pilot.py
"""
import argparse
import io
import json
import platform
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from viseval.evaluate import Evaluator


def draw(o, chart):
    # plots the stored order, so charts that need a sort fail the order check
    fig, ax = plt.subplots()
    xs, ys, cl = o['x_data'], o['y_data'], o['classify']
    lab = lambda v: [str(x) for x in v]
    if chart == 'Pie': ax.pie(ys[0], labels=lab(xs[0]))
    elif chart == 'Bar': ax.bar(lab(xs[0]), ys[0])
    elif chart == 'Line': ax.plot(lab(xs[0]), ys[0])
    elif chart == 'Scatter': ax.scatter(xs[0], ys[0])
    elif chart == 'Stacked Bar':
        b = np.zeros(len(xs[0]))
        for n, y in zip(cl, ys):
            ax.bar(lab(xs[0]), y, bottom=b, label=str(n)); b += np.array(y, float)
        ax.legend()
    elif chart == 'Grouping Line':
        for n, y in zip(cl, ys): ax.plot(lab(xs[0]), y, label=str(n))
        ax.legend()
    elif chart == 'Grouping Scatter':
        for n, x, y in zip(cl, xs, ys): ax.scatter(x, y, label=str(n))
        ax.legend()
    if chart != 'Pie': ax.set_xlabel(o['x_name']); ax.set_ylabel(o['y_name'])
    f = io.StringIO(); plt.savefig(f, format='svg'); plt.close('all')
    return f.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/visEval_dataset')
    parser.add_argument('--output-dir', default='results')
    args = parser.parse_args()
    ev = Evaluator()
    rows = []
    for cid, inst in json.loads(Path(args.dataset, 'visEval.json').read_text()).items():
        try:
            svg, error = draw(inst['vis_obj'], inst['chart']), None
        except Exception as e:
            svg, error = None, f'{type(e).__name__}: {e}'
            plt.close('all')
        for qi in range(len(inst['nl_queries'])):
            row = dict(query_id=f'{cid}:{qi}', chart_id=cid, chart_type=inst['chart'])
            if svg is None:
                rows.append(dict(row, outcome='template_error', rationale=error)); continue
            ctx = {'svg_string': svg, 'library': 'matplotlib'}
            res = [ev.surface_form_check(ctx)]
            gt = {'chart': inst['chart'], 'vis_obj': inst['vis_obj'], 'meta_info': inst['query_meta'][qi]}
            if res[0].answer:
                res += ev.legality_check(ctx, gt)
            fail = next((r for r in res if not r.answer), None)
            rows.append(dict(row, outcome='pass' if fail is None else fail.aspect,
                             rationale='' if fail is None else str(fail.rationale)))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    n, passed = len(rows), sum(r['outcome'] == 'pass' for r in rows)
    summary = dict(
        description='Ground-truth template charts scored by VisEval surface-form and legality checks (no LLM).',
        n_queries=n, n_pass=passed, pass_rate=round(passed / n, 4),
        first_failure_counts=dict(Counter(r['outcome'] for r in rows if r['outcome'] != 'pass')),
        failures_by_chart_type={t: dict(Counter(r['outcome'] for r in rows if r['chart_type'] == t and r['outcome'] != 'pass'))
                                for t in sorted({r['chart_type'] for r in rows})},
        environment=dict(python=platform.python_version(), matplotlib=matplotlib.__version__, numpy=np.__version__))
    (out / 'pilot_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    flags = [dict(query_id=r['query_id'], chart_type=r['chart_type'], rationale=r['rationale'])
             for r in rows if r['outcome'] == 'data check']
    (out / 'pilot_checker_flags.json').write_text(json.dumps(dict(
        description='Correct ground-truth charts rejected by the data check: candidate checker false failures (seed the audit; flagged in V4).',
        n=len(flags), queries=flags), indent=2) + '\n')
    (out / 'pilot_outcomes.json').write_text('[\n' + ',\n'.join(json.dumps(r) for r in rows) + '\n]\n')
    print(f'{passed}/{n} pass ({passed / n:.1%}); first failures: {summary["first_failure_counts"]}')


if __name__ == '__main__':
    main()
