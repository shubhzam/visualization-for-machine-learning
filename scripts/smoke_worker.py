"""Integration check: execute known Python and score it with real VisEval; no API calls."""
from pathlib import Path
import tempfile
from chartbreak.data import load_viseval, reference_code
from chartbreak.execution import DockerRunner, first_failure

cases=load_viseval('data/visEval_dataset')
case=next(c for c in cases if c['features']['chart_type'].lower()=='bar' and not c['features']['binning'] and not c['ground_truth']['vis_obj'].get('classify'))
runner=DockerRunner();runner.preflight()
Path('outputs').mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(dir='outputs') as folder:
    root=Path(folder)
    execution=runner.execute(reference_code(case),case,root/'good')
    assert execution['ok'],execution
    checks=[dict(aspect='code execution',answer=True,rationale='')]+runner.evaluate(execution,case,root/'good')
    assert first_failure(checks)=='pass',checks
    bad=runner.execute('raise RuntimeError("intentional smoke test")',case,root/'bad')
    assert not bad['ok'] and 'intentional smoke test' in bad['error'],bad
    print('PASS: real Matplotlib chart + official VisEval checks; execution errors captured.')
