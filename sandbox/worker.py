"""Runs ONLY inside the disposable container, never on the host."""
import json
import runpy
import traceback
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype'] = 'path'
import matplotlib.pyplot as plt

try:
    runpy.run_path('/input/generated.py', run_name='__main__')
    if len(plt.get_fignums()) != 1:
        raise ValueError('Expected exactly one Matplotlib figure.')
    plt.gcf().savefig('/output/chart.svg', format='svg')
    plt.gcf().savefig('/output/chart.png', format='png', dpi=110)
    result = dict(ok=True, error='')
except BaseException:
    result = dict(ok=False, error=traceback.format_exc()[-6000:])
Path('/output/execution.json').write_text(json.dumps(result))
