"""Normalize VisEval without exposing evaluation answers to model prompts."""
import csv
import hashlib
import json
import random
import re
from pathlib import Path


def split_databases(cases, seed=42, fraction=0.2):
    databases = sorted({c['db_id'] for c in cases})
    if len(databases) < 2 or not 0 < fraction < 1:
        raise ValueError('Need at least two databases and a development fraction between 0 and 1.')
    random.Random(seed).shuffle(databases)
    n = max(1, min(len(databases) - 1, round(len(databases) * fraction)))
    dev = set(databases[:n])
    return [c for c in cases if c['db_id'] in dev], [c for c in cases if c['db_id'] not in dev]


SPLIT_RULE = ('sort the database ids; for seed = 0, 1, 2, ... take random.Random(seed).sample(ids, 29) '
              'as the development pool; keep the first seed whose pool has all 7 chart types '
              'and 17-23% of the queries')


def make_split(cases, n_dev=29, low=0.17, high=0.23, max_seed=1000):
    ids = sorted({c['db_id'] for c in cases})
    types = {c['features']['chart_type'] for c in cases}
    for seed in range(max_seed):
        dev = set(random.Random(seed).sample(ids, n_dev))
        dev_cases = [c for c in cases if c['db_id'] in dev]
        if {c['features']['chart_type'] for c in dev_cases} == types and low <= len(dev_cases) / len(cases) <= high:
            return seed, sorted(dev), sorted(set(ids) - dev)
    raise ValueError('No seed satisfies the split rule.')


def apply_split(cases, path):
    split = json.loads(Path(path).read_text())
    dev_dbs, eval_dbs = set(split['dev_dbs']), set(split['eval_dbs'])
    if dev_dbs & eval_dbs or dev_dbs | eval_dbs != {c['db_id'] for c in cases}:
        raise ValueError(f'{path} does not partition the dataset databases.')
    return [c for c in cases if c['db_id'] in dev_dbs], [c for c in cases if c['db_id'] in eval_dbs]


def config_split(cases, config):
    if config.get('split_file'):
        return apply_split(cases, config['split_file'])
    return split_databases(cases, config['seed'], config.get('development_fraction', .2))


def load_viseval(folder):
    folder = Path(folder).resolve()
    raw = json.loads((folder / 'visEval.json').read_text())
    tables = json.loads((folder / 'databases/db_tables.json').read_text())
    multiple = set(json.loads((folder / 'visEval_multiple.json').read_text()))
    cases = []
    for chart_id, row in raw.items():
        vql = row['vis_query']['VQL']
        # Same relevant-table selection as official Dataset.__get_tables.
        names = [t for t in tables[row['db_id']] if t.lower() in vql.lower().split()]
        paths = [str(folder / 'databases' / row['db_id'] / (t + '.csv')) for t in names]
        if not paths:
            raise ValueError(f'No relevant tables for chart {chart_id}')
        for i, query in enumerate(row['nl_queries']):
            meta = row['query_meta'][i]
            cases.append(dict(
                query_id=f'{chart_id}:{i}', chart_id=chart_id, db_id=row['db_id'],
                query=query, tables=paths,
                features=dict(chart_type=row['chart'], hardness=row['hardness'],
                    join=bool(re.search(r'\bjoin\b', vql, re.I)),
                    sort=row['vis_obj']['sort'] is not None,
                    binning=bool(row['vis_query']['data_part']['binning']),
                    # third channel = classify is set (channel_specified is only what the query names)
                    three_channels=bool(row['vis_obj'].get('classify')),
                    multi_table=chart_id in multiple, num_tables=len(paths), query_length=len(query.split())),
                ground_truth=dict(chart=row['chart'], vis_obj=row['vis_obj'], meta_info=meta),
                reference_sql=row['vis_query']['data_part']['sql_part']))
    return cases


def public_input(case):
    previews = []
    for path in case['tables']:
        with open(path, newline='', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            columns = next(reader)
            rows = []
            for _, row in zip(range(5), reader):
                rows.append(row)
        previews.append(dict(path='/data/' + Path(path).name, columns=columns, preview=rows))
    return dict(query=case['query'], tables=previews)


def reference_code(case):
    """Development-only worked examples for simple, non-binned charts."""
    chart = case['features']['chart_type'].lower()
    names = [Path(p).stem for p in case['tables']]
    start = 'import sqlite3\nimport pandas as pd\nimport matplotlib.pyplot as plt\ncon = sqlite3.connect(":memory:")\n'
    for name in names:
        start += f'pd.read_csv({("/data/" + name + ".csv")!r}).to_sql({name!r}, con, index=False)\n'
    start += f'df = pd.read_sql_query({case["reference_sql"]!r}, con)\nx, y = df.iloc[:, 0], df.iloc[:, 1]\nplt.figure(figsize=(8, 5))\n'
    plot = {'bar': 'plt.bar(x.astype(str), y)', 'pie': 'plt.pie(y, labels=x.astype(str))',
            'line': 'plt.plot(x, y)', 'scatter': 'plt.scatter(x, y)'}[chart]
    return start + plot + '\nplt.tight_layout()\n'


def stratified_sample(evaluation, per_chart_type, seed):
    # n charts per chart type, one paraphrase each. note the overall rate is not benchmark-weighted
    rng, by_type = random.Random(seed), {}
    for c in evaluation:
        by_type.setdefault(c['features']['chart_type'], {}).setdefault(c['chart_id'], []).append(c)
    selected = []
    for chart_type in sorted(by_type):
        charts = by_type[chart_type]
        for chart_id in rng.sample(sorted(charts), min(per_chart_type, len(charts))):
            selected.append(rng.choice(charts[chart_id]))
    return selected


def eligible_examples(dev):
    return [c for c in dev if c['features']['chart_type'].lower() in {'bar','pie','line','scatter'}
            and not c['features']['binning'] and not c['ground_truth']['vis_obj'].get('classify')]


def select_examples(dev, case, count=3, allowed=None):
    # Lexical retrieval, with database/ID tie-breaking; only development data enters.
    words = set(re.findall(r'\w+', case['query'].lower()))
    eligible = [c for c in eligible_examples(dev) if allowed is None or c['query_id'] in allowed]
    eligible.sort(key=lambda c: (-len(words & set(re.findall(r'\w+', c['query'].lower()))), c['query_id']))
    selected, charts = [], set()
    for c in eligible:
        if c['chart_id'] not in charts:
            charts.add(c['chart_id']); selected.append(c)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError('Need three distinct eligible development charts for S2.')
    return selected


def fingerprint(cases):
    h = hashlib.sha256(json.dumps(cases, sort_keys=True).encode())
    for path in sorted({p for c in cases for p in c['tables']}):
        h.update(Path(path).name.encode()); h.update(Path(path).read_bytes())
    return h.hexdigest()
