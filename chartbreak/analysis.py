"""Paired outcomes and chart-clustered bootstrap intervals (standard library)."""
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

EVALUATED = {'pass','invalid','illegal_chart','illegal_data','illegal_order','unparseable'}


def load_results(run_dir):
    return [json.loads(line) for line in (Path(run_dir)/'results.jsonl').read_text().splitlines() if line.strip()]


def summary(records, group_by=('model','strategy'), seed=42, resamples=1000):
    groups = defaultdict(list)
    for row in records:
        key = tuple(row.get(k, row.get('features',{}).get(k)) for k in group_by)
        groups[key].append(row)
    rows = []
    for key, group in sorted(groups.items(), key=lambda x: str(x[0])):
        scored = [r for r in group if r['outcome'] in EVALUATED and not r.get('truncated')]
        clusters = defaultdict(list)
        for row in scored:
            clusters[row['chart_id']].append(int(row['outcome']=='pass'))
        values = list(clusters.values()); rng = random.Random(seed); samples = []
        if len(values) >= 2:
            for _ in range(resamples):
                selected = [rng.choice(values) for _ in values]
                samples.append(sum(map(sum,selected))/sum(map(len,selected)))
            samples.sort()
        n = len(scored)
        rows.append(dict(zip(group_by,key), n_total=len(group), n_evaluated=n,
            n_excluded=len(group)-n, n_charts=len(values),
            pass_rate=sum(r['outcome']=='pass' for r in scored)/n if n else None,
            ci_low=samples[int(.025*len(samples))] if samples else None,
            ci_high=samples[min(len(samples)-1,int(.975*len(samples)))] if samples else None,
            small_sample=len(values)<10, outcomes=dict(Counter(r['outcome'] for r in group))))
    return rows


def flips(records, model, strategy_a='S1', strategy_b='S4'):
    a = {r['query_id']:r for r in records if r['model']==model and r['strategy']==strategy_a}
    b = {r['query_id']:r for r in records if r['model']==model and r['strategy']==strategy_b}
    flows = defaultdict(list)
    for query in sorted(a.keys() & b.keys()):
        if all(r[query]['outcome'] in EVALUATED and not r[query].get('truncated') for r in (a, b)):
            flows[(a[query]['outcome'],b[query]['outcome'])].append(query)
    return [dict(source=s,target=t,count=len(ids),query_ids=ids) for (s,t),ids in sorted(flows.items())]


def export(run_dir):
    run_dir = Path(run_dir); records = load_results(run_dir)
    stats = summary(records)
    (run_dir/'summary.json').write_text(json.dumps(stats,indent=2))
    fields = ['query_id','chart_id','db_id','model','strategy','mode','outcome','elapsed_seconds']
    with (run_dir/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    payload = dict(mode=records[0]['mode'] if records else None, summary=stats,
        feature_slices={f:summary(records,('model','strategy',f)) for f in ['chart_type','hardness','join','sort','binning','three_channels','multi_table','num_tables']},
        flows={m:flips(records,m) for m in sorted({r['model'] for r in records})},cases=records)
    (run_dir/'dashboard.json').write_text(json.dumps(payload,indent=2))
    return payload
