"""Make the dev/eval database split and save it to splits/split_v1.json (once; it is frozen after that)."""
import argparse
import json
from collections import Counter
from pathlib import Path
from chartbreak.data import load_viseval, make_split, SPLIT_RULE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/visEval_dataset')
    parser.add_argument('--output', default='splits/split_v1.json')
    parser.add_argument('--force', action='store_true', help='overwrite an existing (frozen) split')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and not args.force:
        raise SystemExit(f'{output} already exists. The split is frozen, use --force only if the team agrees.')
    cases = load_viseval(args.dataset)
    seed, dev_dbs, eval_dbs = make_split(cases)
    dev = [c for c in cases if c['db_id'] in set(dev_dbs)]
    ev = [c for c in cases if c['db_id'] in set(eval_dbs)]
    describe = lambda part: dict(
        databases=len({c['db_id'] for c in part}), charts=len({c['chart_id'] for c in part}), queries=len(part),
        queries_by_chart_type=dict(sorted(Counter(c['features']['chart_type'] for c in part).items())))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(
        version='split_v1', rule=SPLIT_RULE, seed=seed, viseval_commit='55162fac5765d8873180d962cce7e20e75b2c092',
        development=describe(dev), evaluation=describe(ev),
        dev_dbs=dev_dbs, eval_dbs=eval_dbs), indent=2) + '\n')
    print(f'seed {seed}: {len(dev)} development / {len(ev)} evaluation queries -> {output}')


if __name__ == '__main__':
    main()
