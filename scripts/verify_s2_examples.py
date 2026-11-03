"""Run every S2 example through the sandbox and the checkers; S2 only uses the ones that pass."""
import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path
from chartbreak.data import load_viseval, config_split, eligible_examples, reference_code
from chartbreak.execution import DockerRunner, first_failure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/live.json')
    parser.add_argument('--output', default='configs/s2_verified_examples.json')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    cases = load_viseval(config['dataset_dir'])
    dev, _ = config_split(cases, config)
    runner = DockerRunner(config.get('sandbox_image', 'chartbreak-worker:0.2'), config.get('timeout_seconds', 60))
    runner.preflight()
    verified, rejected = [], {}
    Path('outputs').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir='outputs') as folder:
        for i, case in enumerate(eligible_examples(dev)):
            directory = Path(folder) / str(i)
            execution = runner.execute(reference_code(case), case, directory)
            checks = [dict(aspect='code execution', answer=execution['ok'], rationale=execution['error'])]
            if execution['ok']:
                checks += runner.evaluate(execution, case, directory)
            outcome = first_failure(checks)
            if outcome == 'pass':
                verified.append(case['query_id'])
            else:
                rejected[case['query_id']] = outcome
    by_type = Counter(c['features']['chart_type'] for c in eligible_examples(dev) if c['query_id'] in set(verified))
    Path(args.output).write_text(json.dumps(dict(
        description='Development examples whose reference code passed VisEval checks in the sandbox.',
        split_file=config.get('split_file'),
        sandbox_image=runner.image, sandbox_image_id=runner.image_id,
        n_candidates=len(verified) + len(rejected), n_verified=len(verified),
        verified_by_chart_type=dict(sorted(by_type.items())),
        rejected_by_outcome=dict(Counter(rejected.values())),
        verified=verified, rejected=rejected), indent=2) + '\n')
    print(f'{len(verified)} of {len(verified) + len(rejected)} examples verified -> {args.output}')


if __name__ == '__main__':
    main()
