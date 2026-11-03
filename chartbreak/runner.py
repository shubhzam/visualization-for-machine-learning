import hashlib
import json
import os
import platform
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from . import __version__, demo
from .data import load_viseval, split_databases, select_examples, fingerprint
from .execution import DockerRunner, InfrastructureError, first_failure
from .prompts import messages, extract_code, PROMPT_VERSION
from .providers import ChatClient, ProviderError, load_env


def run_condition(case, strategy, client, runner, directory, examples=(), max_repairs=2):
    conversation = messages(case, strategy, examples)
    attempts, calls = [], []
    def call():
        text, usage = client.complete(conversation)
        calls.append(dict(messages=list(conversation), response=text, usage=usage))
        directory.mkdir(parents=True, exist_ok=True)
        (directory/'calls.json').write_text(json.dumps(calls, indent=2))
        conversation.append(dict(role='assistant', content=text))
        return text
    if strategy == 'S3':
        conversation.append(dict(role='user', content='First give a short plan: joins, filters, aggregation, chart type, axes, and sorting. Do not write code yet.'))
        call()
        conversation.append(dict(role='user', content='Now implement your plan. Return only fenced Python code.'))
    text = call()
    for attempt in range(1 + (max_repairs if strategy == 'S4' else 0)):
        code = extract_code(text)
        target = directory / f'attempt-{attempt}'
        execution = runner.execute(code, case, target)
        attempts.append(dict(code=code, execution=execution))
        # Self-repair sees execution errors and its own reasoning, never legality results.
        if strategy != 'S4' or attempt == max_repairs:
            break
        conversation.append(dict(role='user', content=(
            'Execution succeeded. Review the original question and your own code for mistakes. '
            'Return exactly KEEP if no changes are needed; otherwise return corrected fenced Python code.'
            if execution['ok'] else 'Execution error:\n' + execution['error'] + '\nSelf-check your solution and return corrected fenced Python code.')))
        text = call()
        if execution['ok'] and text.strip() == 'KEEP':
            break
    checks = [dict(aspect='code execution', answer=execution['ok'], rationale=execution['error'] or 'Code executed.')]
    if execution['ok']:
        checks += runner.evaluate(execution, case, directory)
    return dict(outcome=first_failure(checks), checks=checks, attempts=attempts, calls=calls,
                svg=execution['svg'], png=execution['png'], code_path=str(target/'input/generated.py'),
                readability='not_evaluated', example_ids=[c['query_id'] for c in examples])


def run(config_path='configs/demo.json', limit=5):
    load_env()
    config = json.loads(Path(config_path).read_text())
    if limit < 1:
        raise ValueError('limit must be positive; choose an explicit budget before a large run.')
    if config['mode'] not in {'live','demo'}:
        raise ValueError('mode must be live or demo')
    strategies = config['strategies']
    if not strategies or not set(strategies) <= {'S1','S2','S3','S4'}:
        raise ValueError('Use strategies S1, S2, S3, S4.')
    names = [m['name'] for m in config['models']]
    if not names or len(set(names)) != len(names):
        raise ValueError('Model names must be unique and nonempty.')
    if not 0 <= config.get('max_repairs',2) <= 2:
        raise ValueError('The proposal allows at most two repair rounds.')
    clients = {}
    if config['mode'] == 'live':
        cases = load_viseval(config['dataset_dir'])
        dev, evaluation = split_databases(cases, config['seed'], config.get('development_fraction',.2))
        runner = DockerRunner(config.get('sandbox_image','chartbreak-worker:0.1'), config.get('timeout_seconds',60))
        runner.preflight()
        clients = {m['name']:ChatClient(m) for m in config['models']}
        dataset_hash = fingerprint(cases)
    else:
        cases = evaluation = demo.cases(); dev = []; dataset_hash = 'synthetic-demo-v1'
    # Same query set for every model and strategy; no favorable query selection.
    selected = evaluation[:limit]
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    root = (Path(config.get('output_dir','outputs')) / run_id).resolve(); root.mkdir(parents=True)
    safe_config = json.loads(json.dumps(config))
    # Inline secrets are unsupported; redact suspicious extra config fields anyway.
    for m in safe_config['models']:
        for k in list(m):
            if ('key' in k or 'token' in k or 'secret' in k) and not k.endswith('_env') and k != 'max_tokens':
                m[k] = '[redacted]'
    manifest = dict(run_id=run_id, mode=config['mode'], created_utc=datetime.now(timezone.utc).isoformat(),
        config=safe_config, version=__version__, prompt_version=PROMPT_VERSION, python=platform.python_version(),
        dataset_sha256=dataset_hash, development_databases=sorted({c['db_id'] for c in dev}),
        evaluation_databases=sorted({c['db_id'] for c in evaluation}), query_ids=[c['query_id'] for c in selected],
        resolved_models={name:c.model for name,c in clients.items()},
        viseval_commit='55162fac5765d8873180d962cce7e20e75b2c092', status='running')
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    try:
        with (root/'results.jsonl').open('w') as file:
            for model in config['models']:
                for case in selected:
                    for strategy in strategies:
                        key = hashlib.sha256(f"{model['name']}:{case['query_id']}:{strategy}".encode()).hexdigest()[:20]
                        folder = root/'cases'/key
                        start = time.monotonic()
                        if config['mode'] == 'demo':
                            record = demo.record(case,strategy,model,folder)
                        else:
                            folder.mkdir(parents=True)
                            examples = select_examples(dev,case) if strategy == 'S2' else []
                            try:
                                details = run_condition(case,strategy,clients[model['name']],runner,folder,examples,config.get('max_repairs',2))
                            except ProviderError as error:
                                details = dict(outcome='provider_error',checks=[],attempts=[],error=str(error),svg=None,png=None,code_path=None,readability='not_evaluated')
                            record = dict(**{k:v for k,v in case.items() if k not in {'ground_truth','reference_sql'}},
                                model=model['name'], model_id=clients[model['name']].model,strategy=strategy,mode='live',
                                elapsed_seconds=time.monotonic()-start,**details)
                        (folder/'result.json').write_text(json.dumps(record,indent=2))
                        file.write(json.dumps(record)+'\n'); file.flush()
                        print(f"{model['name']} {case['query_id']} {strategy}: {record['outcome']}")
        manifest['status'] = 'complete'
    except BaseException:
        manifest['status'] = 'interrupted'
        raise
    finally:
        (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    from .analysis import export
    export(root)
    return root
