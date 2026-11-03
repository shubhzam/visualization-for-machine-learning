import json
import shutil
import subprocess
import uuid
from pathlib import Path


class InfrastructureError(RuntimeError):
    pass


class DockerRunner:
    def __init__(self, image='chartbreak-worker:0.2', timeout=60):
        self.image, self.timeout = image, timeout
        self.image_id = None

    def preflight(self):
        if not shutil.which('docker'):
            raise InfrastructureError('Install and start Docker Desktop, then build the worker image (README).')
        result = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}', self.image], capture_output=True, text=True)
        if result.returncode:
            raise InfrastructureError('Start Docker and run: docker build -t chartbreak-worker:0.2 sandbox')
        self.image_id = result.stdout.strip()

    def _run(self, script, mounts):
        name = 'chartbreak-' + uuid.uuid4().hex
        cmd = ['docker', 'run', '--rm', '--name', name, '--network', 'none', '--read-only',
               '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--pids-limit', '64',
               '--memory', '1g', '--cpus', '1', '--user', '65534:65534',
               '--tmpfs', '/tmp:rw,noexec,nosuid,size=128m']
        for source, target, access in mounts:
            cmd += ['--mount', f'type=bind,src={Path(source).resolve()},dst={target}' + (',readonly' if access == 'ro' else '')]
        cmd += [self.image, 'python', '/opt/chartbreak/' + script]
        try:
            # File-backed logs avoid unbounded host memory if generated code prints a lot.
            import tempfile
            with tempfile.TemporaryFile() as log:
                result = subprocess.run(cmd, stdout=log, stderr=log, timeout=self.timeout)
                log.seek(0); output = log.read(8000).decode(errors='replace')
            if result.returncode:
                raise InfrastructureError(f'Worker exited {result.returncode}: {output}')
        except subprocess.TimeoutExpired:
            subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=15)
            return False
        return True

    def execute(self, code, case, directory):
        directory = Path(directory)
        source, data, output = (directory / n for n in ['input', 'data', 'output'])
        for p in [source, data, output]:
            p.mkdir(parents=True, exist_ok=True)
        output.chmod(0o777)
        # windows defaults to cp1252, which broke labels like 'é' inside the container
        (source / 'generated.py').write_text(code, encoding='utf-8', newline='')
        for table in case['tables']:
            shutil.copyfile(table, data / Path(table).name)
        completed = self._run('worker.py', [(source,'/input','ro'), (data,'/data','ro'), (output,'/output','rw')])
        if not completed:
            return dict(ok=False, error=f'Execution exceeded {self.timeout} seconds.', svg=None, png=None)
        record = output / 'execution.json'
        if not record.is_file():
            raise InfrastructureError('Worker did not return an execution record.')
        result = json.loads(record.read_text())
        result.update(svg=str(output/'chart.svg') if result['ok'] else None,
                      png=str(output/'chart.png') if result['ok'] else None)
        return result

    def evaluate(self, execution, case, directory):
        directory = Path(directory)
        truth, output = directory / 'truth', directory / 'evaluation'
        truth.mkdir(); output.mkdir(); output.chmod(0o777)
        (truth / 'ground_truth.json').write_text(json.dumps(case['ground_truth']))
        completed = self._run('evaluate.py', [(Path(execution['svg']).parent,'/input','ro'),
                              (truth,'/truth','ro'), (output,'/output','rw')])
        if not completed:
            raise InfrastructureError('Evaluator timed out; this is not a model failure.')
        return json.loads((output / 'checks.json').read_text())


def first_failure(checks):
    mapping = {'code execution':'invalid', 'surface-form check':'invalid',
               'deconstruction':'unparseable', 'chart type check':'illegal_chart',
               'data check':'illegal_data', 'order check':'illegal_order'}
    for check in checks:
        if not check['answer']:
            return mapping.get(check['aspect'], 'evaluation_error')
    # Missing legality checks never imply success.
    aspects = {c['aspect'] for c in checks}
    return 'pass' if {'code execution','surface-form check','deconstruction','chart type check','data check'} <= aspects else 'not_evaluated'
