"""Tests for the harness fixes."""
import io
import json
import tempfile
import unittest
import urllib.error
from collections import Counter
from pathlib import Path
from unittest.mock import patch
from chartbreak.data import load_viseval, split_databases, apply_split, select_examples, stratified_sample, eligible_examples
from chartbreak.execution import DockerRunner
from chartbreak.prompts import is_keep, messages
from chartbreak.providers import ChatClient, MockClient, ProviderError
from chartbreak.runner import run_condition, run
from test_backbone import FakeClient


class AlwaysWorks:
    def __init__(self): self.executions = 0; self.codes = []
    def execute(self, code, case, directory):
        self.executions += 1; self.codes.append(code)
        return dict(ok=True, error='', svg='chart.svg', png=None)
    def evaluate(self, execution, case, directory):
        return [dict(aspect=a, answer=True, rationale='') for a in
                ['surface-form check', 'deconstruction', 'chart type check', 'data check']]


class UpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_viseval('data/visEval_dataset')
        cls.dev, cls.evaluation = split_databases(cls.cases)

    def test_three_channels_means_a_classify_channel(self):
        for c in self.cases:
            self.assertEqual(c['features']['three_channels'], bool(c['ground_truth']['vis_obj'].get('classify')))
        self.assertEqual(sum(c['features']['three_channels'] for c in self.cases), 153)

    def test_stratified_sample(self):
        _, evaluation = apply_split(self.cases, 'splits/split_v1.json')
        sample = stratified_sample(evaluation, 10, 42)
        self.assertEqual(sample, stratified_sample(evaluation, 10, 42))
        self.assertNotEqual(sample, stratified_sample(evaluation, 10, 43))
        self.assertEqual(len({c['chart_id'] for c in sample}), len(sample))
        charts = Counter(c['features']['chart_type'] for c in {c['chart_id']: c for c in evaluation}.values())
        expected = {t: min(10, n) for t, n in charts.items()}
        self.assertEqual(dict(Counter(c['features']['chart_type'] for c in sample)), expected)
        eval_dbs = {c['db_id'] for c in evaluation}
        self.assertTrue(all(c['db_id'] in eval_dbs for c in sample))

    def test_keep_verdicts(self):
        for reply in ['KEEP', 'KEEP.', 'keep', '```KEEP```', '```\nKEEP\n```', 'KEEP, the code is correct.']:
            self.assertTrue(is_keep(reply), reply)
        for reply in ['```python\nimport matplotlib\n```', 'Fix:\n```python\nx = 1\n```', '```\nx = 1\n```']:
            self.assertFalse(is_keep(reply), reply)

    def test_s4_keep_variant_keeps_working_code(self):
        runner, client = AlwaysWorks(), FakeClient(['```python\nx = 1\n```', 'KEEP.'])
        with tempfile.TemporaryDirectory() as folder:
            result = run_condition(self.evaluation[0], 'S4', client, runner, Path(folder))
        self.assertEqual(runner.executions, 1)
        self.assertEqual(result['outcome'], 'pass')
        self.assertEqual(result['attempts'][0]['self_check'], 'keep')

    def test_s4_revision_after_success_runs_new_code(self):
        runner, client = AlwaysWorks(), FakeClient(['```python\nx = 1\n```', '```python\nx = 2\n```', 'KEEP'])
        with tempfile.TemporaryDirectory() as folder:
            run_condition(self.evaluation[0], 'S4', client, runner, Path(folder))
        self.assertEqual(runner.codes, ['x = 1', 'x = 2'])

    def test_s2_uses_only_allowed_examples(self):
        case = self.evaluation[0]
        allowed = {c['query_id'] for c in eligible_examples(self.dev)[::2]}
        examples = select_examples(self.dev, case, allowed=allowed)
        self.assertEqual(len(examples), 3)
        self.assertTrue(all(c['query_id'] in allowed for c in examples))
        self.assertEqual(select_examples(self.dev, case), select_examples(self.dev, case, allowed=None))
        with self.assertRaises(ValueError):
            select_examples(self.dev, case, allowed=set())

    def test_generated_code_is_written_as_utf8(self):
        code = "import matplotlib.pyplot as plt\nplt.bar(['Beyoncé', 'a → b'], [1, 2])\n"
        with tempfile.TemporaryDirectory() as folder, patch.object(DockerRunner, '_run', return_value=False):
            DockerRunner().execute(code, self.evaluation[0], Path(folder))
            self.assertEqual((Path(folder) / 'input' / 'generated.py').read_bytes().decode('utf-8'), code)

    def test_mock_client_is_offline_and_sees_only_public_input(self):
        case = self.evaluation[0]
        with patch('urllib.request.urlopen', side_effect=AssertionError('network used')):
            client = MockClient({'name': 'mock'})
            code, _ = client.complete(messages(case, 'S1'))
            self.assertIn('pd.read_csv', code)
            self.assertEqual(client.complete(messages(case, 'S1') + [{'role': 'user', 'content': 'Execution succeeded. ...'}])[0], 'KEEP')
        self.assertEqual(client.last_meta['model_id_returned'], 'mock-deterministic-v1')
        truth = json.dumps(case['ground_truth']['vis_obj'])
        self.assertNotIn(truth, json.dumps(messages(case, 'S1')))

    def test_retry_then_success_and_metadata(self):
        config = dict(name='t', model='m', base_url='https://example.test/v1', api_key_env='TEST_KEY', retry_base_seconds=0)
        ok = io.BytesIO(json.dumps({'model': 'm-2026', 'choices': [{'message': {'content': 'x'}, 'finish_reason': 'length'}], 'usage': {}}).encode())
        busy = urllib.error.HTTPError('https://example.test', 503, 'busy', {}, None)
        with patch.dict('os.environ', {'TEST_KEY': 'k'}), patch('urllib.request.urlopen', side_effect=[busy, ok]):
            client = ChatClient(config); client.complete([])
        self.assertEqual(client.last_meta, dict(model_id_returned='m-2026', finish_reason='length', retries=1))
        bad = urllib.error.HTTPError('https://example.test', 400, 'bad', {}, None)
        with patch.dict('os.environ', {'TEST_KEY': 'k'}), patch('urllib.request.urlopen', side_effect=[bad]) as call:
            with self.assertRaises(ProviderError):
                ChatClient(config).complete([])
            self.assertEqual(call.call_count, 1)

    def test_truncated_responses_are_flagged_and_excluded(self):
        from chartbreak.analysis import summary
        class Truncating(FakeClient):
            def complete(self, m):
                self.last_meta = dict(finish_reason='length'); return super().complete(m)
        with tempfile.TemporaryDirectory() as folder:
            result = run_condition(self.evaluation[0], 'S1', Truncating(['```python\nx = 1\n```']), AlwaysWorks(), Path(folder))
        self.assertTrue(result['truncated'])
        self.assertTrue(all(k in result['calls'][0] for k in ['model_sent', 'utc_time', 'latency_s', 'finish_reason']))
        rows = [dict(model='m', strategy='S1', chart_id='c', outcome='pass', truncated=True),
                dict(model='m', strategy='S1', chart_id='d', outcome='pass')]
        self.assertEqual(summary(rows)[0]['n_evaluated'], 1)

    def test_manifest_records_code_version(self):
        with tempfile.TemporaryDirectory() as folder:
            config = json.loads(Path('configs/demo.json').read_text()); config['output_dir'] = folder
            path = Path(folder) / 'config.json'; path.write_text(json.dumps(config))
            manifest = json.loads((run(path, 1) / 'manifest.json').read_text())
        self.assertIn('commit', manifest['git']); self.assertIn('dirty', manifest['git'])
        self.assertIsNone(manifest['sandbox_image_id'])


if __name__ == '__main__':
    unittest.main()
