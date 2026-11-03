import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from chartbreak.data import load_viseval, split_databases, select_examples, public_input
from chartbreak.prompts import messages
from chartbreak.execution import first_failure
from chartbreak.runner import run_condition, run
from chartbreak.analysis import summary, flips, load_results


class FakeClient:
    def __init__(self, replies):
        self.replies=iter(replies);self.requests=[]
    def complete(self, messages):
        self.requests.append(json.loads(json.dumps(messages)))
        return next(self.replies), {'total_tokens': 10}


class FakeRunner:
    def __init__(self): self.executions=0;self.evaluations=0
    def execute(self, code, case, directory):
        self.executions+=1
        ok=self.executions>1
        return dict(ok=ok,error='' if ok else 'NameError: x is not defined',svg='chart.svg' if ok else None,png=None)
    def evaluate(self, execution, case, directory):
        self.evaluations+=1
        return [dict(aspect=a,answer=True,rationale='SECRET_GROUND_TRUTH') for a in
                ['surface-form check','deconstruction','chart type check','data check']]


class BackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases=load_viseval('data/visEval_dataset')
        cls.dev,cls.evaluation=split_databases(cls.cases)

    def test_benchmark_shape_and_split(self):
        self.assertEqual(len(self.cases),2524)
        self.assertEqual(len({c['chart_id'] for c in self.cases}),1150)
        self.assertEqual(len({c['db_id'] for c in self.cases}),146)
        self.assertFalse({c['db_id'] for c in self.dev} & {c['db_id'] for c in self.evaluation})
        self.assertEqual(split_databases(self.cases),(self.dev,self.evaluation))

    def test_examples_and_no_answer_leak(self):
        case=self.evaluation[0];examples=select_examples(self.dev,case)
        self.assertEqual(len(examples),3)
        self.assertTrue(all(c['db_id'] != case['db_id'] for c in examples))
        content=json.dumps(public_input(case))
        self.assertNotIn('ground_truth',content);self.assertNotIn('reference_sql',content)
        prompts=messages(case,'S2',examples)
        self.assertEqual(len(prompts),8)

    def test_self_repair_is_bounded_and_blind(self):
        client=FakeClient(['```python\nx\n```','```python\nx=1\n```','KEEP'])
        runner=FakeRunner()
        with tempfile.TemporaryDirectory() as folder:
            result=run_condition(self.evaluation[0],'S4',client,runner,Path(folder))
        self.assertEqual(result['outcome'],'pass')
        self.assertEqual(runner.executions,2);self.assertEqual(runner.evaluations,1)
        self.assertNotIn('SECRET_GROUND_TRUTH',json.dumps(client.requests))
        self.assertIn('NameError',json.dumps(client.requests))

    def test_stages_and_missing_checks(self):
        self.assertEqual(first_failure([]),'not_evaluated')
        self.assertEqual(first_failure([dict(aspect='deconstruction',answer=False)]),'unparseable')
        self.assertEqual(first_failure([dict(aspect='data check',answer=False)]),'illegal_data')

    def test_cluster_summary_excludes_provider_errors(self):
        rows=[dict(model='m',strategy='S1',chart_id='one',outcome=o) for o in ['pass','pass','provider_error']]
        report=summary(rows)[0]
        self.assertEqual(report['n_evaluated'],2);self.assertEqual(report['n_excluded'],1)
        self.assertEqual(report['pass_rate'],1);self.assertIsNone(report['ci_low'])

    def test_paired_flips(self):
        rows=[dict(model='m',query_id='q',strategy=s,outcome=o) for s,o in [('S1','invalid'),('S4','pass')]]
        self.assertEqual(flips(rows,'m'),[dict(source='invalid',target='pass',count=1,query_ids=['q'])])

    def test_demo_writes_all_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            config=json.loads(Path('configs/demo.json').read_text());config['output_dir']=folder
            path=Path(folder)/'config.json';path.write_text(json.dumps(config))
            root=run(path,2)
            self.assertEqual(len(load_results(root)),8)
            self.assertEqual(json.loads((root/'manifest.json').read_text())['status'],'complete')
            for f in ['summary.json','results.csv','dashboard.json']:
                self.assertTrue((root/f).is_file())

if __name__=='__main__': unittest.main()
