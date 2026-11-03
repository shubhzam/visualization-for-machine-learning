import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch
from chartbreak.providers import ChatClient, ProviderError
from chartbreak.runner import run_condition
from test_backbone import FakeClient, FakeRunner


class ProviderTests(unittest.TestCase):
    def config(self):
        return dict(name='test',model='test-model',base_url='https://example.test/v1',api_key_env='TEST_CHARTBREAK_KEY')

    @patch.dict(os.environ, {'TEST_CHARTBREAK_KEY':'test-secret'})
    def test_request_and_usage(self):
        response=io.BytesIO(json.dumps({'choices':[{'message':{'content':'```python\npass\n```'}}],'usage':{'total_tokens':12}}).encode())
        with patch('urllib.request.urlopen',return_value=response) as request:
            text,usage=ChatClient(self.config()).complete([{'role':'user','content':'chart'}])
            sent=request.call_args.args[0]
            self.assertEqual(sent.full_url,'https://example.test/v1/chat/completions')
            self.assertEqual(json.loads(sent.data)['temperature'],0)
            self.assertEqual(usage['total_tokens'],12)
            self.assertIn('pass',text)

    @patch.dict(os.environ, {'TEST_CHARTBREAK_KEY':'test-secret'})
    def test_http_errors_do_not_expose_secrets(self):
        error=urllib.error.HTTPError('https://example.test',401,'test-secret',{},None)
        with patch('urllib.request.urlopen',side_effect=error):
            with self.assertRaises(ProviderError) as result:
                ChatClient(self.config()).complete([])
        self.assertNotIn('test-secret',str(result.exception))
        self.assertIn('401',str(result.exception))

    def test_plan_has_separate_call(self):
        case=dict(query='Plot counts.',tables=[])
        client=FakeClient(['Aggregate counts then draw bars.','```python\npass\n```'])
        runner=FakeRunner();runner.executions=1
        with tempfile.TemporaryDirectory() as folder:
            result=run_condition(case,'S3',client,runner,Path(folder))
        self.assertEqual(len(client.requests),2)
        self.assertIn('Aggregate counts',json.dumps(client.requests[1]))
        self.assertEqual(result['outcome'],'pass')

    def test_repair_stops_at_two_rounds(self):
        class AlwaysFails(FakeRunner):
            def execute(self,*args):
                self.executions+=1
                return dict(ok=False,error='broken code',svg=None,png=None)
        runner=AlwaysFails();client=FakeClient(['bad']*3)
        with tempfile.TemporaryDirectory() as folder:
            result=run_condition(dict(query='Plot counts.',tables=[]),'S4',client,runner,Path(folder))
        self.assertEqual(runner.executions,3)
        self.assertEqual(result['outcome'],'invalid')
        self.assertEqual(runner.evaluations,0)
