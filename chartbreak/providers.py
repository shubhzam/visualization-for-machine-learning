"""Standard-library client for OpenAI-compatible Chat Completions endpoints."""
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


def load_env(path='.env'):
    if not Path(path).exists():
        return
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name, sep, value = line.partition('=')
        if sep:
            os.environ.setdefault(name.strip(), value.strip().strip('\"\''))


class ProviderError(RuntimeError):
    pass


class ChatClient:
    def __init__(self, config):
        self.config = config
        self.model = config.get('model') or os.getenv(config.get('model_env', ''), '')
        self.base_url = config.get('base_url') or os.getenv(config.get('base_url_env', ''), '')
        self.key = os.getenv(config.get('api_key_env', ''), '')
        if not self.model or self.model.startswith('your-') or not self.base_url or not self.key:
            raise ValueError(f"Set the model, base URL, and API key for {config['name']} in .env")
        if not self.base_url.startswith(('https://', 'http://localhost:', 'http://127.0.0.1:')):
            raise ValueError('Use HTTPS for remote endpoints or localhost for a local server.')

        self.last_meta = None

    def complete(self, messages):
        payload = dict(model=self.model, messages=messages,
                       max_tokens=self.config.get('max_tokens', 4096))
        if self.config.get('temperature', 0) is not None:
            payload['temperature'] = self.config.get('temperature', 0)
        req = urllib.request.Request(self.base_url.rstrip('/') + '/chat/completions',
            data=json.dumps(payload).encode(), headers={'Content-Type':'application/json',
            'Authorization': 'Bearer ' + self.key})
        # retry rate limits, 5xx and timeouts; anything else is a provider_error, not a model failure
        retries, base = self.config.get('max_retries', 4), self.config.get('retry_base_seconds', 2)
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=120) as response:
                    result = json.load(response)
                break
            except urllib.error.HTTPError as e:
                # Never persist response bodies: providers may echo credentials or headers.
                if (e.code == 429 or e.code >= 500) and attempt < retries:
                    time.sleep(base * 2 ** attempt); continue
                raise ProviderError(f'Provider HTTP {e.code}; check endpoint/model configuration or quota.') from None
            except (urllib.error.URLError, TimeoutError):
                if attempt < retries:
                    time.sleep(base * 2 ** attempt); continue
                raise ProviderError('Provider connection failed or timed out.') from None
        try:
            choice = result['choices'][0]
            text = choice['message']['content']
            if not isinstance(text, str) or not text.strip():
                raise ValueError()
        except (KeyError, IndexError, TypeError, ValueError):
            raise ProviderError('Provider returned no text completion.') from None
        self.last_meta = dict(model_id_returned=result.get('model'), finish_reason=choice.get('finish_reason'), retries=attempt)
        return text, result.get('usage', {})


class MockClient:
    """Offline stand-in for an LLM so the pipeline can run without an API key. Not a result."""
    def __init__(self, config):
        self.config, self.model, self.last_meta = config, 'mock-deterministic-v1', None

    def complete(self, messages):
        last = messages[-1]['content']
        if 'short plan' in last:
            text = 'Plan: read the first table, count rows per value of its first column, draw the requested chart type.'
        elif last.startswith('Execution succeeded'):
            text = 'KEEP'
        else:
            query = next(json.loads(m['content']) for m in reversed(messages)
                         if m['role'] == 'user' and m['content'].startswith('{'))
            table, words = query['tables'][0], set(re.findall(r'\w+', query['query'].lower()))
            draw = ('plt.pie(counts.values, labels=counts.index)' if 'pie' in words else
                    'plt.plot(counts.index, counts.values)' if 'line' in words else
                    'plt.scatter(counts.index, counts.values)' if 'scatter' in words else
                    'plt.bar(counts.index, counts.values)')
            text = ('```python\nimport pandas as pd\nimport matplotlib.pyplot as plt\n'
                    f"df = pd.read_csv({table['path']!r})\n"
                    f"counts = df[{table['columns'][0]!r}].astype(str).value_counts().sort_index()\n"
                    f'plt.figure(figsize=(8, 5))\n{draw}\nplt.tight_layout()\n```')
        self.last_meta = dict(model_id_returned=self.model, finish_reason='stop', retries=0)
        return text, dict(prompt_tokens=sum(len(m['content']) for m in messages) // 4, completion_tokens=len(text) // 4)
