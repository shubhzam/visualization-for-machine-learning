"""Standard-library client for OpenAI-compatible Chat Completions endpoints."""
import json
import os
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

    def complete(self, messages):
        payload = dict(model=self.model, messages=messages,
                       max_tokens=self.config.get('max_tokens', 4096))
        if self.config.get('temperature', 0) is not None:
            payload['temperature'] = self.config.get('temperature', 0)
        req = urllib.request.Request(self.base_url.rstrip('/') + '/chat/completions',
            data=json.dumps(payload).encode(), headers={'Content-Type':'application/json',
            'Authorization': 'Bearer ' + self.key})
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.load(response)
        except urllib.error.HTTPError as e:
            # Never persist response bodies: providers may echo credentials or headers.
            raise ProviderError(f'Provider HTTP {e.code}; check endpoint/model configuration or quota.') from None
        except (urllib.error.URLError, TimeoutError):
            raise ProviderError('Provider connection failed or timed out.') from None
        try:
            text = result['choices'][0]['message']['content']
            if not isinstance(text, str) or not text.strip():
                raise ValueError()
        except (KeyError, IndexError, TypeError, ValueError):
            raise ProviderError('Provider returned no text completion.') from None
        return text, result.get('usage', {})
