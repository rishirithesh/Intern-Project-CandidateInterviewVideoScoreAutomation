import requests
from urllib.parse import urljoin

base = 'http://localhost:11434'
paths = [
    '/v1/completions',
    '/v1/chat/completions',
    '/v1/completions?model=qwen2.5:3b',
    '/v1/chat/completions?model=qwen2.5:3b',
]
for path in paths:
    try:
        url = urljoin(base, path)
        payload = {
            'model': 'qwen2.5:3b',
            'messages': [{'role': 'user', 'content': 'Hello'}],
            'temperature': 0.0,
            'max_tokens': 20,
        }
        r = requests.post(url, json=payload, timeout=10)
        print(path, r.status_code)
        print(r.text[:1000])
    except Exception as e:
        print(path, 'ERR', repr(e))
