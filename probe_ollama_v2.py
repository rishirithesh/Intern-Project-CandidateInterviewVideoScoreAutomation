import requests
from urllib.parse import urljoin

base = 'http://localhost:11434'
paths = ['/', '/index.html', '/v1/models', '/v1/engines', '/v1/status', '/v1/health', '/v1', '/api/v1/models', '/api/v1/engines', '/api/v1/status', '/api/v1/health']
for path in paths:
    try:
        r = requests.get(urljoin(base, path), timeout=5)
        print(path, r.status_code)
        print(r.text[:400])
    except Exception as e:
        print(path, 'ERR', repr(e))
