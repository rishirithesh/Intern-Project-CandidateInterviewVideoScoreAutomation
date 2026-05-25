import requests
from urllib.parse import urljoin

base = 'http://localhost:11434'
urls = ['/api/models', '/api/health', '/api/ping', '/models', '/health', '/status']
for u in urls:
    try:
        r = requests.get(urljoin(base, u), timeout=5)
        print(u, r.status_code, r.text[:200])
    except Exception as e:
        print(u, 'ERR', repr(e))
