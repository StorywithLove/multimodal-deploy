import http.cookiejar
import json
import sys
import urllib.request

base = 'https://radeon-global.anruicloud.com'
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
def post(path, data):
    req = urllib.request.Request(base + path, data=json.dumps(data).encode(), headers={'Content-Type': 'application/json'}, method='POST')
    with opener.open(req, timeout=20) as response:
        return response.status

post('/auth/email/request-code', {'email': 'code12ab@gmail.com'})
print('REQUESTED', flush=True)
code = sys.stdin.readline().strip()
post('/auth/email/verify', {'email': 'code12ab@gmail.com', 'code': code})
with opener.open(base + '/api/pools/status', timeout=20) as response:
    data = json.load(response)
    option = next(o for p in data['pools'] if p['pool_id'] == 'one-click' for o in p['options'] if o['resource_template'] == 'resource-16c55g1u')
    print(json.dumps({'http_status': response.status, 'computed_at': data['computed_at'], 'stale': data['stale'], 'state': option['state'], 'fit_count': option['fit_count']}), flush=True)
