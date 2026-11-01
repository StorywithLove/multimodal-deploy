#!/usr/bin/env python3
"""Request an email code and save an authenticated Radeon session."""
import http.cookiejar
import json
import os
import pathlib
import sys
import termios
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path('/data/lcx/codex/radeon-monitor')
BASE = 'https://radeon-global.anruicloud.com'
EMAIL = 'code12ab@gmail.com'
os.umask(0o077)
jar = http.cookiejar.MozillaCookieJar(str(ROOT / 'session.cookies'))
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))

def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'}, method='POST')
    with opener.open(req, timeout=20) as response:
        return response.status

try:
    start_ms = int(time.time() * 1000)
    post('/auth/email/request-code', {'email': EMAIL})
    print('CODE_REQUESTED_AFTER_MS', start_ms, flush=True)
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd) if os.isatty(fd) else None
    if previous:
        changed = termios.tcgetattr(fd)
        changed[3] &= ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSANOW, changed)
    try:
        code = sys.stdin.readline().strip()
    finally:
        if previous:
            termios.tcsetattr(fd, termios.TCSANOW, previous)
    if len(code) != 6 or not code.isascii() or not code.isdigit():
        raise ValueError('invalid code format')
    post('/auth/email/verify', {'email': EMAIL, 'code': code})
    with opener.open(BASE + '/api/pools/status', timeout=20) as response:
        data = json.load(response)
    if not isinstance(data.get('pools'), list) or not list(jar):
        raise ValueError('invalid authenticated response')
    temporary = ROOT / 'session.cookies.new'
    jar.save(str(temporary), ignore_discard=True, ignore_expires=True)
    os.chmod(temporary, 0o600)
    os.replace(temporary, ROOT / 'session.cookies')
    print('SESSION_SAVED', flush=True)
except urllib.error.HTTPError as exc:
    print('LOGIN_HTTP_ERROR', exc.code, flush=True)
    sys.exit(1)
except Exception as exc:
    print('LOGIN_ERROR', type(exc).__name__, flush=True)
    sys.exit(1)
