#!/usr/bin/env python3
"""Local MCP: authenticate Radeon via Gmail and read pool capacity."""
import email.utils
import http.cookiejar
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from mcp.server.fastmcp import FastMCP

ROOT = Path('/data/lcx/codex/radeon-gmail')
SCOPE = ['https://www.googleapis.com/auth/gmail.readonly']
BASE = 'https://radeon-global.anruicloud.com'
EMAIL = 'code12ab@gmail.com'
server = FastMCP('radeon-capacity')

def gmail_service():
    token = ROOT / 'token.json'
    if not token.exists():
        raise RuntimeError('Gmail OAuth authorization is required')
    creds = Credentials.from_authorized_user_file(str(token), SCOPE)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token.write_text(creds.to_json(), encoding='utf-8')
        os.chmod(token, 0o600)
    if not creds.valid:
        raise RuntimeError('Gmail OAuth authorization has expired')
    return build('gmail', 'v1', credentials=creds, cache_discovery=False)

def read_code(service, started):
    query = 'from:noreply_radeoncloud@mail.developer.amd.com.cn subject:"Your Radeon Cloud login code" newer_than:1d'
    for _ in range(12):
        messages = service.users().messages().list(userId='me', q=query, maxResults=10).execute().get('messages', [])
        for entry in messages:
            msg = service.users().messages().get(userId='me', id=entry['id'], format='full').execute()
            if int(msg.get('internalDate', '0')) < int(started * 1000) - 5000:
                continue
            import base64
            def bodies(part):
                body = part.get('body', {}).get('data')
                if body:
                    yield base64.urlsafe_b64decode(body + '=' * (-len(body) % 4)).decode('utf-8', 'replace')
                for child in part.get('parts', []):
                    yield from bodies(child)
            for body in bodies(msg['payload']):
                match = re.search(r'verification code is:\s*(\d{6})|class=["\']code["\']\s*>\s*(\d{6})', body, re.I)
                if match:
                    return match.group(1) or match.group(2)
        time.sleep(5)
    raise RuntimeError('Radeon verification email did not arrive in time')

def opener_and_jar():
    jar = http.cookiejar.MozillaCookieJar(str(ROOT / 'session.cookies'))
    if Path(jar.filename).exists():
        jar.load(ignore_discard=True, ignore_expires=False)
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar)), jar

def get_status(opener):
    with opener.open(BASE + '/api/pools/status', timeout=20) as response:
        return json.load(response)

def login(opener, jar):
    service = gmail_service()
    started = time.time()
    def post(path, payload):
        req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'}, method='POST')
        with opener.open(req, timeout=20) as response:
            return json.load(response)
    post('/auth/email/request-code', {'email': EMAIL})
    code = read_code(service, started)
    post('/auth/email/verify', {'email': EMAIL, 'code': code})
    os.umask(0o077)
    jar.save(ignore_discard=True, ignore_expires=True)
    os.chmod(jar.filename, 0o600)

@server.tool()
def radeon_pool_status() -> dict:
    """Read Radeon GPU pool status; renew the web session through Gmail when needed."""
    try:
        opener, jar = opener_and_jar()
        try:
            data = get_status(opener)
        except urllib.error.HTTPError as exc:
            if exc.code != 401:
                raise
            login(opener, jar)
            data = get_status(opener)
        return data
    except Exception as exc:
        return {'error': type(exc).__name__, 'detail': str(exc)[:200]}

if __name__ == '__main__':
    server.run(transport='stdio')
