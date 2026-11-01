#!/usr/bin/env python3
"""Collect Radeon capacity snapshots without launching an instance."""
import datetime as dt
import fcntl
import http.cookiejar
import json
import os
import pathlib
import subprocess
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path('/data/lcx/codex/radeon-monitor')
END = dt.datetime(2026, 12, 1, 6, 45, tzinfo=dt.timezone.utc)
BASE = 'https://radeon-global.anruicloud.com'
ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
os.umask(0o077)

PROMPT = '''Renew the Radeon Cloud session with the connected Gmail plugin. Execute exactly
python3 -u /data/lcx/codex/radeon-monitor/radeon_session_login.py in a persistent PTY,
yield promptly, and wait for CODE_REQUESTED_AFTER_MS. If it prints LOGIN_HTTP_ERROR or
LOGIN_ERROR, stop immediately. Search Gmail for the newest email from
noreply_radeoncloud@mail.developer.amd.com.cn with subject Your Radeon Cloud login code,
received after the printed timestamp. Read the email, extract the six-digit code from its
HTML body, and send only the code plus newline to the waiting process with write_stdin.
The process disables terminal echo. Wait for SESSION_SAVED and report only success or
failure. Never print the code, cookie, message body or credentials. Do not use web search,
inspect unrelated files, create instances or retry the code request. If the message does
not arrive promptly or any action is blocked, stop and report failure.'''

def renew():
    cooldown = ROOT / 'renew_after'
    try:
        retry_after = float(cooldown.read_text())
    except (FileNotFoundError, ValueError):
        retry_after = 0
    if time.time() < retry_after:
        return 'login_backoff'
    cooldown.write_text(str(time.time() + 3600), encoding='utf-8')
    env = os.environ.copy()
    env.update({'HOME': '/data/lcx/codex', 'CODEX_HOME': '/data/lcx/codex/.codex',
                'PATH': '/data/lcx/codex/.local/bin:/usr/local/bin:/usr/bin:/bin'})
    command = ['/data/lcx/codex/.local/bin/codex', 'exec', '--ephemeral',
               '--skip-git-repo-check', '--sandbox', 'workspace-write',
               '-c', 'sandbox_workspace_write.network_access=true', '--color', 'never',
               '--model', 'gpt-5.6-luna', PROMPT]
    try:
        result = subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=240)
    except subprocess.TimeoutExpired:
        return 'login_timeout'
    if result.returncode:
        return 'login_codex_error'
    return None

def snapshot():
    jar = http.cookiejar.MozillaCookieJar(str(ROOT / 'session.cookies'))
    if (ROOT / 'session.cookies').exists():
        jar.load(ignore_discard=True, ignore_expires=False)
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    with opener.open(BASE + '/api/pools/status', timeout=20) as response:
        data = json.load(response)
    option = next(o for p in data['pools'] if p['pool_id'] == 'one-click' for o in p['options'] if o['resource_template'] == 'resource-16c55g1u')
    if data.get('stale') is not False or option.get('state') not in ('available', 'tight', 'full', 'unknown'):
        return {'status': 'unknown', 'reason': 'stale_or_invalid'}
    return {'status': option['state'], 'fit_count': option.get('fit_count'), 'computed_at': data.get('computed_at'), 'age_seconds': data.get('age_seconds')}

def main():
    now = dt.datetime.now(dt.timezone.utc)
    if now >= END:
        return
    with (ROOT / 'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        try:
            try:
                result = snapshot()
            except urllib.error.HTTPError as e:
                if e.code != 401:
                    raise
                login_error = renew()
                if login_error:
                    result = {'status': 'error', 'reason': login_error}
                else:
                    try:
                        result = snapshot()
                    except urllib.error.HTTPError as retry_error:
                        result = {'status': 'error', 'reason': 'login_or_query_http_' + str(retry_error.code)}
                    except Exception as retry_error:
                        result = {'status': 'error', 'reason': 'login_or_query_' + type(retry_error).__name__}
        except urllib.error.HTTPError as e:
            result = {'status': 'unknown', 'reason': 'http_' + str(e.code)}
        except Exception as e:
            result = {'status': 'unknown', 'reason': type(e).__name__}
        row = {'observed_at': now.isoformat(), **result}
        with (ROOT / 'observations.jsonl').open('a', encoding='utf-8') as output:
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps(row, ensure_ascii=False))

if __name__ == '__main__':
    main()
