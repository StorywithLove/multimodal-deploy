"""OCI RCC capacity sampling. Credentials/state live outside the repository."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
from datetime import datetime, timezone, timedelta

ROOT = Path(os.environ.get('RCC_PROBE_HOME', '/home/opc/rcc-probe'))
STATE = ROOT / 'state.json'
LOG = ROOT / 'events.jsonl'
RCC = ROOT / 'venv/bin/rcc'
HOLD_START = datetime(2026, 10, 28, 6, tzinfo=timezone(timedelta(hours=8))).timestamp()
HOLD_END = datetime(2026, 10, 28, 8, tzinfo=timezone(timedelta(hours=8))).timestamp()


def now():
    return datetime.now(timezone.utc).isoformat()


def event(kind, **data):
    with LOG.open('a') as f:
        f.write(json.dumps(dict(at=now(), kind=kind, **data)) + '\n')
    print(kind, json.dumps(data), flush=True)


def save(state):
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(state))
    tmp.replace(STATE)


def cli(*args):
    result = subprocess.run([str(RCC), *args, '--json'], capture_output=True,
                            text=True, timeout=60)
    if result.returncode:
        # Do not put auth tokens or full diagnostic responses in logs.
        raise RuntimeError('RCC command failed: ' + ' '.join(args[:2]))
    payload = json.loads(result.stdout)
    return payload.get('data', payload)


def status():
    return cli('instance', 'status')


def balance():
    return float(cli('credits', 'status')['credits'])


def create_args(retain, request_persistent):
    args = ['instance', 'create', '--env', 'production-1', '--pod-type',
            'one-click', '--resource-template', 'resource-16c55g1u']
    if not retain:
        args.append('--no-ssh')
    elif request_persistent:
        args.append('--use-pvc')
    return args + ['--no-wait', '--idempotency-key', 'oci-probe-' + uuid.uuid4().hex]


def record_balance(state, credits):
    previous = state.get('last_credits')
    if previous is not None and credits < previous:
        state['observed_decrease'] = state.get('observed_decrease', 0.0) + previous - credits
    state['last_credits'] = credits
    save(state)


def cleanup(state):
    if state.get('held_instance'):
        return True
    owned = state.get('owned')
    if not owned:
        return True
    current = status()
    if current.get('status') == 'not_found':
        state.pop('owned', None)
        save(state)
        event('released', instance_id=owned)
        return True
    if current.get('instance_id') != owned:
        state['halt'] = 'ownership_mismatch'
        save(state)
        event('halt', reason='ownership_mismatch')
        return False
    if state.get('retain_on_ready') and HOLD_START <= time.time() < HOLD_END and (
            current.get('ready') or current.get('status') == 'ready'):
        state['held_instance'] = owned
        save(state)
        event('held', instance_id=owned, source='cleanup_watchdog')
        return True
    cli('instance', 'delete', '--yes')
    for _ in range(6):
        time.sleep(3)
        current = status()
        if current.get('status') == 'not_found':
            state.pop('owned', None)
            save(state)
            event('released', instance_id=owned)
            return True
        if current.get('instance_id') not in (None, owned):
            break
    event('cleanup_pending', instance_id=owned)
    return False


def run(mode):
    auth = json.loads((ROOT / 'auth/config.json').read_text())
    os.environ['RCC_TOKEN'] = auth['token']
    os.environ['RCC_URL'] = auth['base_url']
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if state.get('held_instance'):
        return
    if mode == 'cleanup':
        if state.get('owned') and time.time() >= state.get('release_at', 0):
            cleanup(state)
        return
    if state.get('owned'):
        if not cleanup(state):
            return
    if state.get('halt') or state.get('create_intent') or (ROOT / 'PAUSE').exists():
        event('skipped', reason=state.get('halt', 'unresolved_create' if state.get('create_intent') else 'paused'))
        return
    credits = balance()
    if not state:
        state = dict(started=now(), deadline=time.time() + 3 * 86400,
                     initial_credits=credits, observed_decrease=0.0)
        save(state)
    record_balance(state, credits)
    if time.time() >= state['deadline'] or state.get('observed_decrease', 0.0) >= .25 or credits <= .1:
        event('skipped', reason='duration_or_budget_limit', credits=credits)
        return
    current = status()
    if current.get('status') != 'not_found':
        event('skipped', reason='existing_instance', status=current.get('status'))
        return
    # Reserve a cleanup deadline before submitting. A timeout has uncertain ownership:
    # halt rather than retrying creation or deleting an unverified instance.
    state['create_intent'] = now()
    state['retain_on_ready'] = HOLD_START <= time.time() < HOLD_END
    save(state)
    started = time.monotonic()
    try:
        persistent = state['retain_on_ready'] and (ROOT / 'REQUEST_PERSISTENT').exists()
        args = create_args(state['retain_on_ready'], persistent)
        result = cli(*args)
        instance_id = result.get('instance_id')
        if not instance_id:
            raise RuntimeError('Create response has no instance ID')
        state['owned'] = instance_id
        state['release_at'] = time.time() + 120
        state.pop('create_intent', None)
        save(state)
        event('submitted', instance_id=instance_id, credits=credits,
              persistent_requested=persistent)
        success = False
        last = {}
        while time.time() < state['release_at']:
            last = status()
            if last.get('instance_id') not in (None, instance_id):
                raise RuntimeError('Instance changed during probe')
            if last.get('ready') or last.get('status') == 'ready':
                success = True
                if state.get('retain_on_ready') and HOLD_START <= time.time() < HOLD_END:
                    state['held_instance'] = instance_id
                    save(state)
                    event('held', instance_id=instance_id, source='probe')
                break
            if last.get('status') in ('not_found', 'failed', 'error'):
                break
            time.sleep(10)
        persistence = last.get('persistent_available')
        if not isinstance(persistence, bool):
            persistence = None
        event('outcome', success=success, seconds=round(time.monotonic()-started, 2),
              status=last.get('status'), reason=last.get('reason'), instance_id=instance_id,
              persistent_available=persistence)
    except Exception as exc:
        if not state.get('owned'):
            state['halt'] = 'create_uncertain_or_rejected'
            save(state)
        event('error', error=type(exc).__name__)
    finally:
        if state.get('owned') and not state.get('held_instance'):
            cleanup(state)
        after = balance()
        record_balance(state, after)
        event('balance', before=credits, after=after)


def report():
    groups = {}
    if not LOG.exists():
        print('No samples yet.')
        return
    for line in LOG.read_text().splitlines():
        item = json.loads(line)
        if item['kind'] != 'outcome':
            continue
        hour = datetime.fromisoformat(item['at']).astimezone(timezone(timedelta(hours=8))).hour
        bucket = hour // 2 * 2
        row = groups.setdefault(bucket, [0, 0, []])
        row[0] += 1
        if item['success']:
            row[1] += 1
            row[2].append(item['seconds'])
    for hour, (total, ok, waits) in sorted(groups.items()):
        import statistics
        print(f'Beijing {hour:02d}:00-{hour+2:02d}:00 samples={total} ready={ok} '
              f'rate={ok/total:.1%} median_ready_seconds={statistics.median(waits) if waits else "N/A"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['probe', 'cleanup', 'report'])
    args = parser.parse_args()
    if args.mode == 'report':
        report()
    else:
        ROOT.mkdir(parents=True, exist_ok=True)
        with (ROOT / 'probe.lock').open('w') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit(0)
            try:
                run(args.mode)
            except Exception as exc:
                event('error', error=type(exc).__name__)
                raise SystemExit(1)
