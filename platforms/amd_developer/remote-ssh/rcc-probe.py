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
BEIJING = timezone(timedelta(hours=8))


def in_launch_window():
    local = datetime.now(BEIJING)
    start = (4, 0) if local.date().isoformat() >= '2026-10-30' else (5, 30)
    allowed_minutes = (0, 20, 40) if local.date().isoformat() >= '2026-10-30' else (10, 30, 50)
    if local.minute not in allowed_minutes:
        return False
    return (local.hour, local.minute) >= start and local.hour < 9


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


def classify_diagnostics(*values):
    """Keep recognized failure classes, never raw service or workload logs."""
    text = ' '.join(json.dumps(v, ensure_ascii=False) for v in values).lower()
    patterns = {
        'capacity': ('no matching node', 'insufficient gpu', 'no available gpu',
                     'insufficient capacity', 'out of capacity', 'no free gpu'),
        'image_pull': ('imagepullbackoff', 'errimagepull', 'image pull failed'),
        'persistence': ('persistentvolumeclaim', 'persistent volume',
                        'pvc pending', 'pvc failed', 'volume mount failed'),
        'quota': ('quota exceeded', 'insufficient credits', 'insufficient quota'),
        'authorization': ('unauthorized', 'forbidden', 'authentication failed'),
        'rate_limit': ('rate limit', 'too many requests'),
    }
    return [name for name, terms in patterns.items() if any(term in text for term in terms)]


def diagnostic_snapshot(current):
    classes = set(classify_diagnostics(current))
    try:
        logs = cli('instance', 'logs')
        classes.update(classify_diagnostics(logs))
        return classes, 'available'
    except Exception:
        return classes, 'unavailable'


def create_args(retain, request_persistent):
    args = ['instance', 'create', '--env', 'production-1', '--pod-type',
            'one-click', '--resource-template', 'resource-16c55g1u']
    if not retain:
        args.append('--no-ssh')
    args.append('--use-pvc' if request_persistent else '--no-pvc')
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
    if state.get('retain_on_ready') and (
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
    if mode == 'cleanup':
        if state.get('held_instance'):
            return
        if state.get('owned') and time.time() >= state.get('release_at', 0):
            cleanup(state)
        return
    if state.get('held_instance'):
        current = status()
        if current.get('status') != 'not_found':
            return
        held = state.pop('held_instance')
        state.pop('owned', None)
        state['last_credits'] = balance()
        save(state)
        event('held_released_manually', instance_id=held)
    if state.get('owned'):
        if not cleanup(state):
            return
    if state.get('halt') or state.get('create_intent') or (ROOT / 'PAUSE').exists():
        event('skipped', reason=state.get('halt', 'unresolved_create' if state.get('create_intent') else 'paused'))
        return
    if mode != 'verify' and not in_launch_window():
        return
    credits = balance()
    today = datetime.now(BEIJING).date().isoformat()
    if state.get('budget_date') != today:
        state.update(budget_date=today, last_credits=credits, observed_decrease=0.0)
    record_balance(state, credits)
    if state.get('observed_decrease', 0.0) >= .25 or credits <= .1:
        event('skipped', reason='duration_or_budget_limit', credits=credits)
        return
    current = status()
    if current.get('status') != 'not_found':
        event('skipped', reason='existing_instance', status=current.get('status'))
        return
    retain = datetime.now(BEIJING).hour >= 5
    # A failed persistent attempt can be followed by a non-persistent attempt
    # only after the first instance is confirmed absent and its cleanup completes.
    for persistent in (True, False):
        if not attempt(state, credits, retain, persistent):
            return
        credits = balance()
        record_balance(state, credits)
        if state.get('observed_decrease', 0.0) >= .25 or credits <= .1:
            event('skipped', reason='duration_or_budget_limit', credits=credits)
            return


def attempt(state, credits, retain, persistent):
    # Reserve a cleanup deadline before submitting. A timeout has uncertain ownership:
    # halt rather than retrying creation or deleting an unverified instance.
    state['create_intent'] = now()
    state['retain_on_ready'] = retain
    save(state)
    started = time.monotonic()
    try:
        args = create_args(retain, persistent)
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
        classes = set()
        log_samples = {'available': 0, 'unavailable': 0}
        while time.time() < state['release_at']:
            last = status()
            if last.get('instance_id') not in (None, instance_id):
                raise RuntimeError('Instance changed during probe')
            # A disappearing instance may make logs unavailable: sample while
            # it exists, including immediately before the next status check.
            if last.get('status') != 'not_found':
                found, log_state = diagnostic_snapshot(last)
                classes.update(found)
                log_samples[log_state] += 1
            else:
                classes.update(classify_diagnostics(last))
            if last.get('ready') or last.get('status') == 'ready':
                success = True
                if state.get('retain_on_ready'):
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
              status=last.get('status'), failure_classes=sorted(classes),
              log_samples=log_samples, instance_id=instance_id,
              persistent_available=persistence, persistent_requested=persistent)
    except Exception as exc:
        if not state.get('owned'):
            state['halt'] = 'create_uncertain_or_rejected'
            save(state)
        event('error', error=type(exc).__name__)
        return False
    finally:
        if state.get('owned') and not state.get('held_instance'):
            cleanup(state)
        after = balance()
        record_balance(state, after)
        event('balance', before=credits, after=after)
    if success or state.get('owned') or state.get('halt') or state.get('create_intent'):
        return False
    # Only explicit terminal failure permits the fallback; timeouts remain ambiguous.
    return persistent and last.get('status') in ('not_found', 'failed', 'error')


def report():
    groups = {}
    comparison = {}
    if not LOG.exists():
        print('No samples yet.')
        return
    for line in LOG.read_text().splitlines():
        item = json.loads(line)
        if item['kind'] != 'outcome':
            continue
        requested = item.get('persistent_requested')
        if isinstance(requested, bool):
            row = comparison.setdefault(requested, {'total': 0, 'ready': 0, 'classes': {},
                                                     'logs_available': 0})
            row['total'] += 1
            row['ready'] += bool(item.get('success'))
            row['logs_available'] += bool(item.get('log_samples', {}).get('available'))
            for name in item.get('failure_classes', []):
                row['classes'][name] = row['classes'].get(name, 0) + 1
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
    for requested, row in sorted(comparison.items(), reverse=True):
        print(f'persistent_requested={requested} samples={row["total"]} ready={row["ready"]} '
              f'logs_available={row["logs_available"]} classes={json.dumps(row["classes"])}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['probe', 'verify', 'cleanup', 'report'])
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
