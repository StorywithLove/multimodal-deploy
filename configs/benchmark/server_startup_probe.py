#!/usr/bin/env python3
"""Measure process launch to local health readiness; stop only our child process."""
import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--health', default='http://127.0.0.1:8000/health')
    parser.add_argument('--output', required=True)
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--cache-state', default='uncontrolled',
                        choices=['uncontrolled', 'warm', 'cold'])
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.timeout <= 0:
        parser.error('provide a server command after -- and a positive timeout')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def healthy():
        try:
            with opener.open(args.health, timeout=1) as response:
                return response.status == 200
        except Exception:
            return False
    if healthy():
        parser.error('health endpoint is already ready; stop the existing server first')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.with_suffix('.server.log')
    result = {'command': command, 'health': args.health,
              'cache_state': args.cache_state, 'ready': False,
              'metric': 'process_launch_to_health_ready', 'log': str(log_path)}
    with log_path.open('w', encoding='utf-8') as log:
        started = time.perf_counter()
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while time.perf_counter() - started < args.timeout:
                if process.poll() is not None:
                    result['error'] = f'server exited: {process.returncode}'
                    break
                if healthy():
                    result['ready'] = True
                    result['startup_s'] = time.perf_counter() - started
                    break
                time.sleep(0.2)
            else:
                result['error'] = 'startup timeout'
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0 if result['ready'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
