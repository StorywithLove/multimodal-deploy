#!/usr/bin/env python3
"""Authorize Gmail on OCI via a browser and SSH localhost port forwarding."""
import os
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow

root = Path('/data/lcx/codex/radeon-gmail')
os.umask(0o077)
flow = InstalledAppFlow.from_client_secrets_file(str(root / 'credentials.json'), ['https://www.googleapis.com/auth/gmail.readonly'])
creds = flow.run_local_server(host='localhost', port=8765, open_browser=False, access_type='offline', prompt='consent')
(root / 'token.json').write_text(creds.to_json(), encoding='utf-8')
os.chmod(root / 'token.json', 0o600)
print('Gmail authorization complete')
