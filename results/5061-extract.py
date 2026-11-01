import json
import re
from pathlib import Path

html = Path(r"C:\Users\27395\AppData\Local\Temp\p5061.html")
if not html.exists():
    html = Path("/tmp/p5061.html")
text = html.read_text(encoding="utf-8", errors="replace")

idx = text.find('"notebook_json"')
print("notebook_json idx:", idx)
# find the JSON object that has "cells"
start = text.find('"cells"', idx)
print("cells idx:", start)

# walk backwards from idx to find the start of a JSON value; instead, brute force:
# find all occurrences of '"cells":' and try json decoder on surrounding brace
out = []
for m in re.finditer(r'"cells"\s*:', text):
    # find enclosing object start: search backwards for '{' that balances
    i = m.start()
    depth = 0
    j = i
    while j > 0:
        c = text[j]
        if c == '}':
            depth += 1
        elif c == '{':
            if depth == 0:
                break
            depth -= 1
        j -= 1
    dec = json.JSONDecoder()
    try:
        obj, end = dec.raw_decode(text[j:])
    except Exception as e:
        print("decode fail at", j, e)
        continue
    if isinstance(obj, dict) and obj.get("cells"):
        out.append((j, obj))
        print("candidate notebook at", j, "cells:", len(obj["cells"]))

if out:
    nb = out[0][1]
    Path(r"D:\Project\multimodal-deploy\results\5061-notebook.json").write_text(
        json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        print(f"\n===== cell {k} [{cell.get('cell_type')}] =====")
        print(src)
