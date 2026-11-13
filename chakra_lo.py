#!/usr/bin/env python3
"""chakra_lo.py -- pichhle chakra ki copy laao (artifact API se).

Artifact ka naam: chakra-<domain>
Sabse naya, jo expire na hua ho, wahi utaarta hai.
"""
import json, os, sys, urllib.request, zipfile, io

TOKEN = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
REPO = os.environ.get("GITHUB_REPOSITORY", "pdfusion9-cyber/hindi-lattice")
DOMAIN = sys.argv[1] if len(sys.argv) > 1 else "science"
OUT = sys.argv[2] if len(sys.argv) > 2 else "old"

if not TOKEN:
    print("token nahi mila")
    sys.exit(1)


def api(url, raw=False):
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read() if raw else json.load(r)


name = f"chakra-{DOMAIN}"
print(f"dhoondh rahe: {name}")

try:
    d = api(f"https://api.github.com/repos/{REPO}/actions/artifacts")
except Exception as e:
    print("list fail:", type(e).__name__, str(e)[:120])
    sys.exit(1)

arts = [a for a in d.get("artifacts", []) if a["name"] == name and not a.get("expired")]
if not arts:
    print(f"koi {name} nahi mila -- shuru se chalega")
    sys.exit(2)

arts.sort(key=lambda a: a["id"], reverse=True)
pick = arts[0]
print(f"milta: id={pick['id']} {pick['size_in_bytes']/1e6:.0f} MB")

try:
    raw = api(f"https://api.github.com/repos/{REPO}/actions/artifacts/{pick['id']}/zip", raw=True)
except Exception as e:
    print("download fail:", type(e).__name__, str(e)[:120])
    sys.exit(1)

os.makedirs(OUT, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    z.extractall(OUT)
print(f"khol diya: {OUT}")
for root, _, files in os.walk(OUT):
    for f in files:
        if f.endswith(".pt"):
            print("  ", os.path.join(root, f), os.path.getsize(os.path.join(root, f)) // 1000000, "MB")
