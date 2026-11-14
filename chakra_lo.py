#!/usr/bin/env python3
"""chakra_lo.py -- pichhle chakra ki copy laao (S3 redirect theek se).

Gadbad jo thi: GitHub artifact S3 par redirect karta hai, aur hum
Authorization header wahan bhi bhej rahe the -- S3 usse mana kar deta tha (401).

Sudhar: redirect khud handle karo, S3 par bina token ke jao.
"""
import io, json, os, sys, urllib.error, urllib.request, zipfile

TOKEN = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
REPO = os.environ.get("GITHUB_REPOSITORY", "pdfusion9-cyber/hindi-lattice")
DOMAIN = sys.argv[1] if len(sys.argv) > 1 else "science"
OUT = sys.argv[2] if len(sys.argv) > 2 else "old"

if not TOKEN:
    print("token nahi mila")
    sys.exit(1)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def api_json(url):
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


name = f"chakra-{DOMAIN}"
print(f"dhoondh rahe: {name}", flush=True)

try:
    d = api_json(f"https://api.github.com/repos/{REPO}/actions/artifacts")
except Exception as e:
    print("list fail:", type(e).__name__, str(e)[:120])
    sys.exit(1)

arts = [a for a in d.get("artifacts", []) if a["name"] == name and not a.get("expired")]
if not arts:
    print(f"koi {name} nahi mila -- shuru se chalega")
    sys.exit(2)

arts.sort(key=lambda a: a["id"], reverse=True)
pick = arts[0]
print(f"milta: id={pick['id']} {pick['size_in_bytes']/1e6:.0f} MB", flush=True)

zip_url = f"https://api.github.com/repos/{REPO}/actions/artifacts/{pick['id']}/zip"

# 1. redirect dhundo (token ke saath, par follow na karo)
opener = urllib.request.build_opener(NoRedirect)
req = urllib.request.Request(zip_url, headers={
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
})
try:
    opener.open(req, timeout=120)
    print("redirect nahi mila")
    sys.exit(1)
except urllib.error.HTTPError as e:
    if e.code in (301, 302, 303, 307, 308):
        s3 = e.headers.get("Location")
        print(f"redirect mila (HTTP {e.code})", flush=True)
    else:
        print(f"HTTP {e.code}: {str(e)[:120]}")
        sys.exit(1)

if not s3:
    print("Location header nahi mila")
    sys.exit(1)

# 2. S3 se BINA token ke utaaro
print("S3 se utaar rahe (bina token)...", flush=True)
try:
    with urllib.request.urlopen(s3, timeout=300) as r:
        raw = r.read()
except Exception as e:
    print("S3 fail:", type(e).__name__, str(e)[:150])
    sys.exit(1)

print(f"utaara: {len(raw)/1e6:.0f} MB", flush=True)

os.makedirs(OUT, exist_ok=True)
try:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        z.extractall(OUT)
except Exception as e:
    print("unzip fail:", type(e).__name__, str(e)[:120])
    sys.exit(1)

found = 0
for root, _, files in os.walk(OUT):
    for f in files:
        if f.endswith(".pt"):
            p = os.path.join(root, f)
            print("  ", p, os.path.getsize(p) // 1000000, "MB")
            found += 1
print(f"milne wali .pt files: {found}")
if found == 0:
    sys.exit(2)
