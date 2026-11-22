import subprocess
import time
import urllib.request
import sys

print("Starting Vite preview directly with node...")
proc = subprocess.Popen(
    ["node", "node_modules/vite/bin/vite.js", "preview", "--port", "5331", "--strictPort"],
    cwd="repos/GestureSign/web",
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

url = "http://localhost:5331/GestureSign/"
success = False

time.sleep(3)
out = ""
err = ""

try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=3) as response:
        status = response.status
        headers = dict(response.getheaders())
        body = response.read(500).decode('utf-8', errors='ignore')
        print(f"SUCCESS: Preview Server responded with HTTP {status}")
        print(f"Content-Type: {headers.get('Content-Type')}")
        print(f"Snippet:\n{body[:250]}...")
        success = True
except Exception as e:
    print(f"Connection attempt failed: {e}")
finally:
    proc.terminate()
    try:
        out, err = proc.communicate(timeout=3)
    except Exception:
        proc.kill()
        out, err = proc.communicate()

print("STDOUT from Preview:", out)
print("STDERR from Preview:", err)

if not success:
    sys.exit(1)

