import json

with open("repos/GestureSign/web/public/replay/clips.json", "r") as f:
    clips = json.load(f)

print(f"Type: {type(clips)}")
if isinstance(clips, dict):
    print(f"Keys: {list(clips.keys())}")
    for k in clips:
        v = clips[k]
        print(f"Key '{k}': type={type(v)}, len={len(v) if hasattr(v, '__len__') else v}")
        if isinstance(v, list) and len(v) > 0:
            print(f"  First item in '{k}': {type(v[0])}, keys={list(v[0].keys()) if isinstance(v[0], dict) else ''}")

