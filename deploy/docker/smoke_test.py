#!/usr/bin/env python3
"""Smoke-test the StreamVLN server: dummy RGB in, action sequence out.

Actions: 1=forward 25cm, 2=left 15deg, 3=right 15deg, 0=stop.
The server hardcodes the instruction, so the actions are not meaningful here --
this only proves the model loads, runs on the GPU, and returns a parseable sequence.
"""
import io
import json
import sys
import time

import numpy as np
import requests
from PIL import Image

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5801/eval_vln"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3


def post(reset):
    img = Image.fromarray((np.random.rand(480, 640, 3) * 255).astype("uint8"))
    buf = io.BytesIO()
    img.save(buf, "jpeg")
    buf.seek(0)
    t0 = time.time()
    r = requests.post(
        URL,
        files={"image": ("rgb", buf, "image/jpg")},
        data={"json": json.dumps({"reset": reset})},
        timeout=600,
    )
    r.raise_for_status()
    return r.json(), time.time() - t0


print(f"POST {URL}  ({N} requests; first one includes warm-up)")
for i in range(N):
    payload, dt = post(reset=(i == 0))
    actions = payload.get("action")
    glyphs = "".join({0: "STOP", 1: "^", 2: "<", 3: ">"}.get(a, "?") for a in actions)
    print(f"  req {i}: {dt:6.2f}s  action={actions}  {glyphs}")

print("\nsmoke test PASSED")
