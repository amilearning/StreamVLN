# StreamVLN — Install & Deploy Plan (frl 4090 server + Unitree Go2 client)

**What it is:** vision-language navigation — a streaming video VLM (LLaVA-Video / Qwen-1.5)
that takes onboard RGB(-D) + a natural-language instruction and emits nav actions
(forward / turn / stop). Repo cloned to `~/streamvln_ws/StreamVLN`.

**Key win for us:** the *real-world* path is **offboard + habitat-free**. The server
`streamvln/http_realworld_server.py` is a plain Flask app that only loads the model —
**no habitat-sim, no MP3D/HM3D scenes, no VLN-CE episodes** (those are benchmark-only).
So install is *much* smaller than the README's full setup.

Architecture (mirrors our existing policy-server split):
```
 Go2 (ROS2 foxy)                         frl 4090 (GPU server)
 ┌───────────────────┐  RGB(+depth)+pose ┌───────────────────────────┐
 │ realsense2_camera │ ───HTTP POST────▶ │ http_realworld_server.py  │
 │ go2_vln_client.py │  :5801/eval_vln   │   StreamVLNForCausalLM     │
 │  → /api/sport/req │ ◀──action seq──── │   (Qwen-1.5, bf16)         │
 └───────────────────┘                   └───────────────────────────┘
```

---

## Phase 1 — GPU server env (on the frl 4090)
Dedicated conda env (do NOT reuse `genvideo` — different torch/cuda).
```bash
conda create -n streamvln python=3.9 -y
conda activate streamvln
# torch 2.1.2 / cu121 (README says cu12.4; cu121 wheels run fine on a 12.x driver)
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
cd ~/streamvln_ws/StreamVLN
pip install -r requirements.txt          # flask, transformers, decord, bitsandbytes, clip, deepspeed, …
pip install -e .                          # if it has a setup/pyproject; else add repo to PYTHONPATH
```
⚠️ `requirements.txt` pins ~200 packages incl. `deepspeed`, `flash-attn`-adjacent, two
git deps (CLIP, depth_camera_filtering). Expect some version wrangling. habitat lines
in the README are for the *benchmark* — **skip them for real-world**.

## Phase 2 — Model checkpoint (~15 GB, HF)
Only the **real-world** checkpoint is needed:
```bash
huggingface-cli download mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world \
  --local-dir ~/streamvln_ws/checkpoints/streamvln_real_world
```
(The `_v1_3` checkpoint is the VLN-CE benchmark model — not needed for the robot.)

## Phase 3 — Smoke-test the server (frl, offline)
```bash
cd ~/streamvln_ws/StreamVLN/streamvln
python http_realworld_server.py --model_path ~/streamvln_ws/checkpoints/streamvln_real_world --device cuda:0
# serves Flask on 0.0.0.0:5801, endpoint POST /eval_vln
```
Then hit it with a dummy RGB + instruction (small python `requests.post`) to confirm it
loads + returns an action sequence **before** wiring the robot.
- **VRAM:** Qwen-1.5-7B in bf16 ≈ ~15–16 GB + video activations → should fit on the 24 GB
  4090 for inference. If tight, load 8-bit (`bitsandbytes` is already a dep).

## Phase 4 — Go2 client (on the unitree PC, ROS2)
`realworld/go2_vln_client.py` (rclpy) —
- **subscribes:** `/camera/camera/color/image_raw` (Realsense RGB), `/sportmodestate` (Go2 odom via `unitree_api/SportModeState`)
- **publishes:** `/api/sport/request` (`SPORT_API_ID_MOVE = 1008`) — the raw Unitree sport-mode velocity API
- edit the server IP → the frl box; POSTs to `http://<frl-ip>:5801/eval_vln`
```bash
# on go2: launch the realsense driver
ros2 launch realsense2_camera rs_align_depth_launch.py
# then the client
python3 go2_vln_client.py
```

### ⚠️ Adaptation needed for OUR Go2 stack
The stock client talks the **raw unitree sport API** (`/api/sport/request`). Our existing
Go2 bring-up (DOOM / `go2_ws` teleop + the recorder using `/cmd_vel_stamped`) may differ.
Decide the action path:
- (a) use the stock `/api/sport/request` MOVE call as-is (needs `unitree_ros2` / `unitree_sdk2` msgs on the go2), or
- (b) retarget the client to publish `cmd_vel` into our existing controller (cleaner integration with DOOM).
Also confirm the Realsense topic names match our camera launch (the recorder uses the boom D400).

## ✅ WORKING RECIPE (Phases 1–3 verified 2026-08-03 on the frl 4090)
Env `streamvln` (py3.9) built; real-world ckpt at `~/streamvln_ws/checkpoints/streamvln_real_world` (15 GB).
Fixes that were needed on top of the README:
- `av==14.4.0` has no py3.9 wheel → installed **av-13.1.0** instead (bundled ffmpeg).
- `flask` + `numpy-quaternion` were **missing from requirements.txt** → installed manually.
- `clip` git dep failed on new setuptools (`pkg_resources`) → `pip install 'setuptools<70'` + `--no-build-isolation`.
- server hardcoded `attn_implementation="flash_attention_2"` (needs a CUDA build) → patched to read
  `STREAMVLN_ATTN` env, default **`sdpa`** (works everywhere; set `STREAMVLN_ATTN=flash_attention_2` after building flash-attn for speed).

**Launch the server:**
```bash
conda activate streamvln
cd ~/streamvln_ws/StreamVLN/streamvln
PYTHONPATH=~/streamvln_ws/StreamVLN:~/streamvln_ws/StreamVLN/streamvln \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python http_realworld_server.py \
  --model_path ~/streamvln_ws/checkpoints/streamvln_real_world --device cuda:0
# serves Flask on 0.0.0.0:5801  ·  VRAM ~16.7 GB (bf16, sdpa)
```
**Smoke test** (dummy RGB → action list; actions: 1=↑ 2=← 3=→ 0=stop):
```python
import io, json, requests, numpy as np; from PIL import Image
img=Image.fromarray((np.random.rand(480,640,3)*255).astype('uint8'))
b=io.BytesIO(); img.save(b,'jpeg'); b.seek(0)
r=requests.post("http://127.0.0.1:5801/eval_vln",
  files={'image':('rgb',b,'image/jpg')}, data={'json':json.dumps({"reset":True})}, timeout=180)
print(r.json())   # → {'action': [3,3,1,2]}
```
NOTE (smoke stage): the server currently **hardcodes** the instruction (`"Walk forward and immediately
stop when you exit the room."`) and camera_pose. For real use, plumb the instruction through the
request (Phase 4 client work).

## Phase 5 — End-to-end
realsense (go2) → server up (frl) → client (go2) → type an instruction → watch it drive.
Server prints the action sequence; robot moves.

---

## Open decisions (need your call before the heavy install)
1. **Server host:** the frl 4090 workstation (this box), or a different GPU box?
2. **Action interface:** stock unitree sport API (a) vs retarget to our cmd_vel/DOOM path (b)?
3. **Depth:** client can send RGB only or RGB-D (depth sub is commented out in the stock client) — RGB-only is simpler to start.
4. Proceed with the ~15 GB checkpoint download + the ~200-pkg env now, or stage it?

## What we DON'T need (real-world path)
habitat-sim, habitat-lab, MP3D/HM3D scenes, VLN-CE R2R/RxR/EnvDrop/ScaleVLN episodes,
trajectory data, co-training datasets — all benchmark/training only.

---

# ✅ WORKING RECIPE #2 — Jetson Thor (Docker), verified 2026-08-03

Second, independent deployment: the model runs **onboard the Thor**, no 4090 in the loop.
This answers open decision #1 — either host works; Thor measured **~1.2 s/request**, close
enough to the 4090 that offboarding is optional rather than required.

**Ignore the README env entirely on Thor.** Python 3.9 / torch 2.1.2 / CUDA 12.4 cannot
exist on this board — Thor is Blackwell `sm_110`, CUDA 13.0, Ubuntu 24.04, Python 3.12,
L4T R38.4 (JetPack 7). There is no conda channel with a Thor-capable torch. Docker, not conda.

## Approach: layer on an existing image
`policy-server:thor` already carries the hard part — NVIDIA-built
`torch 2.9.0a0+nv25.09` (CUDA 13.0, `sm_110`), `flash_attn 2.7.4.post1`, timm, av, numpy.
Build `streamvln:thor` **FROM** it; the base image is not modified.

Tracing the real import graph of `http_realworld_server.py` → `streamvln_agent.py` →
`stream_video_vln.py` collapses `requirements.txt`'s 155 pins down to four actions:

| Need | Action |
|---|---|
| `transformers==4.45.1` | **downgrade** from the base image's 5.14.1 (+ `tokenizers<0.21`, `huggingface_hub<1.0`) |
| `flask` | **install** — missing from `requirements.txt` |
| `quaternion`, `depth_camera_filtering` | **stub** — imported at module scope, never called |
| torch / torchvision / flash_attn / timm / av / numpy / PIL / einops | already present ✓ |
| habitat, decord, deepspeed, torch-scatter, clip, bitsandbytes | **skip** — benchmark/training only, or guarded by `try/except` |

## ⚠️ The transformers pin is the whole ballgame
StreamVLN's vendored LLaVA targets the 4.45-era API. On transformers 5.x it breaks
**silently — no exception, just wrong behavior**:
- `model/stream_video_vln.py:418` overrides `prepare_inputs_for_generation(..., num_logits_to_keep=...)`.
  That kwarg was renamed `logits_to_keep` after 4.46, so on 5.x the override stops matching.
- `streamvln_agent.py:252` hand-carries `past_key_values` as a legacy tuple across steps;
  5.x is `Cache`-only. (On 4.45 the server logs the matching
  *"From v4.47 onwards ... will return a Cache instance"* deprecation notice — that legacy
  tuple is exactly what the agent depends on.)
- `llava/model/language_model/modeling_llama.py:32,44` imports `LlamaFlashAttention2` and
  `is_flash_attn_greater_or_equal_2_10`, both removed in 5.x. This one is survivable —
  `llava/model/__init__.py` wraps sub-model imports in `try/except` and only prints.

Install with `--no-deps` so pip can never touch the NVIDIA torch build, which is
unreplaceable (no upstream wheel targets `sm_110`).

## ⚠️ The SigLIP vision tower is NOT in the checkpoint
`config.json` sets `mm_vision_tower = google/siglip-so400m-patch14-384` and fetches it by
repo id at load time. It must be pre-staged into an HF cache or the server cannot start
offline — which is what you want on a robot anyway.

## Dockerfile (`~/streamvln_thor/Dockerfile`)
```dockerfile
FROM policy-server:thor

# --no-deps protects torch/numpy (unreplaceable Jetson builds). transformers' remaining
# runtime deps are already present from the base image's 5.x install.
# --ignore-installed blinker: base image's blinker is a Debian package with no RECORD
# file, so pip cannot uninstall it when Flask pulls a newer one.
RUN pip install --no-cache-dir --no-deps "transformers==4.45.1" \
 && pip install --no-cache-dir --ignore-installed blinker \
      "tokenizers==0.20.3" "huggingface_hub<1.0" flask

# Imported at module scope but never called; both fight aarch64/numpy-1.x builds.
RUN printf 'def filter_depth(*a, **k):\n    raise NotImplementedError("unused on the real-world path")\n' \
      > /usr/local/lib/python3.12/dist-packages/depth_camera_filtering.py \
 && printf 'def __getattr__(name):\n    raise AttributeError(name)\n' \
      > /usr/local/lib/python3.12/dist-packages/quaternion.py

# http_realworld_server.py:28 does ImageFont.truetype("DejaVuSansMono.ttf") on EVERY
# request. The font ships only inside matplotlib's private data dir, where PIL will not
# look — so publish it on the system font path rather than patching the server.
RUN mkdir -p /usr/share/fonts/truetype/dejavu \
 && cp /usr/local/lib/python3.12/dist-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSansMono.ttf \
       /usr/share/fonts/truetype/dejavu/ \
 && python3 -c "from PIL import ImageFont; ImageFont.truetype('DejaVuSansMono.ttf', 20); print('font OK')"

RUN python3 -c "import torch, transformers, flask, quaternion, depth_camera_filtering as d; \
assert transformers.__version__ == '4.45.1', transformers.__version__; \
print('OK torch', torch.__version__, '| transformers', transformers.__version__)"

ENV STREAMVLN_ATTN=sdpa \
    PYTHONPATH=/workspace/StreamVLN:/workspace/StreamVLN/streamvln \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HUB_OFFLINE=1
WORKDIR /workspace/StreamVLN/streamvln
```
Note: nothing CUDA-dependent can be asserted at build time — no GPU is visible during
`docker build` (`get_device_capability()` and even `get_arch_list()` both fail). Check
those at runtime instead.

## Fetch the two model artifacts
```bash
mkdir -p ~/streamvln_ckpt/hf_cache
# 1) policy checkpoint (~15 GB)
docker run --rm --network host --user $(id -u):$(id -g) \
  -e HF_HOME=/ckpt/.hfcache -e HOME=/ckpt -v ~/streamvln_ckpt:/ckpt streamvln:thor \
  hf download mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world \
    --local-dir /ckpt/streamvln_real_world
# 2) SigLIP vision tower (~3.3 GB) — required, see above
docker run --rm --network host --user $(id -u):$(id -g) \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=0 -e HOME=/hf -v ~/streamvln_ckpt/hf_cache:/hf \
  streamvln:thor hf download google/siglip-so400m-patch14-384
```

## Launch (`~/streamvln_thor/run_server.sh`)
Code and checkpoint are **mounted, not baked in**, so the server stays editable without a rebuild.
```bash
docker run --rm -it --runtime nvidia --network host --ipc=host --name streamvln_server \
  -e STREAMVLN_ATTN=sdpa -e HF_HOME=/hf \
  -v ~/StreamVLN:/workspace/StreamVLN \
  -v ~/streamvln_ckpt/streamvln_real_world:/ckpt:ro \
  -v ~/streamvln_ckpt/hf_cache:/hf:ro \
  streamvln:thor \
  python3 http_realworld_server.py --model_path /ckpt --device cuda:0
```

## Measured on Thor
```
POST http://127.0.0.1:5801/eval_vln
  req 0:  1.24s  action=[3, 3, 1, 3]  >>^>
  req 1:  1.17s  action=[3, 3, 3, 1]  >>>^
  req 2:  1.15s  action=[3, 3, 3, 3]  >>>>
```
~1.2 s/request (each request = 4 `evaluator.step` calls, 1 model `generate`).
Memory is a non-issue: **33 GB of 122 GB unified** — `nvidia-smi` reports `N/A` for
memory on Thor, so read `free -g` instead. `flash_attn` 2.7.4.post1 is present and
sm_110-capable, so `STREAMVLN_ATTN=flash_attention_2` is worth trying for extra speed
once `sdpa` is confirmed working.

## Still TODO before driving a robot
1. **The instruction is hardcoded** (`http_realworld_server.py:74`) and the client sends
   only `{"reset": bool}` — no instruction field exists. The smoke-test actions above are
   therefore meaningless; they prove the pipeline runs, not that it understands anything.
   Plumbing the instruction through the POST body is the first real change needed.
2. `go2_vln_client.py` **will not import** — it calls `ReadWriteLock()` at line 31 but only
   does `from pid_controller import *`, which does not export it. Add `from utils import ReadWriteLock`.
3. The server writes `runs<MMDD-HHMM>/rgb_N_annotated.png` into the CWD, i.e. straight into
   the mounted git working tree. Redirect it or gitignore it.
