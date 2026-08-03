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
