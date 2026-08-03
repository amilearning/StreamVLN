# StreamVLN — self-contained deployment

Everything needed to run StreamVLN on a real robot, in one place: a containerised
inference **server**, and a ROS 2 **policy requester** node.

```
  camera  ──image──▶ ┌────────────────────┐ ──HTTP──▶ ┌──────────────────────┐
  /prompt ──text───▶ │ streamvln_policy   │  1 frame  │ http_realworld_server │
  /joy   ──deadman─▶ │   (rclpy, no GPU)  │ ◀4 tokens─│   (GPU, Docker)       │
                     └────────┬───────────┘           └──────────────────────┘
                       TwistStamped
                              ▼
                      mux / deadman ──▶ base
```

The split is deliberate: the **node holds no model** (plain `rclpy` + `numpy` +
`requests`), so it runs on a robot with no GPU while the server runs wherever the GPU is —
same box, another machine on the LAN, or a workstation.

| | |
|---|---|
| `docker/` | Server image + launcher + smoke test. See [`../DEPLOYMENT_PLAN.md`](../DEPLOYMENT_PLAN.md) for the full recipe and the reasoning behind every pin. |
| `ros2_ws/src/streamvln_policy/` | The ROS 2 node. See its [README](ros2_ws/src/streamvln_policy/README.md). |

## Quickstart

**1. Build the server image and fetch the models** (~18 GB total, once).
Full instructions, including the two non-obvious requirements, are in
[`../DEPLOYMENT_PLAN.md`](../DEPLOYMENT_PLAN.md). Short version:

```bash
cd deploy/docker && docker build -t streamvln:thor .
# checkpoint (~15 GB) + the SigLIP vision tower (~3.3 GB, NOT bundled in the checkpoint)
```

**2. Start the server**

```bash
./deploy/run_server.sh          # foreground; -d to detach
```

**3. Run the node — on the bench first**

```bash
./deploy/run_policy.sh --bench  # scratch topic + gate off: NOTHING reaches the base
```

Builds the workspace on first run. Once the behaviour looks right, drop `--bench` to
publish to the real `cmd_vel` topic with the deadman gate armed:

```bash
./deploy/run_policy.sh                      # live
./deploy/run_policy.sh --host 192.168.1.50  # server on another machine
./deploy/run_policy.sh --tokens 4 --chunk 2.0
```

**4. Give it an instruction**

```bash
ros2 topic pub -1 /prompt std_msgs/msg/String "{data: 'Walk forward and stop at the door.'}"
```

Hold the deadman button and the robot drives.

## Watching it think

| where | what |
|---|---|
| `http://<server>:5801/debug` | **live annotated view** — the frame the model actually saw, with frame id, inference time and the chosen actions burned in. Auto-refreshes; open it in a browser. |
| `http://<server>:5801/status` | the same, as JSON, for scripts |
| `/streamvln/status` (ROS) | node side: state, prompt, gate, plan remaining, latency, error counts |
| `runs<MMDD-HHMM>/` | every annotated frame written to disk for after-the-fact review |

The `/debug` view is also the quickest way to catch a channel-order mistake: if colours
look inverted (blue objects appearing orange), `bgr_swap` is wrong.

## Read this before driving a robot

- **The deadman gate is the safety boundary.** `joy_gate_enable` defaults to `true` and
  actions only play while the configured button is held. **Verify `joy_button` against
  your own pad** — a wrong index either disables everything or, worse, gates on a button
  you did not intend. Test with `cmd_vel_topic` pointed at a scratch topic first.
- **The server holds ONE session.** Its KV-cache is global state, so exactly one client
  may talk to it at a time. Two clients corrupt the session; the server serializes
  requests and self-resets on failure, but do not rely on that as a feature.
- **Speed follows from `chunk_seconds` and `execute_tokens`.** StreamVLN's tokens are
  0.25 m and 15°, and `chunk_seconds` is the budget for the tokens actually executed.
  The default — 2 of 4 tokens over 1 s — gives 0.5 m/s and 30 °/s, replanning every
  0.5 m. To slow down, raise `chunk_seconds`; lowering `v_max` instead makes the robot
  under-travel relative to what the model believes it did.

## Measured on Jetson Thor (JetPack 7, sm_110, bf16 + sdpa)

| | |
|---|---|
| inference latency | ~0.77 s median, ~0.9 s typical under load, 1.2–1.6 s spikes |
| memory | ~33 GB of 122 GB unified |
| control loop | 1 request/s at `chunk_seconds: 1.0`, **96–98 % motion duty cycle** |
| default motion | 2 tokens/s → 0.5 m/s, 30 °/s, replan every 0.5 m |

`flash_attention_2` was measured and is **40 % slower** than `sdpa` here (1.29 s vs
0.77 s). Keep the `sdpa` default.
