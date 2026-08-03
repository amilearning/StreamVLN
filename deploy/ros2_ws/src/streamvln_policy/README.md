# streamvln_policy

ROS 2 policy requester for StreamVLN. Subscribes to a camera and an instruction, asks a
StreamVLN inference server what to do, and plays the returned discrete actions out as
velocity commands.

**Holds no model and needs no GPU** — `rclpy`, `numpy`, `requests`, `Pillow`. The server
does the thinking; this node does the plumbing and the timing.

## Interface

| direction | topic (default) | type | meaning |
|---|---|---|---|
| sub | `/boom/boom/color/image_raw` | `sensor_msgs/Image` | onboard RGB (`rgb8`/`bgr8`/`mono8`) |
| sub | `/prompt` | `std_msgs/String` | the instruction; a new one resets the session |
| sub | `/joy` | `sensor_msgs/Joy` | deadman — actions play only while held |
| pub | `/genvideo/cmd_vel` | `geometry_msgs/TwistStamped` | to the mux/deadman, then the base |
| pub | `/streamvln/status` | `std_msgs/String` | JSON diagnostics at 2 Hz |

States: `IDLE` (no prompt) → `RUNNING` → `DONE` (model emitted STOP; latched until a new
prompt). Zeros are published continuously in every state except while executing.

## How actions become motion

StreamVLN returns **4 discrete tokens** per request:
`1`=forward 0.25 m, `2`=left 15°, `3`=right 15°, `0`=stop.

`chunk_seconds` is the wall-clock budget for all 4, so each token gets `chunk_seconds/4`.
In the default `geometric` mode the speeds are *derived* so each token still covers the
increment the model was trained on:

| `chunk_seconds` | per token | forward | turn |
|---|---|---|---|
| **1.0** (default) | 0.25 s | **1.00 m/s** | **60 °/s** |
| 2.0 | 0.50 s | 0.50 m/s | 30 °/s |

`v_max`/`w_max` are a safety clamp, not a tuning knob. **If a clamp binds, the robot
under-travels relative to what StreamVLN believes it did** — and the model's memory
assumes those exact increments, so navigation degrades in a way that is invisible from
outside. The node warns loudly at startup and on the first clamped chunk. To go slower,
raise `chunk_seconds` (geometry stays exact) rather than lowering `v_max`.

Set `speed_mode: capped` to deliberately take the opposite trade: obey the caps and
accept the shortfall.

## Request pipelining

The next request is fired once the buffered plan drops below `request_lead_s`, so
inference overlaps execution and motion stays continuous.

**`request_lead_s` must be ≥ the server's response time.** With ~0.8 s inference, a lead
of 0.35 s measured a 77 % duty cycle (visible stuttering); a lead of 0.90 s measured
**96 %**. `gap_hold_s` holds the last velocity briefly when a response is late, rather
than showing a stutter — keep it small, since it is un-commanded motion.

Consequence of pipelining: the frame a chunk was computed from is up to one chunk old by
the time that chunk's last token executes. This is inherent, and the reason not to make
`chunk_seconds` large.

## Gotchas that will not announce themselves

- **Channel order.** The server does `np.asarray(pil_image)[..., ::-1]`, because the stock
  Go2 client hands it a `bgr8` array through PIL. Sending true RGB therefore reaches the
  model channel-swapped — no error, just quietly worse navigation. `bgr_swap: true`
  (default) reproduces the stock byte order. Only turn it off if you patch the server.
- **Session state lives on the server**, not here. The client owns only the `reset` flag.
  A new prompt must reset, or the new instruction is read against the old visual memory.
- **One client only.** The server's KV-cache is global; two clients corrupt it and every
  later request fails. After `reset_after_errors` consecutive failures the node forces a
  reset rather than retrying into a dead session.

## Configuration

All parameters live in [`config/streamvln_policy.yaml`](config/streamvln_policy.yaml),
documented inline. Nothing is hardcoded — a different robot changes the yaml, not the code.

```bash
ros2 launch streamvln_policy streamvln_policy.launch.py
ros2 launch streamvln_policy streamvln_policy.launch.py host:=192.168.1.50 chunk_seconds:=2.0
ros2 launch streamvln_policy streamvln_policy.launch.py \
    cmd_vel_topic:=/scratch/cmd_vel joy_gate_enable:=false   # BENCH ONLY: nothing reaches the base
```

Test on a scratch `cmd_vel_topic` before pointing it at the real mux.

## Standalone check

`action_map.py` is ROS-free and independently runnable:

```bash
python3 streamvln_policy/action_map.py
```
