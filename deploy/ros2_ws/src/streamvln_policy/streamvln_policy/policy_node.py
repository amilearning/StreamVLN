"""StreamVLN policy requester.

Bridges a camera + an instruction to base velocity commands by asking a StreamVLN
inference server what to do:

    <image_topic>   sensor_msgs/Image      onboard RGB
    <prompt_topic>  std_msgs/String        natural-language instruction
    <joy_topic>     sensor_msgs/Joy        deadman gate (actions only play while held)
        |
        |  HTTP POST http://<host>:<port>/eval_vln   (one frame -> 4 discrete tokens)
        v
    <cmd_vel_topic> geometry_msgs/TwistStamped       -> mux/deadman -> base

The node holds NO model: it is plain rclpy + numpy + requests, so it runs on a robot with
no GPU while the server runs wherever the GPU is.

Design notes worth knowing before changing anything:

* Channel order. The server does `np.asarray(pil_image)[..., ::-1]` because the stock Go2
  client hands it a bgr8 array via PIL. Sending true RGB therefore reaches the model
  channel-swapped -- with no error, just quietly worse navigation. `bgr_swap` (default
  true) reproduces the stock client's byte order. Leave it on unless the server is patched.

* Session state lives on the SERVER. StreamVLN keeps a KV-cache across requests; the
  client owns only the `reset` flag. A new prompt must reset, or the new instruction is
  interpreted against the old visual memory.

* Requests are pipelined. A chunk plays for `chunk_seconds` while the next request is
  already in flight (fired once the buffered plan drops below `request_lead_s`), so motion
  is continuous rather than stop-and-go. The frame a chunk was computed from is therefore
  up to one chunk old by the time the last token of that chunk executes -- inherent to
  pipelining, and the reason `chunk_seconds` should not be made large.
"""
import io
import json
import math
import threading
import time
from typing import List, Optional

import numpy as np
import rclpy
import requests
from geometry_msgs.msg import TwistStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, Joy
from std_msgs.msg import String

from streamvln_policy.action_map import (TOKENS_PER_CHUNK, ChunkPlan, Segment,
                                         format_actions, plan_chunk)

try:
    from PIL import Image as PILImage
except ImportError as exc:                       # pragma: no cover
    raise ImportError("streamvln_policy needs Pillow: pip install pillow") from exc

# --- node states -------------------------------------------------------------------
IDLE = "IDLE"          # no prompt yet, or stopped; publishes zeros
RUNNING = "RUNNING"    # requesting + executing
DONE = "DONE"          # model emitted STOP; latched until a new prompt arrives


def _sensor_qos(depth: int = 1) -> QoSProfile:
    """BEST_EFFORT/VOLATILE, matching typical camera publishers."""
    return QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                      history=QoSHistoryPolicy.KEEP_LAST,
                      durability=QoSDurabilityPolicy.VOLATILE,
                      depth=depth)


def image_msg_to_rgb(msg: Image) -> np.ndarray:
    """sensor_msgs/Image -> (H, W, 3) uint8 RGB. No cv_bridge dependency."""
    enc = msg.encoding.lower()
    h, w = int(msg.height), int(msg.width)
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if enc in ("rgb8", "bgr8"):
        step = int(msg.step) if msg.step else w * 3
        arr = buf.reshape(h, step)[:, : w * 3].reshape(h, w, 3)
        if enc == "bgr8":
            arr = arr[..., ::-1]
        return np.ascontiguousarray(arr)
    if enc == "mono8":
        step = int(msg.step) if msg.step else w
        gray = buf.reshape(h, step)[:, :w]
        return np.ascontiguousarray(np.stack([gray] * 3, axis=-1))
    raise ValueError(f"unsupported image encoding: {msg.encoding!r}")


class StreamVlnPolicyNode(Node):

    def __init__(self) -> None:
        super().__init__("streamvln_policy_node")
        p = self._declare_params()

        # --- server ---
        self._url = f"http://{p['host']}:{int(p['port'])}/eval_vln"
        self._timeout = float(p["request_timeout_s"])

        # --- motion shaping ---
        self._chunk_seconds = float(p["chunk_seconds"])
        self._step_m = float(p["step_m"])
        self._turn_deg = float(p["turn_deg"])
        self._v_max = float(p["v_max"])
        self._w_max = float(p["w_max"])
        self._speed_mode = str(p["speed_mode"])
        self._execute_tokens = int(p["execute_tokens"])
        self._honor_stop_beyond = bool(p["honor_stop_beyond_horizon"])
        self._gap_hold_s = float(p["gap_hold_s"])
        self._request_lead_s = float(p["request_lead_s"])
        self._control_hz = float(p["control_hz"])
        self._cmd_frame = str(p["cmd_frame_id"])
        self._bgr_swap = bool(p["bgr_swap"])
        self._jpeg_quality = int(p["jpeg_quality"])
        self._reset_on_prompt_change = bool(p["reset_on_prompt_change"])
        self._stop_behavior = str(p["stop_behavior"]).strip().lower()
        if self._stop_behavior not in ("latch", "continue"):
            raise ValueError(f"stop_behavior must be 'latch' or 'continue', got {self._stop_behavior!r}")
        self._stops = 0
        self._verbose = bool(p["verbose"])

        # --- joy gate ---
        self._joy_gate_enable = bool(p["joy_gate_enable"])
        self._joy_button = int(p["joy_button"])
        self._joy_topic = str(p["joy_topic"])
        self._joy_go = False
        self._joy_seen = False
        self._no_joy_ticks = 0

        # --- shared state (guarded by _lock) ---
        self._lock = threading.Lock()
        self._state = IDLE
        self._segments: List[Segment] = []
        self._seg_idx = 0
        self._seg_t = 0.0
        self._last_seg: Optional[Segment] = None
        self._starved_t = 0.0
        self._stop_after_plan = False
        self._pending_reset = True
        self._prompt: str = str(p["default_prompt"])
        self._have_prompt = bool(self._prompt) and bool(p["autostart"])
        # autostart must also leave IDLE, or the inference loop never fires and the node
        # sits silently doing nothing with a prompt it never uses.
        if self._have_prompt:
            self._state = RUNNING
        self._frame: Optional[np.ndarray] = None
        self._frame_stamp = 0.0
        self._last_actions: List[int] = []
        self._last_latency = 0.0
        self._requests = 0
        self._resets = 0
        self._errors = 0
        self._consec_errors = 0
        self._reset_after_errors = int(p["reset_after_errors"])
        self._last_error = ""

        self._warn_clamped_once = False
        self._shutdown = threading.Event()

        # Refuse to start on a geometry the robot cannot deliver: in 'geometric' mode a
        # bound cap means silent under-travel, which corrupts the model's spatial prior.
        n_exec = (TOKENS_PER_CHUNK if self._execute_tokens <= 0
                  else min(self._execute_tokens, TOKENS_PER_CHUNK))
        v_need = self._step_m / (self._chunk_seconds / n_exec)
        w_need = math.radians(self._turn_deg) / (self._chunk_seconds / n_exec)
        self.get_logger().info(
            f"chunk={self._chunk_seconds:.2f}s over {n_exec}/{TOKENS_PER_CHUNK} tokens "
            f"({self._chunk_seconds / n_exec:.3f}s/token) requires "
            f"v={v_need:.2f} m/s, w={math.degrees(w_need):.1f} deg/s "
            f"[caps v_max={self._v_max:.2f}, w_max={math.degrees(self._w_max):.1f} deg/s, "
            f"mode={self._speed_mode}]")
        if self._speed_mode == "geometric" and (v_need > self._v_max + 1e-9
                                                or w_need > self._w_max + 1e-9):
            self.get_logger().warn(
                "SPEED CAP BINDS: the robot will under-travel each token relative to the "
                "0.25 m / 15 deg increments StreamVLN was trained on. Raise chunk_seconds "
                "(slower but geometry-true) or switch speed_mode to 'capped' to accept it.")

        cbg = ReentrantCallbackGroup()
        img_qos = _sensor_qos() if bool(p["camera_qos_sensor_data"]) else 1
        self.create_subscription(Image, str(p["image_topic"]),
                                 self._on_image, img_qos, callback_group=cbg)
        self.create_subscription(String, str(p["prompt_topic"]),
                                 self._on_prompt, 10, callback_group=cbg)
        self.create_subscription(Joy, str(p["joy_topic"]),
                                 self._on_joy, 10, callback_group=cbg)

        self._cmd_pub = self.create_publisher(TwistStamped, str(p["cmd_vel_topic"]), 10)
        self._status_pub = self.create_publisher(String, str(p["status_topic"]), 10)

        self.create_timer(1.0 / self._control_hz, self._on_control_tick, callback_group=cbg)
        self.create_timer(1.0 / max(0.1, float(p["status_hz"])),
                          self._on_status_tick, callback_group=cbg)

        self._infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self._infer_thread.start()

        self.get_logger().info(
            f"streamvln_policy_node up | server={self._url} | "
            f"image={p['image_topic']} prompt={p['prompt_topic']} cmd={p['cmd_vel_topic']} | "
            f"joy_gate={'ON button %d on %s' % (self._joy_button, p['joy_topic']) if self._joy_gate_enable else 'OFF'}")

    # ------------------------------------------------------------------ parameters
    def _declare_params(self) -> dict:
        defaults = {
            # server
            "host": "127.0.0.1",
            "port": 5801,
            "request_timeout_s": 30.0,
            # topics
            "image_topic": "/boom/boom/color/image_raw",
            "prompt_topic": "/prompt",
            "cmd_vel_topic": "/genvideo/cmd_vel",
            "status_topic": "/streamvln/status",
            "joy_topic": "/joy_teleop/joy",
            "cmd_frame_id": "base_link",
            "camera_qos_sensor_data": True,
            "status_hz": 2.0,
            # motion
            "chunk_seconds": 1.0,
            "step_m": 0.25,
            "turn_deg": 15.0,
            "v_max": 1.0,
            "w_max": 1.2,
            "speed_mode": "geometric",
            # Receding horizon: play only the first N of the 4 returned tokens, then
            # re-plan from a fresh frame. 0 = play all 4.
            "execute_tokens": 2,
            "honor_stop_beyond_horizon": True,
            "gap_hold_s": 0.25,
            "request_lead_s": 0.90,
            "control_hz": 20.0,
            # gate
            "joy_gate_enable": True,
            "joy_button": 3,
            # behaviour
            "bgr_swap": True,
            "jpeg_quality": 90,
            "reset_on_prompt_change": True,
            "reset_after_errors": 3,
            # What to do when the model emits STOP:
            #   latch     halt and wait for a new prompt (default; the model said done)
            #   continue  reset the session and keep navigating
            "stop_behavior": "latch",
            "default_prompt": "",
            "autostart": False,
            "verbose": True,
        }
        out = {}
        for k, v in defaults.items():
            self.declare_parameter(k, v)
            out[k] = self.get_parameter(k).value
        return out

    # ------------------------------------------------------------------ callbacks
    def _on_image(self, msg: Image) -> None:
        try:
            rgb = image_msg_to_rgb(msg)
        except ValueError as exc:
            self.get_logger().throttle_duration_sec = 5.0
            self.get_logger().warn(str(exc))
            return
        with self._lock:
            self._frame = rgb
            self._frame_stamp = time.monotonic()

    def _on_prompt(self, msg: String) -> None:
        text = (msg.data or "").strip()
        if not text:
            return
        with self._lock:
            same = (text == self._prompt and self._have_prompt)
            self._prompt = text
            self._have_prompt = True
            if not same or self._state != RUNNING:
                self._state = RUNNING
                self._stop_after_plan = False
                if self._reset_on_prompt_change:
                    self._pending_reset = True
                    self._clear_plan_locked()
        self.get_logger().info(f"prompt: {text!r} -> {self._state}")

    def _on_joy(self, msg: Joy) -> None:
        self._joy_seen = True
        if 0 <= self._joy_button < len(msg.buttons):
            go = bool(msg.buttons[self._joy_button])
            if go != self._joy_go and self._verbose:
                self.get_logger().info(f"joy gate {'OPEN' if go else 'CLOSED'}")
            self._joy_go = go

    # ------------------------------------------------------------------ plan helpers
    def _clear_plan_locked(self) -> None:
        self._segments = []
        self._seg_idx = 0
        self._seg_t = 0.0
        self._starved_t = 0.0
        self._last_seg = None

    def _remaining_locked(self) -> float:
        if self._seg_idx >= len(self._segments):
            return 0.0
        total = sum(s.duration for s in self._segments[self._seg_idx:])
        return max(0.0, total - self._seg_t)

    def _append_plan(self, plan: ChunkPlan) -> None:
        with self._lock:
            # Compact consumed segments so the list cannot grow without bound, keeping the
            # in-flight segment at index 0 with its elapsed time intact.
            self._segments = self._segments[self._seg_idx:]
            self._seg_idx = 0
            if not self._segments:
                self._seg_t = 0.0
            self._segments.extend(plan.segments)
            if plan.stop:
                self._stop_after_plan = True

    def _gate_open(self) -> bool:
        if not self._joy_gate_enable:
            return True
        return self._joy_go

    # ------------------------------------------------------------------ control loop
    def _on_control_tick(self) -> None:
        dt = 1.0 / self._control_hz
        vx, wz = 0.0, 0.0

        with self._lock:
            running = self._state == RUNNING
            gate = self._gate_open()
            if running and gate:
                if self._seg_idx < len(self._segments):
                    seg = self._segments[self._seg_idx]
                    vx, wz = seg.vx, seg.wz
                    self._last_seg = seg
                    self._starved_t = 0.0
                    self._seg_t += dt
                    while (self._seg_idx < len(self._segments)
                           and self._seg_t >= self._segments[self._seg_idx].duration):
                        self._seg_t -= self._segments[self._seg_idx].duration
                        self._seg_idx += 1
                else:
                    # Plan exhausted. Briefly hold the last velocity so a late response
                    # does not show up as a visible stutter, then fall back to zero.
                    self._starved_t += dt
                    if self._stop_after_plan:
                        self._stop_after_plan = False
                        self._clear_plan_locked()
                        if self._stop_behavior == "continue":
                            # Keep going. The server latches its own `terminate` flag on a
                            # STOP and then short-circuits every later request without
                            # running the model, so only a reset clears it.
                            self._pending_reset = True
                            self._stops += 1
                            self.get_logger().info(
                                f"STOP token #{self._stops} -> continuing (session reset)")
                        else:
                            self._state = DONE
                            self.get_logger().info("STOP token reached -> DONE (awaiting new prompt)")
                    elif self._last_seg is not None and self._starved_t <= self._gap_hold_s:
                        vx, wz = self._last_seg.vx, self._last_seg.wz

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._cmd_frame
        msg.twist.linear.x = float(vx)
        msg.twist.angular.z = float(wz)
        self._cmd_pub.publish(msg)

    # ------------------------------------------------------------------ inference loop
    def _infer_loop(self) -> None:
        while not self._shutdown.is_set():
            with self._lock:
                state = self._state
                have_prompt = self._have_prompt
                frame = self._frame
                prompt = self._prompt
                reset = self._pending_reset
                remaining = self._remaining_locked()
                stop_pending = self._stop_after_plan

            if state != RUNNING or not have_prompt or frame is None:
                self._shutdown.wait(0.05)
                continue
            if stop_pending and not reset:
                # A STOP is already committed for the tail of the current plan. Anything
                # requested now is discarded when the plan drains, and against a server
                # that has latched `terminate` it is a stream of 0.01 s no-ops. Wait for
                # the plan to finish so the reset can go out instead.
                self._shutdown.wait(0.05)
                continue
            if remaining > self._request_lead_s:
                # Enough motion still buffered; requesting now would run too far ahead.
                self._shutdown.wait(0.02)
                continue

            try:
                actions, latency = self._request(frame, prompt, reset)
            except Exception as exc:                      # network / server failure
                with self._lock:
                    self._errors += 1
                    self._consec_errors += 1
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    consec = self._consec_errors
                    # A corrupted server session fails every request until it is reset,
                    # so stop retrying into it and force a reset on the next attempt.
                    if consec >= self._reset_after_errors:
                        self._pending_reset = True
                        self._clear_plan_locked()
                self.get_logger().warn(
                    f"request failed ({consec}): {self._last_error}"
                    f"{' -- forcing session reset' if consec >= self._reset_after_errors else ''}")
                # Back off so a dead server is not hammered at the request rate.
                self._shutdown.wait(min(2.0, 0.25 * consec))
                continue

            with self._lock:
                # Clear ONLY the reset this request actually carried. Clearing
                # unconditionally loses a reset raised by the control tick while the
                # request was in flight -- which is exactly what happens on a STOP, and
                # it leaves the server's `terminate` latch set forever.
                if reset:
                    self._pending_reset = False
                    self._resets += 1
                self._consec_errors = 0
                self._requests += 1
                self._last_actions = list(actions)
                self._last_latency = latency
                still_running = self._state == RUNNING

            if not still_running:
                continue                                   # prompt changed mid-flight

            plan = plan_chunk(actions, self._chunk_seconds, self._step_m, self._turn_deg,
                              self._v_max, self._w_max, self._speed_mode,
                              self._execute_tokens, self._honor_stop_beyond)
            if plan.clamped and not self._warn_clamped_once:
                self._warn_clamped_once = True
                self.get_logger().warn(
                    f"speed cap bound (v={plan.v_cmd:.2f} m/s, w={math.degrees(plan.w_cmd):.1f} deg/s)"
                    " -- chunk under-travels vs the trained increments")
            self._append_plan(plan)

            if not plan.segments:
                # Nothing to execute (typically a latched STOP returning [0] instantly).
                # Without a floor here the loop would re-request at the server's reply
                # rate and hammer it at ~100 Hz.
                self._shutdown.wait(0.2)

            if self._verbose:
                self.get_logger().info(
                    f"[{self._requests}] {latency:.2f}s {format_actions(actions)} "
                    f"{list(actions)}{' RESET' if reset else ''}{' STOP' if plan.stop else ''}")

    def _request(self, frame: np.ndarray, prompt: str, reset: bool):
        """POST one frame + instruction, return (actions, latency_seconds)."""
        arr = frame[..., ::-1] if self._bgr_swap else frame
        buf = io.BytesIO()
        PILImage.fromarray(np.ascontiguousarray(arr)).save(
            buf, format="jpeg", quality=self._jpeg_quality)
        buf.seek(0)

        payload = {"reset": bool(reset), "instruction": prompt}
        t0 = time.monotonic()
        resp = requests.post(self._url,
                             files={"image": ("rgb", buf, "image/jpg")},
                             data={"json": json.dumps(payload)},
                             timeout=self._timeout)
        resp.raise_for_status()
        latency = time.monotonic() - t0
        actions = [int(a) for a in resp.json().get("action", [])]
        return actions, latency

    # ------------------------------------------------------------------ status
    def _on_status_tick(self) -> None:
        with self._lock:
            frame_age = (time.monotonic() - self._frame_stamp) if self._frame is not None else None
            status = {
                "state": self._state,
                "prompt": self._prompt,
                "gate_open": self._gate_open(),
                "joy_seen": self._joy_seen,
                "frame_age_s": round(frame_age, 3) if frame_age is not None else None,
                "plan_remaining_s": round(self._remaining_locked(), 3),
                "last_actions": self._last_actions,
                "last_latency_s": round(self._last_latency, 3),
                "chunk_seconds": self._chunk_seconds,
                "requests": self._requests,
                "resets": self._resets,
                "stops": self._stops,
                "stop_behavior": self._stop_behavior,
                "errors": self._errors,
                "consec_errors": self._consec_errors,
                "last_error": self._last_error,
            }
        msg = String()
        msg.data = json.dumps(status)
        self._status_pub.publish(msg)

        # A gate topic nobody publishes to looks exactly like "robot won't move" with no
        # error anywhere. Say so out loud rather than sitting there silently stopped.
        if self._joy_gate_enable and not self._joy_seen and status["state"] == RUNNING:
            self._no_joy_ticks += 1
            if self._no_joy_ticks % 10 == 1:        # ~every 5 s at status_hz=2
                self.get_logger().warn(
                    f"no Joy messages on '{self._joy_topic}' -- the deadman gate can never "
                    f"open, so nothing will move. Check the topic name and that the "
                    f"joystick node is running (ros2 topic hz {self._joy_topic}).")
        else:
            self._no_joy_ticks = 0

    # ------------------------------------------------------------------ shutdown
    def destroy_node(self) -> bool:
        self._shutdown.set()
        try:
            stop = TwistStamped()
            stop.header.stamp = self.get_clock().now().to_msg()
            stop.header.frame_id = self._cmd_frame
            self._cmd_pub.publish(stop)          # leave the base commanded to zero
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StreamVlnPolicyNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
