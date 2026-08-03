"""Open-loop map: StreamVLN discrete actions -> (vx, wz) timed velocity commands.

StreamVLN emits a sequence of 4 discrete tokens (see streamvln_agent.py):
    1  ↑  MOVE FORWARD 25 cm
    2  ←  TURN LEFT   15 deg
    3  →  TURN RIGHT  15 deg
    0  STOP

Unlike the stock go2_vln_client.py (which accumulates actions into a goal pose and
PID-tracks it with odometry feedback = CLOSED loop), this is OPEN loop: each action
is executed as a fixed-duration constant-velocity (vx, wz) command, then a stop.
No odometry, no feedback — dead-simple and controller-agnostic.

Usage:
    from action_to_cmdvel import action_to_cmdvel, execute_sequence
    # 1) pure mapping (framework-agnostic):
    vx, wz, dur = action_to_cmdvel(1)          # (0.25, 0.0, 1.0)  -> forward 25 cm
    # 2) execute a whole sequence via a send(vx, wz) callback you provide:
    stopped = execute_sequence([3,3,1,0], send=lambda vx, wz: pub_twist(vx, wz))
"""
import math
import time

# ── what each token physically means (fixed increments the model was trained on) ──
STEP_M   = 0.25                    # ↑ forward step, metres
TURN_RAD = math.radians(15.0)      # ←/→ turn, radians

# ── open-loop drive speeds — TUNE to your robot's comfortable limits ──
V_FWD  = 0.25                      # m/s   during a forward step  (→ 1.0 s per ↑)
W_TURN = 0.60                      # rad/s during a turn          (→ ~0.44 s per ←/→)


def action_to_cmdvel(a: int):
    """action id -> (vx [m/s], wz [rad/s], duration [s]).
    Open-loop fixed move. STOP / unknown -> all zeros. wz>0 = turn left (CCW)."""
    if a == 1:                                   # ↑ forward 25 cm
        return (V_FWD, 0.0, STEP_M / V_FWD)
    if a == 2:                                   # ← left 15°
        return (0.0,  +W_TURN, TURN_RAD / W_TURN)
    if a == 3:                                   # → right 15°
        return (0.0,  -W_TURN, TURN_RAD / W_TURN)
    return (0.0, 0.0, 0.0)                        # 0 STOP / done


def execute_sequence(actions, send, rate_hz: float = 20.0,
                     stop_pause: float = 0.10, sleep=time.sleep):
    """Run an action sequence open-loop. `send(vx, wz)` is your publish callback
    (e.g. a geometry_msgs/Twist publisher, or the Go2 sport-API move(vx, 0, wz)).
    Holds each action's (vx, wz) for its duration at `rate_hz`, then sends a stop.
    Returns True if a STOP (0) token was reached (task done)."""
    dt = 1.0 / rate_hz
    for a in actions:
        if a == 0:                               # STOP → halt, signal done
            send(0.0, 0.0)
            return True
        vx, wz, dur = action_to_cmdvel(a)
        t = 0.0
        while t < dur:                           # hold the velocity for the fixed duration
            send(vx, wz)
            sleep(dt); t += dt
        send(0.0, 0.0)                           # clean stop between discrete moves
        if stop_pause:
            sleep(stop_pause)
    return False


if __name__ == "__main__":                       # quick self-check
    for a, name in [(1, "↑fwd"), (2, "←left"), (3, "→right"), (0, "STOP")]:
        print(f"{a} {name}: vx,wz,dur = {action_to_cmdvel(a)}")
    log = []
    execute_sequence([3, 3, 1, 0], send=lambda vx, wz: log.append((round(vx, 2), round(wz, 2))),
                     sleep=lambda *_: None)
    print("sample [3,3,1,0] cmd stream (first 5):", log[:5], "…", "STOP reached")
