"""StreamVLN discrete action tokens -> timed (vx, wz) velocity segments.

StreamVLN emits a fixed-length sequence of discrete tokens (see streamvln_agent.py):

    1  UP     MOVE FORWARD by `step_m` (0.25 m as trained)
    2  LEFT   TURN LEFT   by `turn_deg` (15 deg as trained)
    3  RIGHT  TURN RIGHT  by `turn_deg`
    0  STOP   task complete

This module turns one such chunk into constant-velocity segments that play out over a
fixed wall-clock budget (`chunk_seconds`). It is deliberately free of ROS and of any
StreamVLN import so it can be unit-tested standalone.

Two speed modes, and the difference matters more than it looks:

  geometric  Speeds are derived from the time budget so each token still covers the
             increment the model was TRAINED on (0.25 m / 15 deg). v_max / w_max act
             purely as a safety clamp; if a clamp binds, the caller is told, because a
             bound clamp means the robot silently under-travels relative to what
             StreamVLN believes it did -- and the model's memory assumes those exact
             increments.

  capped     Obey the speed limits first and accept the shortfall. Honest about being
             slower than the model thinks, and the shortfall is reported.
"""
import math
from typing import List, NamedTuple, Sequence, Tuple

STOP, FORWARD, LEFT, RIGHT = 0, 1, 2, 3
TOKENS_PER_CHUNK = 4


class Segment(NamedTuple):
    """One constant-velocity slice: hold (vx, wz) for `duration` seconds."""
    vx: float
    wz: float
    duration: float
    token: int


class ChunkPlan(NamedTuple):
    segments: List[Segment]
    stop: bool           # a STOP token was reached; nothing after it is executed
    clamped: bool        # a speed cap bound, so the chunk under-travels
    v_cmd: float         # the forward speed actually used
    w_cmd: float         # the turn rate actually used


def chunk_speeds(chunk_seconds: float,
                 step_m: float = 0.25,
                 turn_deg: float = 15.0,
                 v_max: float = 1.0,
                 w_max: float = 1.2,
                 tokens: int = TOKENS_PER_CHUNK,
                 mode: str = "geometric") -> Tuple[float, float, bool]:
    """Speeds needed to play `tokens` tokens in `chunk_seconds`.

    Returns (v_cmd, w_cmd, clamped). `clamped` is True when a cap bound and the chunk
    will therefore cover less ground than the model assumes.
    """
    if chunk_seconds <= 0.0:
        raise ValueError("chunk_seconds must be > 0")
    if tokens <= 0:
        raise ValueError("tokens must be > 0")

    per_token = chunk_seconds / float(tokens)
    v_want = step_m / per_token
    w_want = math.radians(turn_deg) / per_token

    v_cmd = min(v_want, v_max)
    w_cmd = min(w_want, w_max)
    clamped = (v_cmd < v_want - 1e-9) or (w_cmd < w_want - 1e-9)

    if mode not in ("geometric", "capped"):
        raise ValueError(f"unknown speed mode: {mode!r}")
    # Both modes clamp identically -- they differ only in what the caller is expected to
    # do about it. 'geometric' treats a bound clamp as a misconfiguration to warn about;
    # 'capped' treats it as the intended trade.
    return v_cmd, w_cmd, clamped


def plan_chunk(actions: Sequence[int],
               chunk_seconds: float,
               step_m: float = 0.25,
               turn_deg: float = 15.0,
               v_max: float = 1.0,
               w_max: float = 1.2,
               mode: str = "geometric") -> ChunkPlan:
    """Turn one action chunk into timed velocity segments.

    The per-token duration is `chunk_seconds / TOKENS_PER_CHUNK` regardless of how many
    tokens actually arrived, so a short or truncated chunk keeps the same cadence rather
    than stretching to fill the budget.
    """
    v_cmd, w_cmd, clamped = chunk_speeds(
        chunk_seconds, step_m, turn_deg, v_max, w_max, TOKENS_PER_CHUNK, mode)
    per_token = chunk_seconds / float(TOKENS_PER_CHUNK)

    segments: List[Segment] = []
    stop = False
    for a in actions:
        a = int(a)
        if a == STOP:
            stop = True
            break                                  # nothing after a STOP is executed
        if a == FORWARD:
            segments.append(Segment(v_cmd, 0.0, per_token, a))
        elif a == LEFT:
            segments.append(Segment(0.0, +w_cmd, per_token, a))
        elif a == RIGHT:
            segments.append(Segment(0.0, -w_cmd, per_token, a))
        else:
            # Unknown token: hold still for its slot rather than guessing.
            segments.append(Segment(0.0, 0.0, per_token, a))

    return ChunkPlan(segments, stop, clamped, v_cmd, w_cmd)


def format_actions(actions: Sequence[int]) -> str:
    """Compact human-readable form, e.g. [3,3,1,0] -> '>>^.'"""
    glyph = {STOP: ".", FORWARD: "^", LEFT: "<", RIGHT: ">"}
    return "".join(glyph.get(int(a), "?") for a in actions)


if __name__ == "__main__":                          # quick self-check
    for cs in (1.0, 2.0):
        v, w, c = chunk_speeds(cs)
        print(f"chunk={cs}s -> v={v:.3f} m/s, w={math.degrees(w):.1f} deg/s, clamped={c}")
    p = plan_chunk([3, 3, 1, 0], chunk_seconds=1.0)
    print("plan [3,3,1,0]:", format_actions([3, 3, 1, 0]),
          f"stop={p.stop} segments={len(p.segments)} total={sum(s.duration for s in p.segments):.2f}s")
    for s in p.segments:
        print(f"   token={s.token} vx={s.vx:.2f} wz={s.wz:+.2f} dur={s.duration:.2f}")
