#!/usr/bin/env bash
# Start the StreamVLN ROS 2 policy node (no GPU needed).
#
#   ./deploy/run_policy.sh                        # normal: deadman gate ON, real cmd_vel
#   ./deploy/run_policy.sh --bench                # SAFE: scratch topic, no gate, nothing moves
#   ./deploy/run_policy.sh --host 192.168.1.50    # server on another machine
#   ./deploy/run_policy.sh --tokens 4 --chunk 2.0
#
# Builds the workspace on first run. Extra args are passed through to ros2 launch.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$HERE/ros2_ws"
ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"

EXTRA=()
BENCH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bench)  BENCH=1; shift ;;
    --host)   EXTRA+=("host:=$2"); shift 2 ;;
    --port)   EXTRA+=("port:=$2"); shift 2 ;;
    --chunk)  EXTRA+=("chunk_seconds:=$2"); shift 2 ;;
    --tokens) EXTRA+=("execute_tokens:=$2"); shift 2 ;;
    --topic)  EXTRA+=("cmd_vel_topic:=$2"); shift 2 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *)        EXTRA+=("$1"); shift ;;
  esac
done

if [[ $BENCH -eq 1 ]]; then
  # Nothing can reach the base: output goes to a scratch topic and the gate is off so
  # the loop runs without a joystick. Use this to watch behaviour before going live.
  EXTRA+=("cmd_vel_topic:=/streamvln/test_cmd_vel" "joy_gate_enable:=false")
  echo ">>> BENCH MODE: publishing to /streamvln/test_cmd_vel, deadman gate DISABLED"
  echo ">>> nothing reaches the base."
fi

[[ -f "$ROS_SETUP" ]] || { echo "ERROR: no ROS 2 at $ROS_SETUP (set ROS_SETUP=...)" >&2; exit 1; }
# ROS setup scripts reference unbound variables, so -u has to come off around them.
set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"

if [[ ! -f "$WS/install/setup.bash" ]]; then
  echo "building workspace (first run)..."
  (cd "$WS" && colcon build --packages-select streamvln_policy)
fi
# shellcheck disable=SC1091
source "$WS/install/setup.bash"
set -u

echo "launching streamvln_policy_node ${EXTRA[*]:-}"
echo "  give it an instruction with:"
echo "    ros2 topic pub -1 /prompt std_msgs/msg/String \"{data: 'Walk forward and stop at the door.'}\""
exec ros2 launch streamvln_policy streamvln_policy.launch.py "${EXTRA[@]}"
