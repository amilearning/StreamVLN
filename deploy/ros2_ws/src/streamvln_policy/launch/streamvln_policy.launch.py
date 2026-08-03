"""Launch the StreamVLN policy requester.

    ros2 launch streamvln_policy streamvln_policy.launch.py
    ros2 launch streamvln_policy streamvln_policy.launch.py host:=192.168.131.148
    ros2 launch streamvln_policy streamvln_policy.launch.py chunk_seconds:=2.0
    ros2 launch streamvln_policy streamvln_policy.launch.py joy_gate_enable:=false  # BENCH ONLY

The yaml stays authoritative: a command-line argument is forwarded only when it is
non-empty, so unset arguments cannot clobber a typed parameter with an empty string.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# name -> caster used when the argument is actually provided
_OVERRIDES = {
    "host": str,
    "port": int,
    "image_topic": str,
    "prompt_topic": str,
    "cmd_vel_topic": str,
    "chunk_seconds": float,
    "execute_tokens": int,
    "joy_gate_enable": lambda s: s.strip().lower() in ("1", "true", "yes", "on"),
    "joy_button": int,
    "stop_behavior": str,
    "autostart": lambda v: v.strip().lower() in ("1","true","yes","on"),
    "default_prompt": str,
}

_DEFAULT_PARAMS = os.path.join(
    get_package_share_directory("streamvln_policy"), "config", "streamvln_policy.yaml")


def _make_node(context, *_args, **_kwargs):
    overrides = {}
    for name, cast in _OVERRIDES.items():
        raw = LaunchConfiguration(name).perform(context).strip()
        if raw:                                   # empty => leave the yaml value alone
            overrides[name] = cast(raw)

    params = [LaunchConfiguration("params_file").perform(context)]
    if overrides:
        params.append(overrides)

    return [Node(package="streamvln_policy",
                 executable="policy_node",
                 name="streamvln_policy_node",
                 output="screen",
                 emulate_tty=True,
                 parameters=params)]


def generate_launch_description() -> LaunchDescription:
    decls = [DeclareLaunchArgument("params_file", default_value=_DEFAULT_PARAMS)]
    decls += [DeclareLaunchArgument(name, default_value="") for name in _OVERRIDES]
    return LaunchDescription(decls + [OpaqueFunction(function=_make_node)])
