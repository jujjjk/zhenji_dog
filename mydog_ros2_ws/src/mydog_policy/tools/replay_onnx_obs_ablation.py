#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replay one real-robot CSV observation through ONNX, then ablate observation groups.

Example:
    cd ~/mydog_ros2_ws
    source install/setup.bash

    python3 src/mydog_policy/tools/replay_onnx_obs_ablation.py \
      --csv_path /home/jetson/mydog_ros2_ws/log/前进_原scale_10nm_v005_20260709_180015.csv \
      --onnx_path /home/jetson/mydog_ros2_ws/src/mydog_policy/resource/fanfan_yaw_clean_5100.onnx \
      --target_cmd_x 0.06
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np


DEFAULT_CSV_PATH = (
    "/home/jetson/mydog_ros2_ws/log/"
    "前进_原scale_10nm_v005_20260709_180015.csv"
)
DEFAULT_ONNX_PATH = (
    "/home/jetson/mydog_ros2_ws/src/mydog_policy/resource/"
    "fanfan_yaw_clean_5100.onnx"
)
DEFAULT_LOG_DIR = "/home/jetson/mydog_ros2_ws/log"

DISPLAY_JOINTS = [
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
]
DISPLAY_FULL_NAMES = [f"{name}_joint" for name in DISPLAY_JOINTS]
FOCUS_FULL_NAMES = [
    "FL_thigh_joint",
    "RR_thigh_joint",
    "RL_hip_joint",
    "FL_hip_joint",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replay a real CSV ONNX observation and test which observation groups "
            "drive asymmetric raw actions."
        )
    )
    parser.add_argument("--csv_path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--onnx_path", default=DEFAULT_ONNX_PATH)
    parser.add_argument("--target_cmd_x", type=float, default=0.06)
    parser.add_argument("--cmd_tol", type=float, default=0.003)
    parser.add_argument(
        "--timestamp",
        default="",
        help="Optional exact CSV timestamp to replay. If omitted, selects the middle of the longest target cmd_x segment.",
    )
    return parser.parse_args()


def import_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime is not installed.")
        print("Install it with:")
        print("  pip install onnxruntime")
        sys.exit(2)
    return ort


def load_deployment_config(session):
    metadata = session.get_modelmeta().custom_metadata_map
    raw_config = metadata.get("fanfan_deployment_config")
    if not raw_config:
        print("ERROR: ONNX is missing fanfan_deployment_config metadata.")
        sys.exit(2)
    try:
        return json.loads(raw_config)
    except json.JSONDecodeError as exc:
        print(f"ERROR: invalid fanfan_deployment_config JSON: {exc}")
        sys.exit(2)


def is_static_int(value):
    return isinstance(value, int) and value > 0


def concrete_shape(shape):
    concrete = []
    for dim in shape:
        concrete.append(int(dim) if is_static_int(dim) else 1)
    return tuple(concrete)


def numpy_dtype(onnx_type):
    if onnx_type in ("tensor(double)", "tensor(float64)"):
        return np.float64
    if onnx_type in ("tensor(int64)",):
        return np.int64
    if onnx_type in ("tensor(int32)",):
        return np.int32
    return np.float32


def find_obs_input(inputs, obs_dim):
    for item in inputs:
        shape = list(item.shape)
        if shape and shape[-1] == obs_dim:
            return item
        if len(shape) >= 2 and shape[1] == obs_dim:
            return item
    return inputs[0]


def resolve_obs_dim(obs_input, config):
    meta_dim = int(config.get("dimensions", {}).get("observations", -1))
    shape = list(obs_input.shape)
    if shape and is_static_int(shape[-1]):
        graph_dim = int(shape[-1])
        if meta_dim > 0 and meta_dim != graph_dim:
            print(
                "ERROR: ONNX graph and metadata observation dimensions differ: "
                f"graph={graph_dim}, metadata={meta_dim}"
            )
            sys.exit(2)
        return graph_dim
    if meta_dim in (36, 48, 50, 52):
        return meta_dim
    print(f"ERROR: cannot infer observation dim from input shape={shape}")
    sys.exit(2)


def obs_feed_array(obs, obs_input):
    shape = list(obs_input.shape)
    if len(shape) == 1:
        actual = obs.reshape(obs.shape[0])
    elif len(shape) == 2:
        actual = obs.reshape(1, obs.shape[0])
    else:
        print(f"ERROR: unsupported observation input rank: shape={shape}")
        sys.exit(2)
    if shape and is_static_int(shape[-1]) and int(shape[-1]) != obs.shape[0]:
        print("ERROR: ONNX input dimension and constructed obs dimension mismatch.")
        print(f"  expected input shape: {shape}")
        print(f"  actual obs shape: {actual.shape}")
        sys.exit(2)
    return actual.astype(np.float32)


def output_action(raw_output, output_transform):
    action = np.asarray(raw_output, dtype=np.float32).reshape(-1)
    if action.shape[0] < 12:
        print(f"ERROR: ONNX output has fewer than 12 values: shape={action.shape}")
        sys.exit(2)
    action = action[:12]
    if output_transform == "tanh":
        action = np.tanh(action)
    elif output_transform not in ("identity", "none"):
        print(f"ERROR: unsupported output_transform={output_transform!r}")
        sys.exit(2)
    return action.astype(np.float32)


def read_csv_groups(path):
    if not os.path.exists(path):
        print(f"ERROR: CSV file does not exist: {path}")
        sys.exit(2)
    groups = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        required = [
            "time",
            "policy_index",
            "policy_joint_name",
            "cmd_x",
            "obs_base_lin_x",
            "obs_base_lin_y",
            "obs_base_lin_z",
            "obs_base_ang_x",
            "obs_base_ang_y",
            "obs_base_ang_z",
            "obs_gravity_x",
            "obs_gravity_y",
            "obs_gravity_z",
            "obs_cmd_x",
            "obs_cmd_y",
            "obs_cmd_wz",
            "obs_joint_pos_policy",
            "obs_joint_vel_policy",
            "obs_last_action_policy",
            "action_raw_policy",
        ]
        missing = [name for name in required if name not in header]
        if missing:
            print(f"ERROR: CSV is missing required columns: {missing}")
            sys.exit(2)
        for row in reader:
            groups[row["time"]].append(row)
    return groups


def f(row, key, default=0.0):
    value = row.get(key, "")
    if value is None or value == "":
        return float(default)
    return float(value)


def vector_by_policy(rows, key):
    vec = np.full(12, np.nan, dtype=np.float32)
    names = [None] * 12
    for row in rows:
        idx = int(float(row["policy_index"]))
        if not 0 <= idx < 12:
            continue
        vec[idx] = f(row, key)
        names[idx] = row.get("policy_joint_name", "")
    if not np.all(np.isfinite(vec)):
        print(f"ERROR: timestamp is missing policy indices for {key}: {vec}")
        sys.exit(2)
    return vec.astype(np.float32), names


def select_timestamp(groups, target_cmd_x, cmd_tol, timestamp=""):
    cycle_rows = []
    for t, rows in groups.items():
        if len(rows) < 12:
            continue
        try:
            first = rows[0]
            cycle_rows.append((float(t), t, rows, f(first, "cmd_x")))
        except Exception:
            continue
    cycle_rows.sort(key=lambda item: item[0])
    if not cycle_rows:
        print("ERROR: no complete timestamp groups found in CSV.")
        sys.exit(2)

    if timestamp:
        if timestamp not in groups:
            print(f"ERROR: requested timestamp {timestamp!r} not found.")
            sys.exit(2)
        return timestamp, groups[timestamp]

    candidates = [
        item for item in cycle_rows if abs(item[3] - float(target_cmd_x)) <= float(cmd_tol)
    ]
    if not candidates:
        print(
            "ERROR: no timestamp matches "
            f"target_cmd_x={target_cmd_x} within cmd_tol={cmd_tol}."
        )
        sys.exit(2)

    runs = []
    current = [candidates[0]]
    for item in candidates[1:]:
        if item[0] - current[-1][0] <= 0.35:
            current.append(item)
        else:
            runs.append(current)
            current = [item]
    runs.append(current)
    best = max(runs, key=len)
    selected = best[len(best) // 2]
    return selected[1], selected[2]


def build_obs_from_csv(rows, obs_dim):
    rows_sorted = sorted(rows, key=lambda row: int(float(row["policy_index"])))
    first = rows_sorted[0]
    obs = np.zeros(obs_dim, dtype=np.float32)

    obs[0:3] = np.asarray(
        [f(first, "obs_base_lin_x"), f(first, "obs_base_lin_y"), f(first, "obs_base_lin_z")],
        dtype=np.float32,
    )
    obs[3:6] = np.asarray(
        [f(first, "obs_base_ang_x"), f(first, "obs_base_ang_y"), f(first, "obs_base_ang_z")],
        dtype=np.float32,
    )
    obs[6:9] = np.asarray(
        [f(first, "obs_gravity_x"), f(first, "obs_gravity_y"), f(first, "obs_gravity_z")],
        dtype=np.float32,
    )
    obs[9:12] = np.asarray(
        [f(first, "obs_cmd_x"), f(first, "obs_cmd_y"), f(first, "obs_cmd_wz")],
        dtype=np.float32,
    )
    obs[12:24], policy_names = vector_by_policy(rows_sorted, "obs_joint_pos_policy")
    obs[24:36], _ = vector_by_policy(rows_sorted, "obs_joint_vel_policy")
    if obs_dim >= 48:
        obs[36:48], _ = vector_by_policy(rows_sorted, "obs_last_action_policy")
    if obs_dim >= 50:
        phase = f(first, "global_gait_phase", f(first, "cpg_phase", 0.0)) % 1.0
        obs[48:50] = np.asarray(
            [math.sin(2.0 * math.pi * phase), math.cos(2.0 * math.pi * phase)],
            dtype=np.float32,
        )
    if obs_dim >= 52:
        # The current CSV does not store heading obs directly.  For yaw=0 clean
        # replay this matches ObsBuilder's no-heading fallback.
        obs[50:52] = np.asarray([0.0, 1.0], dtype=np.float32)

    csv_action, _ = vector_by_policy(rows_sorted, "action_raw_policy")
    return obs.astype(np.float32), csv_action.astype(np.float32), policy_names


def clean_obs_like(real_obs):
    clean = np.zeros_like(real_obs)
    clean[0:3] = 0.0
    clean[3:6] = 0.0
    clean[6:9] = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    clean[9:12] = real_obs[9:12]
    clean[12:24] = 0.0
    clean[24:36] = 0.0
    if clean.shape[0] >= 48:
        clean[36:48] = 0.0
    if clean.shape[0] >= 50:
        clean[48:50] = real_obs[48:50]
    if clean.shape[0] >= 52:
        clean[50:52] = real_obs[50:52]
    return clean.astype(np.float32)


def make_cases(real_obs):
    cases = []

    def add(name, obs):
        cases.append((name, obs.astype(np.float32).copy()))

    add("A_real_obs", real_obs)

    obs = real_obs.copy()
    if obs.shape[0] >= 48:
        obs[36:48] = 0.0
    add("B_zero_last_action", obs)

    obs = real_obs.copy()
    obs[12:24] = 0.0
    add("C_zero_joint_pos", obs)

    obs = real_obs.copy()
    obs[24:36] = 0.0
    add("D_zero_joint_vel", obs)

    obs = real_obs.copy()
    obs[6:9] = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    add("E_neutral_gravity", obs)

    obs = real_obs.copy()
    obs[3:6] = 0.0
    add("F_zero_ang_vel", obs)

    clean = clean_obs_like(real_obs)
    obs = clean.copy()
    if obs.shape[0] >= 48 and real_obs.shape[0] >= 48:
        obs[36:48] = real_obs[36:48]
    add("G_clean_but_real_last_action", obs)

    obs = clean.copy()
    obs[12:24] = real_obs[12:24]
    add("H_clean_but_real_joint_pos", obs)

    obs = clean.copy()
    obs[12:24] = real_obs[12:24]
    if obs.shape[0] >= 48 and real_obs.shape[0] >= 48:
        obs[36:48] = real_obs[36:48]
    add("I_clean_but_real_joint_pos_and_last_action", obs)

    return cases


def run_onnx(session, outputs, obs_input, extra_feeds, obs, output_transform):
    feed = dict(extra_feeds)
    feed[obs_input.name] = obs_feed_array(obs, obs_input)
    raw_outputs = session.run([item.name for item in outputs], feed)
    return output_action(raw_outputs[0], output_transform)


def display_order_action(action, joint_names):
    by_name = {name: float(action[i]) for i, name in enumerate(joint_names)}
    missing = [name for name in DISPLAY_FULL_NAMES if name not in by_name]
    if missing:
        print(f"ERROR: ONNX joint_names missing required joints: {missing}")
        sys.exit(2)
    return np.asarray([by_name[name] for name in DISPLAY_FULL_NAMES], dtype=np.float32)


def focus_values(action_display):
    by_name = {
        full_name: float(action_display[i])
        for i, full_name in enumerate(DISPLAY_FULL_NAMES)
    }
    return {name.replace("_joint", ""): by_name[name] for name in FOCUS_FULL_NAMES}


def print_case(case_name, action_display):
    print("")
    print(case_name)
    print(f"{'joint_name':<10s}  {'action_raw':>11s}")
    print("-" * 24)
    for joint, value in zip(DISPLAY_JOINTS, action_display):
        print(f"{joint:<10s}  {float(value):+11.6f}")
    focus = focus_values(action_display)
    print("Focus:", ", ".join(f"{k}={v:+.6f}" for k, v in focus.items()))


def print_judgment(results):
    real = results["A_real_obs"]
    focus_real = focus_values(real)

    def drop(case_name, joint):
        case_focus = focus_values(results[case_name])
        if "thigh" in joint:
            return focus_real[joint] - case_focus[joint]
        return case_focus[joint] - focus_real[joint]

    print("")
    print("Ablation judgment")
    print("-" * 60)
    last_drop = max(
        drop("B_zero_last_action", "FL_thigh"),
        drop("B_zero_last_action", "RR_thigh"),
        drop("B_zero_last_action", "RL_hip"),
        drop("B_zero_last_action", "FL_hip"),
    )
    joint_pos_drop = max(
        drop("C_zero_joint_pos", "FL_thigh"),
        drop("C_zero_joint_pos", "RR_thigh"),
        drop("C_zero_joint_pos", "RL_hip"),
        drop("C_zero_joint_pos", "FL_hip"),
    )

    print(f"zero_last_action max focus relief: {last_drop:+.6f}")
    print(f"zero_joint_pos   max focus relief: {joint_pos_drop:+.6f}")

    if last_drop > 0.30:
        print(
            "If zero_last_action makes FL_thigh/RR_thigh or RL_hip/FL_hip "
            "drop strongly, last_action self-feedback is a primary suspect."
        )
    if joint_pos_drop > 0.30:
        print(
            "If zero_joint_pos makes the biased joints drop strongly, "
            "q_current/default/zero mapping is a primary suspect."
        )

    clean_last = focus_values(results["G_clean_but_real_last_action"])
    clean_joint = focus_values(results["H_clean_but_real_joint_pos"])
    clean_both = focus_values(results["I_clean_but_real_joint_pos_and_last_action"])

    def reproduces_bias(focus):
        return (
            focus["FL_thigh"] > 0.70
            or focus["RR_thigh"] > 0.70
            or focus["RL_hip"] < -0.60
            or focus["FL_hip"] < -0.60
        )

    if reproduces_bias(clean_last):
        print(
            "clean_but_real_last_action reproduces strong bias: "
            "last_action order/feedback is the biggest suspect."
        )
    if reproduces_bias(clean_joint):
        print(
            "clean_but_real_joint_pos reproduces strong bias: "
            "joint_pos/default/zero mapping is the biggest suspect."
        )
    if reproduces_bias(clean_both):
        print(
            "clean_but_real_joint_pos_and_last_action reproduces strong bias: "
            "joint_pos and last_action together are sufficient to trigger it."
        )
    if last_drop <= 0.30 and joint_pos_drop <= 0.30 and not (
        reproduces_bias(clean_last) or reproduces_bias(clean_joint) or reproduces_bias(clean_both)
    ):
        print(
            "No single tested group strongly explains the bias. Inspect gravity, "
            "base velocity, gait phase, or model/checkpoint behavior next."
        )


def write_csv(results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(DEFAULT_LOG_DIR, f"onnx_obs_ablation_{timestamp}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case"] + DISPLAY_JOINTS)
        for case_name, action_display in results.items():
            writer.writerow([case_name] + [f"{float(v):.9f}" for v in action_display])
    return path


def main():
    args = parse_args()
    ort = import_onnxruntime()
    if not os.path.exists(args.onnx_path):
        print(f"ERROR: ONNX file does not exist: {args.onnx_path}")
        sys.exit(2)

    session = ort.InferenceSession(
        args.onnx_path,
        providers=["CPUExecutionProvider"],
    )
    config = load_deployment_config(session)
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if not inputs or not outputs:
        print("ERROR: ONNX session has no inputs or no outputs.")
        sys.exit(2)

    joint_names = list(config.get("joint_names", []))
    if len(joint_names) != 12:
        print(f"ERROR: expected 12 joint_names in ONNX metadata, got {len(joint_names)}")
        sys.exit(2)
    obs_dim_meta = int(config.get("dimensions", {}).get("observations", -1))
    obs_input = find_obs_input(inputs, obs_dim_meta)
    obs_dim = resolve_obs_dim(obs_input, config)
    output_transform = str(
        config.get("control", {}).get("output_transform", "identity")
    ).lower()

    groups = read_csv_groups(args.csv_path)
    selected_time, selected_rows = select_timestamp(
        groups,
        target_cmd_x=args.target_cmd_x,
        cmd_tol=args.cmd_tol,
        timestamp=args.timestamp,
    )
    real_obs, csv_action_policy, csv_policy_names = build_obs_from_csv(
        selected_rows, obs_dim
    )

    extra_feeds = {}
    for item in inputs:
        if item.name == obs_input.name:
            continue
        extra_feeds[item.name] = np.zeros(
            concrete_shape(item.shape),
            dtype=numpy_dtype(item.type),
        )

    print("ONNX observation ablation replay")
    print(f"csv_path: {args.csv_path}")
    print(f"onnx_path: {args.onnx_path}")
    print(f"selected timestamp: {selected_time}")
    print(f"selected cmd_x: {f(selected_rows[0], 'cmd_x'):.6f}")
    print(f"selected obs_cmd_x: {f(selected_rows[0], 'obs_cmd_x'):.6f}")
    print("")
    print("Inputs:")
    for item in inputs:
        marker = "  <-- observation" if item.name == obs_input.name else ""
        print(f"  name={item.name!r} shape={item.shape} type={item.type}{marker}")
    print("Outputs:")
    for item in outputs:
        print(f"  name={item.name!r} shape={item.shape} type={item.type}")
    print("")
    print(f"observation_dim: {obs_dim}")
    print(f"ONNX metadata policy order: {joint_names}")
    print(f"CSV policy order by policy_index: {csv_policy_names}")
    print(f"display order: {DISPLAY_JOINTS}")
    if len(inputs) > 1:
        print("Extra ONNX inputs detected; initializing them to zero.")
    else:
        print("Single-frame ONNX observation input detected.")
    print(
        "Clean ablation cases keep the selected real command and gait/heading "
        "terms, while neutralizing state terms."
    )

    cases = make_cases(real_obs)
    results = {}
    for case_name, obs in cases:
        action_policy = run_onnx(
            session,
            outputs,
            obs_input,
            extra_feeds,
            obs,
            output_transform,
        )
        action_display = display_order_action(action_policy, joint_names)
        results[case_name] = action_display
        print_case(case_name, action_display)

    replay_policy = run_onnx(
        session,
        outputs,
        obs_input,
        extra_feeds,
        real_obs,
        output_transform,
    )
    replay_diff = replay_policy - csv_action_policy
    print("")
    print("Replay check against CSV action_raw_policy in ONNX policy order")
    print(f"max_abs_diff: {float(np.max(np.abs(replay_diff))):.9f}")
    print(f"mean_abs_diff: {float(np.mean(np.abs(replay_diff))):.9f}")
    print("policy_index  joint_name          csv_action    replay_action          diff")
    print("-" * 74)
    for i, name in enumerate(joint_names):
        print(
            f"{i:12d}  {name:<16s}  "
            f"{float(csv_action_policy[i]):+11.6f}  "
            f"{float(replay_policy[i]):+13.6f}  "
            f"{float(replay_diff[i]):+11.6f}"
        )

    print_judgment(results)
    csv_path = write_csv(results)
    print("")
    print(f"CSV written: {csv_path}")


if __name__ == "__main__":
    main()
