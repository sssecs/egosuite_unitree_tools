#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

BODY_KEY = "observation.state.body_world"
BODY_ROT_KEY = "observation.state.body_world_rotation"
LEFT_HAND_ROT_KEY = "observation.state.hand_left_world_rotation"
RIGHT_HAND_ROT_KEY = "observation.state.hand_right_world_rotation"
FRAME_KEY = "frame_index"
TIMESTAMP_KEY = "timestamp"

PELVIS = 0
LEFT_ANKLE = 7
RIGHT_ANKLE = 8
LEFT_FOOT = 10
RIGHT_FOOT = 11
HEAD = 15
LEFT_SHOULDER = 16
RIGHT_SHOULDER = 17
LEFT_ELBOW = 18
RIGHT_ELBOW = 19
LEFT_WRIST = 20
RIGHT_WRIST = 21

G1_HEIGHT_M = 1.320
G1_WRIST_SPAN_M = 0.85219

CONTROL_INTERFACE_VERSION = "3.2"


def normalize_quaternion(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(n, 1e-12)


def quat_multiply(q1, q2):
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    out = np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)
    return normalize_quaternion(out)


def find_episode_roots(root):
    root = Path(root)
    roots = []
    for info in root.rglob("meta/info.json"):
        if ".cache" in info.parts:
            continue
        ep = info.parent.parent
        if (ep / "data").exists():
            roots.append(ep)
    return sorted(roots)


def _read_pose_column(table, key, shape):
    values = table[key].to_pylist()
    out = np.full((len(values), *shape), np.nan, dtype=np.float64)
    expected = int(np.prod(shape))
    for i, value in enumerate(values):
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float64)
        if arr.size != expected:
            raise RuntimeError(
                f"Unexpected shape in {key} row {i}: {arr.shape}, expected {shape}"
            )
        out[i] = arr.reshape(shape)
    return out


def load_episode(episode_root):
    import pyarrow.dataset as pads

    episode_root = Path(episode_root)
    with (episode_root / "meta" / "info.json").open("r", encoding="utf-8") as f:
        info = json.load(f)

    fps = float(info.get("fps", 30.0))
    dataset = pads.dataset(str(episode_root / "data"), format="parquet")
    available = set(dataset.schema.names)

    if BODY_KEY not in available:
        raise RuntimeError(f"{BODY_KEY} missing; full-body pose is required.")

    wanted = [
        BODY_KEY,
        BODY_ROT_KEY,
        LEFT_HAND_ROT_KEY,
        RIGHT_HAND_ROT_KEY,
        FRAME_KEY,
        TIMESTAMP_KEY,
    ]
    table = dataset.to_table(columns=[k for k in wanted if k in available])

    data = {
        "body": _read_pose_column(table, BODY_KEY, (22, 3)),
        "body_rot": (
            _read_pose_column(table, BODY_ROT_KEY, (22, 4))
            if BODY_ROT_KEY in table.column_names else None
        ),
        "left_hand_rot": (
            _read_pose_column(table, LEFT_HAND_ROT_KEY, (21, 4))
            if LEFT_HAND_ROT_KEY in table.column_names else None
        ),
        "right_hand_rot": (
            _read_pose_column(table, RIGHT_HAND_ROT_KEY, (21, 4))
            if RIGHT_HAND_ROT_KEY in table.column_names else None
        ),
    }

    if FRAME_KEY in table.column_names:
        frame_index = np.asarray(table[FRAME_KEY].to_pylist())
        order = np.argsort(frame_index)
        frame_index = frame_index[order]
    else:
        order = np.arange(len(data["body"]))
        frame_index = np.arange(len(data["body"]))

    for key in ("body", "body_rot", "left_hand_rot", "right_hand_rot"):
        if data[key] is not None:
            data[key] = data[key][order]

    if TIMESTAMP_KEY in table.column_names:
        timestamps = np.asarray(
            table[TIMESTAMP_KEY].to_pylist(),
            dtype=np.float64,
        )[order]
    else:
        timestamps = frame_index.astype(np.float64) / fps

    data["frame_index"] = frame_index
    data["timestamps"] = timestamps
    data["fps"] = fps
    data["info"] = info
    return data


def _median_distance(a, b):
    d = np.linalg.norm(a - b, axis=-1)
    d = d[np.isfinite(d)]
    return float(np.median(d)) if len(d) else np.nan


def estimate_human_wrist_span(body):
    return float(
        _median_distance(body[:, LEFT_SHOULDER], body[:, RIGHT_SHOULDER])
        + _median_distance(body[:, LEFT_SHOULDER], body[:, LEFT_ELBOW])
        + _median_distance(body[:, LEFT_ELBOW], body[:, LEFT_WRIST])
        + _median_distance(body[:, RIGHT_SHOULDER], body[:, RIGHT_ELBOW])
        + _median_distance(body[:, RIGHT_ELBOW], body[:, RIGHT_WRIST])
    )


def moving_median_1d(x, window):
    x = np.asarray(x, dtype=np.float64)
    if window <= 1 or len(x) <= 1:
        return x.copy()
    window = int(window)
    if window % 2 == 0:
        window += 1
    max_window = len(x) if len(x) % 2 else len(x) - 1
    window = max(1, min(window, max_window))
    if window <= 1:
        return x.copy()
    h = window // 2
    padded = np.pad(x, (h, h), mode="edge")
    sw = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(sw, axis=-1)


def moving_median_xy(xy, window):
    xy = np.asarray(xy, dtype=np.float64)
    if window <= 1:
        return xy.copy()
    out = np.empty_like(xy)
    out[:, 0] = moving_median_1d(xy[:, 0], window)
    out[:, 1] = moving_median_1d(xy[:, 1], window)
    return out


def estimate_ground_z_per_frame(body, window=5):
    foot_z = np.stack(
        [body[:, LEFT_FOOT, 2], body[:, RIGHT_FOOT, 2]],
        axis=1,
    )
    ground = np.full(len(body), np.nan, dtype=np.float64)

    for i in range(len(body)):
        valid = foot_z[i, np.isfinite(foot_z[i])]
        if len(valid):
            ground[i] = np.min(valid)

    missing = ~np.isfinite(ground)
    if np.any(missing):
        ankle_z = np.stack(
            [body[:, LEFT_ANKLE, 2], body[:, RIGHT_ANKLE, 2]],
            axis=1,
        )
        for i in np.where(missing)[0]:
            valid = ankle_z[i, np.isfinite(ankle_z[i])]
            if len(valid):
                ground[i] = np.min(valid)

    valid_idx = np.where(np.isfinite(ground))[0]
    if len(valid_idx) == 0:
        raise RuntimeError("Cannot estimate ground from feet/ankles.")

    missing_idx = np.where(~np.isfinite(ground))[0]
    if len(missing_idx):
        ground[missing_idx] = np.interp(
            missing_idx, valid_idx, ground[valid_idx]
        )

    return moving_median_1d(ground, window)


def estimate_ground_reference(body, window=5):
    trace = estimate_ground_z_per_frame(body, window=window)
    return float(np.median(trace)), trace


def estimate_human_height_proxy(body, ground_z_reference):
    h = body[:, HEAD, 2] - float(ground_z_reference)
    h = h[np.isfinite(h)]
    if len(h) == 0:
        raise RuntimeError("Cannot estimate human height proxy.")
    return float(np.percentile(h, 95.0))


def robust_initial_xy(xy, timestamps, seconds=1.0):
    xy = np.asarray(xy, dtype=np.float64)
    ts = np.asarray(timestamps, dtype=np.float64)
    if len(xy) == 0:
        raise RuntimeError("Empty trajectory.")
    ids = np.where(ts <= ts[0] + float(seconds))[0]
    if len(ids) < min(5, len(xy)):
        ids = np.arange(min(5, len(xy)))
    return np.nanmedian(xy[ids], axis=0)


def torso_heading_from_shoulders(body):
    # EgoSuite: +X forward, +Y left.
    y_left = (
        body[:, LEFT_SHOULDER, :2]
        - body[:, RIGHT_SHOULDER, :2]
    )
    n = np.linalg.norm(y_left, axis=1, keepdims=True)
    y_left = y_left / np.maximum(n, 1e-12)
    x_forward = np.stack([y_left[:, 1], -y_left[:, 0]], axis=1)
    return np.unwrap(np.arctan2(x_forward[:, 1], x_forward[:, 0]))


def _wrist_orientation(body_rot, hand_rot, body_idx, n):
    if hand_rot is not None:
        return normalize_quaternion(hand_rot[:, 0]), "hand wrist rotation"
    if body_rot is not None:
        return normalize_quaternion(body_rot[:, body_idx]), "body wrist rotation"
    return (
        np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1)),
        "identity fallback",
    )


def retarget_positions(positions, scale, origin_xy, ground_z):
    p = np.asarray(positions, dtype=np.float64).copy()
    p[..., 0] = scale * (p[..., 0] - origin_xy[0])
    p[..., 1] = scale * (p[..., 1] - origin_xy[1])
    p[..., 2] = scale * (p[..., 2] - ground_z)
    return p


def extract_g1_command(
    data,
    *,
    scale_mode="conservative",
    robot_height=G1_HEIGHT_M,
    robot_wrist_span=G1_WRIST_SPAN_M,
    explicit_scale=None,
    ground_window=5,
    origin_window_seconds=1.0,
    torso_filter_window=5,
    left_q_offset=(1.0, 0.0, 0.0, 0.0),
    right_q_offset=(1.0, 0.0, 0.0, 0.0),
    shoulder_min=None,
    shoulder_max=None,
    torso_xy_tolerance=0.10,
    shoulder_height_tolerance=0.05,
    torso_heading_tolerance=0.15,
    locomotion_mode="auto",
):
    if locomotion_mode not in ("auto", "feet_locked"):
        raise ValueError("locomotion_mode must be auto or feet_locked")

    body = np.asarray(data["body"], dtype=np.float64)
    body_rot = data.get("body_rot")
    left_hand_rot = data.get("left_hand_rot")
    right_hand_rot = data.get("right_hand_rot")
    timestamps = np.asarray(data["timestamps"], dtype=np.float64)
    n = len(body)

    ground_z_reference, ground_z_trace = estimate_ground_reference(
        body, window=ground_window
    )
    human_height = estimate_human_height_proxy(
        body, ground_z_reference
    )
    human_wrist_span = estimate_human_wrist_span(body)

    height_scale = float(robot_height) / human_height
    wrist_span_scale = float(robot_wrist_span) / human_wrist_span

    if explicit_scale is not None:
        scale = float(explicit_scale)
        scale_description = "explicit"
        limiting_factor = "explicit"
    elif scale_mode == "conservative":
        scale = min(height_scale, wrist_span_scale)
        scale_description = "min(height, wrist_span)"
        limiting_factor = (
            "height" if height_scale <= wrist_span_scale else "wrist_span"
        )
    elif scale_mode == "height":
        scale = height_scale
        scale_description = "height"
        limiting_factor = "height"
    elif scale_mode == "wrist_span":
        scale = wrist_span_scale
        scale_description = "wrist_span"
        limiting_factor = "wrist_span"
    else:
        raise ValueError(f"Unknown scale_mode: {scale_mode}")

    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"Invalid morphology scale: {scale}")

    shoulder_mid = 0.5 * (
        body[:, LEFT_SHOULDER]
        + body[:, RIGHT_SHOULDER]
    )
    origin_xy = robust_initial_xy(
        shoulder_mid[:, :2],
        timestamps,
        seconds=origin_window_seconds,
    )

    left_wrist_pos_world = retarget_positions(
        body[:, LEFT_WRIST],
        scale,
        origin_xy,
        ground_z_reference,
    )
    right_wrist_pos_world = retarget_positions(
        body[:, RIGHT_WRIST],
        scale,
        origin_xy,
        ground_z_reference,
    )
    torso_center_world = retarget_positions(
        shoulder_mid,
        scale,
        origin_xy,
        ground_z_reference,
    )
    torso_center_xy_world = torso_center_world[:, :2].copy()

    if torso_filter_window > 1:
        torso_center_xy_world = moving_median_xy(
            torso_center_xy_world,
            torso_filter_window,
        )

    left_shoulder_height = scale * (
        body[:, LEFT_SHOULDER, 2] - ground_z_reference
    )
    right_shoulder_height = scale * (
        body[:, RIGHT_SHOULDER, 2] - ground_z_reference
    )

    if shoulder_min is not None:
        left_shoulder_height = np.maximum(
            left_shoulder_height, float(shoulder_min)
        )
        right_shoulder_height = np.maximum(
            right_shoulder_height, float(shoulder_min)
        )
    if shoulder_max is not None:
        left_shoulder_height = np.minimum(
            left_shoulder_height, float(shoulder_max)
        )
        right_shoulder_height = np.minimum(
            right_shoulder_height, float(shoulder_max)
        )

    torso_heading_world = torso_heading_from_shoulders(body)
    if torso_filter_window > 1:
        torso_heading_world = moving_median_1d(
            torso_heading_world,
            torso_filter_window,
        )

    left_q, left_source = _wrist_orientation(
        body_rot, left_hand_rot, LEFT_WRIST, n
    )
    right_q, right_source = _wrist_orientation(
        body_rot, right_hand_rot, RIGHT_WRIST, n
    )

    left_offset = normalize_quaternion(np.asarray(left_q_offset, dtype=np.float64))
    right_offset = normalize_quaternion(np.asarray(right_q_offset, dtype=np.float64))

    left_wrist_quat_world_wxyz = quat_multiply(
        left_q, np.broadcast_to(left_offset, left_q.shape)
    )
    right_wrist_quat_world_wxyz = quat_multiply(
        right_q, np.broadcast_to(right_offset, right_q.shape)
    )

    return {
        "timestamp": timestamps,
        "left_wrist_pos_world": left_wrist_pos_world,
        "left_wrist_quat_world_wxyz": left_wrist_quat_world_wxyz,
        "right_wrist_pos_world": right_wrist_pos_world,
        "right_wrist_quat_world_wxyz": right_wrist_quat_world_wxyz,
        "torso_center_xy_world": torso_center_xy_world,
        "left_shoulder_height": left_shoulder_height,
        "right_shoulder_height": right_shoulder_height,
        "torso_heading_world": torso_heading_world,
        "ground_z_world": np.zeros(n, dtype=np.float64),

        "source_ground_z_trace": ground_z_trace,
        "source_ground_z_reference": float(ground_z_reference),
        "source_origin_xy_world": np.asarray(origin_xy, dtype=np.float64),

        "torso_xy_tolerance": float(torso_xy_tolerance),
        "shoulder_height_tolerance": float(shoulder_height_tolerance),
        "torso_heading_tolerance": float(torso_heading_tolerance),
        "locomotion_mode": locomotion_mode,

        "scale": float(scale),
        "scale_mode": scale_description,
        "limiting_factor": limiting_factor,
        "height_scale": float(height_scale),
        "wrist_span_scale": float(wrist_span_scale),
        "robot_height": float(robot_height),
        "robot_wrist_span": float(robot_wrist_span),
        "human_height_proxy": float(human_height),
        "human_wrist_span": float(human_wrist_span),
        "left_orientation_source": left_source,
        "right_orientation_source": right_source,
    }


extract_controls = extract_g1_command


CONTROL_INTERFACE_YAML = """# Unitree G1 sparse whole-body command
interface:
  name: g1_sparse_whole_body_command
  version: "3.2"

coordinate_system:
  frame_id: g1_command_world
  handedness: right-handed
  axes: {x: forward, y: left, z: up}
  position_unit: m
  angle_unit: rad
  position_semantics: absolute_in_retargeted_command_world
  origin:
    xy: robust initial human shoulder-midpoint XY
    z: robust episode ground plane
  ground_z: 0.0

retargeting:
  scale:
    symbol: s
    default: conservative
    formula: >-
      min(robot_height / human_height_proxy,
          robot_wrist_span / human_wrist_span)
  transform:
    xy: p_cmd_xy = s * (p_human_xy - source_origin_xy)
    z: p_cmd_z = s * (p_human_z - source_ground_z_reference)
  applied_to:
    - left_wrist_pos_world
    - right_wrist_pos_world
    - torso_center_xy_world
    - left_shoulder_height
    - right_shoulder_height
    - all translational motion
  not_applied_to:
    - wrist orientations
    - torso_heading_world
    - time

whole_body_command:
  left_wrist:
    priority: strong
    position_key: left_wrist_pos_world
    orientation_key: left_wrist_quat_world_wxyz

  right_wrist:
    priority: strong
    position_key: right_wrist_pos_world
    orientation_key: right_wrist_quat_world_wxyz

  torso:
    priority: soft
    center_xy_key: torso_center_xy_world
    left_shoulder_height_key: left_shoulder_height
    right_shoulder_height_key: right_shoulder_height
    heading_key: torso_heading_world
    default_xy_tolerance_m: 0.10
    default_shoulder_height_tolerance_m: 0.05
    default_heading_tolerance_rad: 0.15

controller_constraints:
  locomotion_mode:
    values: [auto, feet_locked]
    default: auto
    training_target: false

explicitly_not_in_command:
  - base_position
  - base_xy
  - base_velocity
  - base_yaw
  - footstep_target
  - head_position
  - head_orientation
"""


def write_control_interface_yaml(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CONTROL_INTERFACE_YAML, encoding="utf-8")
    return path


def export_g1_command(path, command, episode_root):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "episode": str(episode_root),
        "interface_name": "g1_sparse_whole_body_command",
        "interface_version": CONTROL_INTERFACE_VERSION,
        "frame_id": "g1_command_world",
        "position_semantics": "absolute_in_retargeted_command_world",
        "ground_z_m": 0.0,
        "position_transform_xy": (
            "p_cmd_xy = scale * (p_human_xy - source_origin_xy_world)"
        ),
        "position_transform_z": (
            "p_cmd_z = scale * (p_human_z - source_ground_z_reference)"
        ),
        "source_origin_xy_world": command["source_origin_xy_world"].tolist(),
        "source_ground_z_reference": command["source_ground_z_reference"],
        "morphology_scale": command["scale"],
        "scale_mode": command["scale_mode"],
        "limiting_factor": command["limiting_factor"],
        "height_scale": command["height_scale"],
        "wrist_span_scale": command["wrist_span_scale"],
        "robot_height_m": command["robot_height"],
        "robot_wrist_span_m": command["robot_wrist_span"],
        "human_height_proxy_m": command["human_height_proxy"],
        "human_wrist_span_m": command["human_wrist_span"],
        "left_orientation_source": command["left_orientation_source"],
        "right_orientation_source": command["right_orientation_source"],
        "locomotion_mode": command["locomotion_mode"],
        "torso_xy_tolerance_m": command["torso_xy_tolerance"],
        "shoulder_height_tolerance_m": command["shoulder_height_tolerance"],
        "torso_heading_tolerance_rad": command["torso_heading_tolerance"],
    }

    np.savez_compressed(
        path,
        timestamp=command["timestamp"].astype(np.float32),
        left_wrist_pos_world=command["left_wrist_pos_world"].astype(np.float32),
        left_wrist_quat_world_wxyz=command["left_wrist_quat_world_wxyz"].astype(np.float32),
        right_wrist_pos_world=command["right_wrist_pos_world"].astype(np.float32),
        right_wrist_quat_world_wxyz=command["right_wrist_quat_world_wxyz"].astype(np.float32),
        torso_center_xy_world=command["torso_center_xy_world"].astype(np.float32),
        left_shoulder_height=command["left_shoulder_height"].astype(np.float32),
        right_shoulder_height=command["right_shoulder_height"].astype(np.float32),
        torso_heading_world=command["torso_heading_world"].astype(np.float32),
        ground_z_world=command["ground_z_world"].astype(np.float32),
        source_ground_z_trace=command["source_ground_z_trace"].astype(np.float32),
        source_origin_xy_world=command["source_origin_xy_world"].astype(np.float32),
        source_ground_z_reference=np.array(
            command["source_ground_z_reference"], dtype=np.float32
        ),
        metadata_json=np.array(
            json.dumps(metadata, indent=2, ensure_ascii=False)
        ),
    )
    return path


export_controls = export_g1_command


def convert_episode(episode_root, **kwargs):
    data = load_episode(episode_root)
    command = extract_g1_command(data, **kwargs)
    return data, command
