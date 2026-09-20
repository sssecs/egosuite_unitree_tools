"""17-D cross-embodiment descriptor defined by the project design document."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PELVIS = 0
LEFT_HIP, RIGHT_HIP = 1, 2
LEFT_KNEE, RIGHT_KNEE = 4, 5
LEFT_ANKLE, RIGHT_ANKLE = 7, 8
LEFT_FOOT, RIGHT_FOOT = 10, 11
LEFT_SHOULDER, RIGHT_SHOULDER = 16, 17
GRAVITY = 9.81
STYLE_DESCRIPTOR_VERSION = "1.0-17d"

# V1 order from cross_embodiment_descriptor_design.md, section 4.
STYLE_DESCRIPTOR_NAMES = (
    "pelvis_height_neutral_norm",
    "effective_leg_length_left",
    "effective_leg_length_right",
    "knee_ground_distance_left",
    "knee_ground_distance_right",
    "torso_pitch",
    "shoulder_mid_rel_pelvis_forward",
    "shoulder_mid_rel_pelvis_left",
    "shoulder_pelvis_yaw_difference",
    "left_foot_rel_pelvis_forward",
    "left_foot_rel_pelvis_left",
    "right_foot_rel_pelvis_forward",
    "right_foot_rel_pelvis_left",
    "foot_pseudo_contact_left",
    "foot_pseudo_contact_right",
    "knee_pseudo_contact_left",
    "knee_pseudo_contact_right",
)


def _norm(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x, axis=-1)


def _gradient(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    if len(x) < 2:
        return np.zeros_like(x, dtype=np.float64)
    safe_t = np.asarray(t, dtype=np.float64).copy()
    if np.any(~np.isfinite(safe_t)) or np.any(np.diff(safe_t) <= 0):
        safe_t = np.arange(len(x), dtype=np.float64) / 30.0
    return np.gradient(x, safe_t, axis=0, edge_order=1)


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) <= 2:
        return x.copy()
    window = min(int(window), len(x) if len(x) % 2 else len(x) - 1)
    window = max(1, window | 1)
    pad = window // 2
    padded = np.pad(x, ((pad, pad),) + ((0, 0),) * (x.ndim - 1), mode="edge")
    views = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0)
    return np.nanmedian(views, axis=-1)


def _leg_lengths(body: np.ndarray) -> np.ndarray:
    lengths = []
    for hip, knee, ankle in (
        (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
        (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
    ):
        lengths.append(
            np.nanmedian(_norm(body[:, hip] - body[:, knee]))
            + np.nanmedian(_norm(body[:, knee] - body[:, ankle]))
        )
    lengths = np.asarray(lengths, dtype=np.float64)
    if np.any(~np.isfinite(lengths)) or np.any(lengths <= 1e-6):
        raise RuntimeError("Cannot estimate valid characteristic leg lengths.")
    return lengths


def _ground(body: np.ndarray) -> np.ndarray:
    feet = np.stack([body[:, LEFT_FOOT, 2], body[:, RIGHT_FOOT, 2]], axis=1)
    ankles = np.stack([body[:, LEFT_ANKLE, 2], body[:, RIGHT_ANKLE, 2]], axis=1)
    ground = np.nanmin(feet, axis=1)
    missing = ~np.isfinite(ground)
    ground[missing] = np.nanmin(ankles[missing], axis=1)
    valid = np.flatnonzero(np.isfinite(ground))
    if len(valid) == 0:
        raise RuntimeError("Cannot estimate ground from feet/ankles.")
    bad = np.flatnonzero(~np.isfinite(ground))
    if len(bad):
        ground[bad] = np.interp(bad, valid, ground[valid])
    return _smooth(ground, 5)


def _heading_axes(left_point: np.ndarray, right_point: np.ndarray):
    """Return forward/left axes and heading from a bilateral landmark pair."""
    left = left_point[:, :2] - right_point[:, :2]
    left /= np.maximum(_norm(left)[:, None], 1e-8)
    forward = np.stack([left[:, 1], -left[:, 0]], axis=1)
    heading = np.unwrap(np.arctan2(forward[:, 1], forward[:, 0]))
    return forward, left, heading


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _neutral_pelvis_height(pelvis_height: np.ndarray) -> float:
    """Estimate neutral standing height from the episode's upper posture band."""
    valid = pelvis_height[np.isfinite(pelvis_height) & (pelvis_height > 1e-6)]
    if len(valid) == 0:
        raise RuntimeError("Cannot estimate neutral pelvis height.")
    neutral = float(np.percentile(valid, 95.0))
    if neutral <= 1e-6:
        raise RuntimeError("Neutral pelvis height is invalid.")
    return neutral


def extract_human_style_descriptor(
    body: np.ndarray,
    timestamps: np.ndarray,
    *,
    ground_z: np.ndarray | None = None,
    neutral_pelvis_height: float | None = None,
    smoothing_window: int = 5,
    contact_height_sigma: float = 0.06,
    contact_velocity_sigma: float = 0.12,
) -> dict:
    """Extract the design-specified V1 17-D human descriptor.

    Horizontal relations use the shoulder-heading frame. Distances other than
    pelvis height are normalized by mean leg length. Pelvis height is divided
    by an episode neutral-height estimate (95th percentile) unless explicitly
    supplied. Contacts use the same dimensionless height/velocity RBF formula
    intended for both human landmarks and robot FK landmarks.
    """
    body = np.asarray(body, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if body.ndim != 3 or body.shape[1:] != (22, 3) or len(body) != len(timestamps):
        raise ValueError("Expected body (T,22,3) and timestamps (T,)")
    if len(body) == 0:
        raise ValueError("Cannot describe an empty episode")
    if smoothing_window < 1:
        raise ValueError("smoothing_window must be positive")

    leg_lengths = _leg_lengths(body)
    length = float(np.mean(leg_lengths))
    ground = _ground(body) if ground_z is None else np.asarray(ground_z, float)
    if ground.shape != (len(body),):
        raise ValueError("ground_z must contain one value per frame")

    pelvis = body[:, PELVIS]
    shoulder_mid = 0.5 * (body[:, LEFT_SHOULDER] + body[:, RIGHT_SHOULDER])
    forward, left, shoulder_heading = _heading_axes(
        body[:, LEFT_SHOULDER], body[:, RIGHT_SHOULDER]
    )
    _, _, pelvis_heading = _heading_axes(body[:, LEFT_HIP], body[:, RIGHT_HIP])

    def local_xy(xy: np.ndarray) -> np.ndarray:
        return np.stack(
            [np.sum(xy * forward, axis=-1), np.sum(xy * left, axis=-1)],
            axis=-1,
        )

    pelvis_height_m = pelvis[:, 2] - ground
    neutral_height = (
        _neutral_pelvis_height(pelvis_height_m)
        if neutral_pelvis_height is None else float(neutral_pelvis_height)
    )
    if not np.isfinite(neutral_height) or neutral_height <= 1e-6:
        raise ValueError("neutral_pelvis_height must be finite and positive")
    pelvis_height_norm = pelvis_height_m / neutral_height

    leg_extension = np.stack(
        [
            _norm(body[:, LEFT_HIP] - body[:, LEFT_ANKLE]) / leg_lengths[0],
            _norm(body[:, RIGHT_HIP] - body[:, RIGHT_ANKLE]) / leg_lengths[1],
        ],
        axis=1,
    )
    knee_ground = np.stack(
        [
            np.maximum(body[:, LEFT_KNEE, 2] - ground, 0.0) / length,
            np.maximum(body[:, RIGHT_KNEE, 2] - ground, 0.0) / length,
        ],
        axis=1,
    )

    torso_up = shoulder_mid - pelvis
    torso_up /= np.maximum(_norm(torso_up)[:, None], 1e-8)
    torso_forward = np.sum(torso_up[:, :2] * forward, axis=-1)
    torso_pitch = np.arctan2(torso_forward, torso_up[:, 2])

    shoulder_rel = local_xy((shoulder_mid - pelvis)[:, :2]) / length
    yaw_difference = _wrap_angle(shoulder_heading - pelvis_heading)

    foot_relative = []
    for foot_index in (LEFT_FOOT, RIGHT_FOOT):
        foot_relative.append(local_xy((body[:, foot_index] - pelvis)[:, :2]) / length)

    velocity_scale = np.sqrt(GRAVITY * length)
    landmark_velocity = {
        index: _norm(_gradient(body[:, index], timestamps)) / velocity_scale
        for index in (LEFT_FOOT, RIGHT_FOOT, LEFT_KNEE, RIGHT_KNEE)
    }

    def pseudo_contact(index: int) -> np.ndarray:
        height = np.maximum(body[:, index, 2] - ground, 0.0) / length
        return np.exp(
            -((height / float(contact_height_sigma)) ** 2)
            -((landmark_velocity[index] / float(contact_velocity_sigma)) ** 2)
        )

    contacts = np.column_stack(
        [
            pseudo_contact(LEFT_FOOT),
            pseudo_contact(RIGHT_FOOT),
            pseudo_contact(LEFT_KNEE),
            pseudo_contact(RIGHT_KNEE),
        ]
    )

    descriptor = np.column_stack(
        [
            pelvis_height_norm,
            leg_extension,
            knee_ground,
            torso_pitch,
            shoulder_rel,
            yaw_difference,
            foot_relative[0],
            foot_relative[1],
            contacts,
        ]
    )
    descriptor = _smooth(descriptor, smoothing_window)
    valid_mask = np.all(np.isfinite(descriptor), axis=1)
    if descriptor.shape[1] != len(STYLE_DESCRIPTOR_NAMES):
        raise AssertionError("Style descriptor schema mismatch")

    return {
        "descriptor": descriptor,
        "timestamp": timestamps,
        "names": STYLE_DESCRIPTOR_NAMES,
        "valid_mask": valid_mask,
        "characteristic_leg_length_m": length,
        "left_leg_length_m": float(leg_lengths[0]),
        "right_leg_length_m": float(leg_lengths[1]),
        "neutral_pelvis_height_m": neutral_height,
        "contact_height_sigma": float(contact_height_sigma),
        "contact_velocity_sigma": float(contact_velocity_sigma),
        "version": STYLE_DESCRIPTOR_VERSION,
    }


def export_style_descriptor(path: str | Path, style: dict, episode_root: str | Path) -> Path:
    """Write a versioned 17-D descriptor sidecar independent of commands."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "episode": str(episode_root),
        "descriptor_version": style["version"],
        "descriptor_dimension": len(style["names"]),
        "descriptor_names": list(style["names"]),
        "design_document": "cross_embodiment_descriptor_design.md",
        "characteristic_leg_length_m": style["characteristic_leg_length_m"],
        "left_leg_length_m": style["left_leg_length_m"],
        "right_leg_length_m": style["right_leg_length_m"],
        "neutral_pelvis_height_m": style["neutral_pelvis_height_m"],
        "neutral_pelvis_height_estimator": "episode pelvis height 95th percentile",
        "contact_height_sigma": style["contact_height_sigma"],
        "contact_velocity_sigma": style["contact_velocity_sigma"],
        "normalization": {
            "pelvis_height": "neutral_pelvis_height",
            "other_distances": "mean_characteristic_leg_length",
            "contact_velocity": "sqrt(g * mean_characteristic_leg_length)",
            "horizontal_frame": "instantaneous_shoulder_heading",
        },
    }
    np.savez_compressed(
        path,
        timestamp=np.asarray(style["timestamp"], dtype=np.float32),
        style_descriptor=np.asarray(style["descriptor"], dtype=np.float32),
        style_valid_mask=np.asarray(style["valid_mask"], dtype=bool),
        style_descriptor_names=np.asarray(style["names"]),
        metadata_json=np.array(json.dumps(metadata, indent=2, ensure_ascii=False)),
    )
    return path
