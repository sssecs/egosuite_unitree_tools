"""Geometry-based filtering for EgoSuite full-body episodes.

The filter deliberately operates at episode level.  Removing isolated frames
would create time discontinuities in motion trajectories; instead, a
per-frame mask is returned for auditing and an episode is rejected only when
the seated geometry is persistent.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


PELVIS = 0
LEFT_HIP, RIGHT_HIP = 1, 2
LEFT_KNEE, RIGHT_KNEE = 4, 5
LEFT_ANKLE, RIGHT_ANKLE = 7, 8
LEFT_FOOT, RIGHT_FOOT = 10, 11


@dataclass(frozen=True)
class SittingFilterConfig:
    """Dimensionless thresholds for persistent seated-pose detection."""

    max_pelvis_height: float = 0.85
    max_thigh_verticality: float = 0.55
    min_shin_verticality: float = 0.60
    min_knee_ground_clearance: float = 0.10
    persistence_seconds: float = 0.50
    persistence_fraction: float = 0.80
    reject_fraction: float = 0.60


def _safe_norm(x: np.ndarray) -> np.ndarray:
    return np.maximum(np.linalg.norm(x, axis=-1), 1e-8)


def _leg_lengths(body: np.ndarray) -> tuple[float, float]:
    lengths = []
    for hip, knee, ankle in (
        (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
        (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
    ):
        thigh = np.nanmedian(_safe_norm(body[:, hip] - body[:, knee]))
        shin = np.nanmedian(_safe_norm(body[:, knee] - body[:, ankle]))
        lengths.append(float(thigh + shin))
    if not np.all(np.isfinite(lengths)) or min(lengths) <= 1e-6:
        raise RuntimeError("Cannot estimate valid human leg lengths.")
    return lengths[0], lengths[1]


def _ground_trace(body: np.ndarray) -> np.ndarray:
    z = np.stack(
        [
            body[:, LEFT_FOOT, 2],
            body[:, RIGHT_FOOT, 2],
            body[:, LEFT_ANKLE, 2],
            body[:, RIGHT_ANKLE, 2],
        ],
        axis=1,
    )
    ground = np.nanmin(z, axis=1)
    valid = np.flatnonzero(np.isfinite(ground))
    if len(valid) == 0:
        raise RuntimeError("Cannot estimate ground from feet/ankles.")
    missing = np.flatnonzero(~np.isfinite(ground))
    if len(missing):
        ground[missing] = np.interp(missing, valid, ground[valid])
    return ground


def _persistent_mask(mask: np.ndarray, window: int, fraction: float) -> np.ndarray:
    window = max(1, min(int(window), len(mask)))
    if window == 1:
        return mask.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    left = window // 2
    right = window - 1 - left
    padded = np.pad(mask.astype(np.float64), (left, right), mode="edge")
    score = np.convolve(padded, kernel, mode="valid")
    return score >= float(fraction)


def assess_sitting_episode(
    body: np.ndarray,
    timestamps: np.ndarray | None = None,
    *,
    fps: float = 30.0,
    ground_z: np.ndarray | None = None,
    config: SittingFilterConfig | None = None,
) -> dict:
    """Return an auditable seated-geometry assessment for one episode.

    Sitting is distinguished from kneeling by knee clearance and from most
    squats by the combination of near-horizontal thighs and near-vertical
    shins.  Temporal persistence prevents brief low postures from rejecting an
    otherwise standing episode.
    """
    cfg = config or SittingFilterConfig()
    body = np.asarray(body, dtype=np.float64)
    if body.ndim != 3 or body.shape[1:] != (22, 3) or len(body) == 0:
        raise ValueError(f"body must have shape (T, 22, 3), got {body.shape}")

    if timestamps is not None and len(timestamps) > 1:
        dt = np.diff(np.asarray(timestamps, dtype=np.float64))
        dt = dt[np.isfinite(dt) & (dt > 1e-6)]
        effective_fps = 1.0 / float(np.median(dt)) if len(dt) else float(fps)
    else:
        effective_fps = float(fps)

    left_leg, right_leg = _leg_lengths(body)
    length = 0.5 * (left_leg + right_leg)
    ground = _ground_trace(body) if ground_z is None else np.asarray(ground_z, float)
    if ground.shape != (len(body),):
        raise ValueError("ground_z must contain one value per frame")

    pelvis_height = (body[:, PELVIS, 2] - ground) / length
    thigh_verticality = np.mean(
        [
            np.abs(body[:, hip, 2] - body[:, knee, 2])
            / _safe_norm(body[:, hip] - body[:, knee])
            for hip, knee in ((LEFT_HIP, LEFT_KNEE), (RIGHT_HIP, RIGHT_KNEE))
        ],
        axis=0,
    )
    shin_verticality = np.mean(
        [
            np.abs(body[:, knee, 2] - body[:, ankle, 2])
            / _safe_norm(body[:, knee] - body[:, ankle])
            for knee, ankle in ((LEFT_KNEE, LEFT_ANKLE), (RIGHT_KNEE, RIGHT_ANKLE))
        ],
        axis=0,
    )
    knee_clearance = np.minimum(
        body[:, LEFT_KNEE, 2] - ground,
        body[:, RIGHT_KNEE, 2] - ground,
    ) / length

    finite = np.all(
        np.stack(
            [pelvis_height, thigh_verticality, shin_verticality, knee_clearance]
        ),
        axis=0,
    )
    candidate = (
        finite
        & (pelvis_height < cfg.max_pelvis_height)
        & (thigh_verticality < cfg.max_thigh_verticality)
        & (shin_verticality > cfg.min_shin_verticality)
        & (knee_clearance > cfg.min_knee_ground_clearance)
    )
    window = max(1, int(round(cfg.persistence_seconds * effective_fps)))
    seated_mask = _persistent_mask(candidate, window, cfg.persistence_fraction)
    seated_fraction = float(np.mean(seated_mask))

    return {
        "reject": seated_fraction >= cfg.reject_fraction,
        "reason": "persistent_seated_pose" if seated_fraction >= cfg.reject_fraction else "keep",
        "seated_fraction": seated_fraction,
        "candidate_fraction": float(np.mean(candidate)),
        "seated_mask": seated_mask,
        "candidate_mask": candidate,
        "characteristic_leg_length_m": float(length),
        "median_pelvis_height_norm": float(np.nanmedian(pelvis_height)),
        "median_thigh_verticality": float(np.nanmedian(thigh_verticality)),
        "median_shin_verticality": float(np.nanmedian(shin_verticality)),
        "median_knee_clearance_norm": float(np.nanmedian(knee_clearance)),
        "config": asdict(cfg),
    }
