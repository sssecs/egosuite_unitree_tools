#!/usr/bin/env python3
"""
Robust interactive player for G1 sparse command NPZ.

This version deliberately does NOT use matplotlib.animation.FuncAnimation.
It uses the GUI canvas timer directly, so playback frame progression is
explicit and cannot stop because of an implicit animation frame generator.

Keyboard:
    SPACE       pause / resume
    LEFT        previous frame
    RIGHT       next frame
    UP          faster
    DOWN        slower
    R           reset to frame 0
    Q / ESC     quit

Useful diagnostics:
    --start-frame 15
    --no-human
    --style path/to/episode.style.npz
    --debug

If rendering fails at a particular frame, the full traceback and frame index
are printed to the terminal instead of silently stopping the animation.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cross_embodiment_style_descriptor import (
    STYLE_DESCRIPTOR_NAMES,
    STYLE_DESCRIPTOR_VERSION,
)
from egosuite_to_g1_command import (
    LEFT_WRIST,
    RIGHT_WRIST,
    load_episode,
)


BODY_EDGES = [
    (0, 1), (0, 2), (0, 3),
    (3, 6), (6, 9), (9, 12), (12, 15),
    (12, 13), (13, 16), (16, 18), (18, 20),
    (12, 14), (14, 17), (17, 19), (19, 21),
    (1, 4), (4, 7), (7, 10),
    (2, 5), (5, 8), (8, 11),
]

REQUIRED_KEYS = [
    "timestamp",
    "left_wrist_pos_world",
    "left_wrist_quat_world_wxyz",
    "right_wrist_pos_world",
    "right_wrist_quat_world_wxyz",
    "torso_center_xy_world",
    "left_shoulder_height",
    "right_shoulder_height",
    "torso_heading_world",
    "ground_z_world",
]

STYLE_REQUIRED_KEYS = [
    "style_descriptor",
    "style_descriptor_names",
]

STYLE_SHORT_NAMES = {
    "pelvis_height_neutral_norm": "pelvis height/neutral",
    "effective_leg_length_left": "leg extension L",
    "effective_leg_length_right": "leg extension R",
    "knee_ground_distance_left": "knee-ground L",
    "knee_ground_distance_right": "knee-ground R",
    "torso_pitch": "torso pitch",
    "shoulder_mid_rel_pelvis_forward": "shoulder-pelvis/fwd",
    "shoulder_mid_rel_pelvis_left": "shoulder-pelvis/left",
    "shoulder_pelvis_yaw_difference": "shoulder-pelvis yaw",
    "left_foot_rel_pelvis_forward": "foot L rel/fwd",
    "left_foot_rel_pelvis_left": "foot L rel/left",
    "right_foot_rel_pelvis_forward": "foot R rel/fwd",
    "right_foot_rel_pelvis_left": "foot R rel/left",
    "foot_pseudo_contact_left": "foot contact L",
    "foot_pseudo_contact_right": "foot contact R",
    "knee_pseudo_contact_left": "knee contact L",
    "knee_pseudo_contact_right": "knee contact R",
}


def load_command(path):
    path = Path(path).expanduser().resolve()

    with np.load(path, allow_pickle=False) as f:
        missing = [
            key for key in REQUIRED_KEYS
            if key not in f.files
        ]
        if missing:
            raise RuntimeError(
                "Missing required NPZ keys:\n  "
                + "\n  ".join(missing)
            )

        cmd = {
            key: np.asarray(f[key])
            for key in REQUIRED_KEYS
        }

        metadata = {}
        if "metadata_json" in f.files:
            raw = f["metadata_json"]
            if raw.ndim == 0:
                raw = raw.item()
            try:
                metadata = json.loads(str(raw))
            except Exception:
                metadata = {}

    n = len(cmd["timestamp"])

    if n == 0:
        raise RuntimeError("Command contains zero frames.")

    for key, value in cmd.items():
        if value.shape[0] != n:
            raise RuntimeError(
                f"Length mismatch: {key} has "
                f"{value.shape[0]} frames, timestamp has {n}."
            )

    expected_shapes = {
        "left_wrist_pos_world": (n, 3),
        "left_wrist_quat_world_wxyz": (n, 4),
        "right_wrist_pos_world": (n, 3),
        "right_wrist_quat_world_wxyz": (n, 4),
        "torso_center_xy_world": (n, 2),
        "left_shoulder_height": (n,),
        "right_shoulder_height": (n,),
        "torso_heading_world": (n,),
        "ground_z_world": (n,),
    }

    for key, shape in expected_shapes.items():
        if cmd[key].shape != shape:
            raise RuntimeError(
                f"Shape mismatch: {key} has shape {cmd[key].shape}, "
                f"expected {shape}."
            )

    return path, cmd, metadata


def load_style_descriptor(path, fallback_timestamps=None):
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as f:
        missing = [key for key in STYLE_REQUIRED_KEYS if key not in f.files]
        if missing:
            raise RuntimeError(
                "Missing required style NPZ keys:\n  " + "\n  ".join(missing)
            )
        descriptor = np.asarray(f["style_descriptor"], dtype=np.float64)
        names = tuple(str(value) for value in np.asarray(f["style_descriptor_names"]))
        if "timestamp" in f.files:
            timestamps = np.asarray(f["timestamp"], dtype=np.float64)
        elif fallback_timestamps is not None and len(fallback_timestamps) == len(descriptor):
            timestamps = np.asarray(fallback_timestamps, dtype=np.float64)
        else:
            timestamps = np.arange(len(descriptor), dtype=np.float64)
        valid_mask = (
            np.asarray(f["style_valid_mask"], dtype=bool)
            if "style_valid_mask" in f.files else np.all(np.isfinite(descriptor), axis=1)
        )
        metadata = {}
        if "metadata_json" in f.files:
            raw = f["metadata_json"]
            if raw.ndim == 0:
                raw = raw.item()
            try:
                metadata = json.loads(str(raw))
            except Exception:
                metadata = {}

    if descriptor.ndim != 2 or descriptor.shape[1] != len(names):
        raise RuntimeError(
            f"Style descriptor shape {descriptor.shape} is incompatible with "
            f"{len(names)} names."
        )
    if len(descriptor) == 0 or timestamps.shape != (len(descriptor),):
        raise RuntimeError("Style descriptor has invalid or empty timestamps.")
    if valid_mask.shape != (len(descriptor),):
        raise RuntimeError("style_valid_mask must contain one value per frame.")
    if len(set(names)) != len(names):
        raise RuntimeError("Style descriptor names must be unique.")
    if names != STYLE_DESCRIPTOR_NAMES:
        version = metadata.get("descriptor_version", "unknown")
        raise RuntimeError(
            f"Unsupported style schema v{version} with {len(names)} dimensions; "
            f"expected v{STYLE_DESCRIPTOR_VERSION} with "
            f"{len(STYLE_DESCRIPTOR_NAMES)} dimensions. Regenerate this sidecar "
            "with egodemo_to_g1_commands.py."
        )
    return path, {
        "descriptor": descriptor,
        "names": names,
        "timestamp": timestamps,
        "valid_mask": valid_mask,
        "metadata": metadata,
    }


def quat_to_matrix(q):
    q = np.asarray(q, dtype=np.float64)

    if q.shape != (4,):
        return np.eye(3)

    if not np.all(np.isfinite(q)):
        return np.eye(3)

    norm = np.linalg.norm(q)

    if norm < 1e-12:
        return np.eye(3)

    w, x, y, z = q / norm

    return np.array([
        [
            1 - 2 * (y*y + z*z),
            2 * (x*y - z*w),
            2 * (x*z + y*w),
        ],
        [
            2 * (x*y + z*w),
            1 - 2 * (x*x + z*z),
            2 * (y*z - x*w),
        ],
        [
            2 * (x*z - y*w),
            2 * (y*z + x*w),
            1 - 2 * (x*x + y*y),
        ],
    ], dtype=np.float64)


def transform_body_to_command_world(body, metadata):
    body = np.asarray(
        body,
        dtype=np.float64,
    ).copy()

    scale = float(
        metadata.get(
            "morphology_scale",
            1.0,
        )
    )

    origin = np.asarray(
        metadata.get(
            "source_origin_xy_world",
            [0.0, 0.0],
        ),
        dtype=np.float64,
    )

    ground = float(
        metadata.get(
            "source_ground_z_reference",
            0.0,
        )
    )

    body[..., 0] = (
        scale * (body[..., 0] - origin[0])
    )
    body[..., 1] = (
        scale * (body[..., 1] - origin[1])
    )
    body[..., 2] = (
        scale * (body[..., 2] - ground)
    )

    return body


def nearest_indices(source_t, target_t):
    source_t = np.asarray(
        source_t,
        dtype=np.float64,
    )
    target_t = np.asarray(
        target_t,
        dtype=np.float64,
    )

    ids = np.searchsorted(
        source_t,
        target_t,
    )
    ids = np.clip(
        ids,
        0,
        len(source_t) - 1,
    )

    left = np.maximum(
        ids - 1,
        0,
    )

    choose_left = (
        np.abs(
            target_t - source_t[left]
        )
        <
        np.abs(
            source_t[ids] - target_t
        )
    )

    ids[choose_left] = left[choose_left]
    return ids


def verify_wrist_scaling(raw_body, cmd, metadata, source_indices):
    expected = transform_body_to_command_world(
        raw_body,
        metadata,
    )

    left_expected = expected[
        source_indices,
        LEFT_WRIST,
    ]
    right_expected = expected[
        source_indices,
        RIGHT_WRIST,
    ]

    left_actual = np.asarray(
        cmd["left_wrist_pos_world"],
        dtype=np.float64,
    )
    right_actual = np.asarray(
        cmd["right_wrist_pos_world"],
        dtype=np.float64,
    )

    return {
        "left_error": (
            left_actual - left_expected
        ),
        "right_error": (
            right_actual - right_expected
        ),
    }


class CommandPlayer:
    def __init__(
        self,
        command_path,
        cmd,
        metadata,
        human_body=None,
        human_indices=None,
        verification=None,
        style=None,
        style_indices=None,
        fps=None,
        start_frame=0,
        trail_frames=45,
        debug=False,
    ):
        self.command_path = command_path
        self.cmd = cmd
        self.metadata = metadata

        self.human_body = human_body
        self.human_indices = human_indices
        self.verification = verification
        self.style = style
        self.style_indices = (
            np.asarray(style_indices, dtype=np.int64)
            if style_indices is not None else (
                nearest_indices(style["timestamp"], cmd["timestamp"])
                if style is not None else None
            )
        )

        self.n = len(
            cmd["timestamp"]
        )

        self.i = int(
            np.clip(
                start_frame,
                0,
                self.n - 1,
            )
        )

        self.paused = False
        self.debug = bool(debug)
        self.trail_frames = max(
            0,
            int(trail_frames),
        )

        self.speed = 1.0

        timestamps = np.asarray(
            cmd["timestamp"],
            dtype=np.float64,
        )

        if fps is not None:
            self.base_fps = float(fps)
        elif len(timestamps) > 1:
            dt = np.diff(timestamps)
            dt = dt[
                np.isfinite(dt)
                & (dt > 1e-6)
            ]
            self.base_fps = (
                1.0 / float(np.median(dt))
                if len(dt)
                else 30.0
            )
        else:
            self.base_fps = 30.0

        if self.style is None:
            self.fig = plt.figure(figsize=(11, 8))
            self.ax = self.fig.add_subplot(111, projection="3d")
            self.style_axes = []
            self.style_value_ax = None
            self.style_cursors = []
            self.style_value_text = None
        else:
            self.fig = plt.figure(figsize=(18, 10))
            grid = self.fig.add_gridspec(
                3,
                3,
                width_ratios=(1.75, 1.10, 0.95),
                hspace=0.34,
                wspace=0.28,
            )
            self.ax = self.fig.add_subplot(grid[:, 0], projection="3d")
            self.style_axes = [
                self.fig.add_subplot(grid[0, 1]),
                self.fig.add_subplot(grid[1, 1]),
                self.fig.add_subplot(grid[2, 1]),
            ]
            self.style_value_ax = self.fig.add_subplot(grid[:, 2])
            self.setup_style_display()

        self.fig.canvas.mpl_connect(
            "key_press_event",
            self.on_key,
        )

        self.fig.canvas.mpl_connect(
            "close_event",
            self.on_close,
        )

        self.closed = False

        self.compute_fixed_limits()

        # Use the GUI timer directly. This is intentionally more explicit
        # than FuncAnimation for interactive inspection.
        self.timer = self.fig.canvas.new_timer(
            interval=self.interval_ms()
        )
        self.timer.add_callback(
            self.on_timer
        )
        self.timer.start()

        if self.debug:
            print(
                "[player] GUI timer started; "
                "player/timer will be held by strong references.",
                flush=True,
            )

        self.render_frame()

    def style_column(self, name):
        try:
            index = self.style["names"].index(name)
        except ValueError:
            return None
        return self.style["descriptor"][:, index]

    def setup_style_display(self):
        """Create persistent descriptor traces; only cursors/text change per frame."""
        timestamps = self.style["timestamp"]
        groups = [
            (
                "Posture",
                [
                    ("pelvis_height_neutral_norm", "pelvis/neutral"),
                    ("effective_leg_length_left", "leg ext L"),
                    ("effective_leg_length_right", "leg ext R"),
                    ("knee_ground_distance_left", "knee-ground L"),
                    ("knee_ground_distance_right", "knee-ground R"),
                    ("torso_pitch", "torso pitch"),
                ],
            ),
            (
                "Heading-local relative placement",
                [
                    ("shoulder_mid_rel_pelvis_forward", "shoulder/fwd"),
                    ("shoulder_mid_rel_pelvis_left", "shoulder/left"),
                    ("shoulder_pelvis_yaw_difference", "yaw difference"),
                    ("left_foot_rel_pelvis_forward", "foot L/fwd"),
                    ("left_foot_rel_pelvis_left", "foot L/left"),
                    ("right_foot_rel_pelvis_forward", "foot R/fwd"),
                    ("right_foot_rel_pelvis_left", "foot R/left"),
                ],
            ),
            (
                "Foot / knee pseudo-contact",
                [
                    ("foot_pseudo_contact_left", "foot L"),
                    ("foot_pseudo_contact_right", "foot R"),
                    ("knee_pseudo_contact_left", "knee L"),
                    ("knee_pseudo_contact_right", "knee R"),
                ],
            ),
        ]
        self.style_cursors = []
        for axis, (title, signals) in zip(self.style_axes, groups):
            for name, label in signals:
                values = self.style_column(name)
                if values is not None:
                    axis.plot(timestamps, values, linewidth=1.15, label=label)
            axis.axhline(0.0, color="black", linewidth=0.6, alpha=0.25)
            cursor = axis.axvline(timestamps[0], color="black", linewidth=1.5)
            self.style_cursors.append(cursor)
            axis.set_title(title, fontsize=10)
            axis.set_xlabel("time [s]")
            axis.grid(alpha=0.20)
            axis.legend(loc="upper right", fontsize=7, ncol=2)

        self.style_value_ax.set_axis_off()
        self.style_value_ax.set_title(
            f"Current {len(self.style['names'])}-D descriptor", fontsize=11
        )
        self.style_value_text = self.style_value_ax.text(
            0.0,
            0.98,
            "",
            transform=self.style_value_ax.transAxes,
            va="top",
            family="monospace",
            fontsize=8.3,
        )

    def update_style_display(self, command_index):
        if self.style is None:
            return
        style_index = int(self.style_indices[command_index])
        time = float(self.style["timestamp"][style_index])
        for cursor in self.style_cursors:
            cursor.set_xdata([time, time])

        values = self.style["descriptor"][style_index]
        valid = bool(self.style["valid_mask"][style_index])
        section_starts = {
            0: "POSTURE",
            6: "RELATIVE GEOMETRY",
            13: "PSEUDO-CONTACT",
        }
        lines = [
            f"style frame {style_index + 1}/{len(self.style['descriptor'])}",
            f"t={time:.3f}s  valid={valid}",
            "",
        ]
        for index, (name, value) in enumerate(zip(self.style["names"], values)):
            if index in section_starts:
                if index:
                    lines.append("")
                lines.append(section_starts[index])
            short = STYLE_SHORT_NAMES.get(name, name[:20])
            lines.append(f"{short:<20} {value:>8.4f}")
        self.style_value_text.set_text("\n".join(lines))

    def interval_ms(self):
        return max(
            1,
            int(
                round(
                    1000.0
                    / max(
                        self.base_fps
                        * self.speed,
                        1e-6,
                    )
                )
            ),
        )

    def update_timer_interval(self):
        self.timer.interval = (
            self.interval_ms()
        )

    def compute_fixed_limits(self):
        z = 0.5 * (
            np.asarray(
                self.cmd[
                    "left_shoulder_height"
                ],
                dtype=np.float64,
            )
            +
            np.asarray(
                self.cmd[
                    "right_shoulder_height"
                ],
                dtype=np.float64,
            )
        )

        torso = np.column_stack([
            self.cmd[
                "torso_center_xy_world"
            ],
            z,
        ])

        point_sets = [
            np.asarray(
                self.cmd[
                    "left_wrist_pos_world"
                ],
                dtype=np.float64,
            ),
            np.asarray(
                self.cmd[
                    "right_wrist_pos_world"
                ],
                dtype=np.float64,
            ),
            torso,
        ]

        if self.human_body is not None:
            point_sets.append(
                self.human_body.reshape(
                    -1,
                    3,
                )
            )

        points = np.concatenate(
            point_sets,
            axis=0,
        )

        valid = np.all(
            np.isfinite(points),
            axis=1,
        )
        points = points[valid]

        if len(points) == 0:
            self.xlim = (-1, 1)
            self.ylim = (-1, 1)
            self.zlim = (-0.05, 1.8)
            return

        lo = np.min(
            points,
            axis=0,
        )
        hi = np.max(
            points,
            axis=0,
        )

        lo[2] = min(
            lo[2],
            0.0,
        )
        hi[2] = max(
            hi[2],
            0.0,
        )

        # Equal XY scaling.
        cx = 0.5 * (
            lo[0] + hi[0]
        )
        cy = 0.5 * (
            lo[1] + hi[1]
        )

        span_xy = max(
            hi[0] - lo[0],
            hi[1] - lo[1],
            0.8,
        )

        half = (
            0.5 * span_xy
            + 0.15
        )

        self.xlim = (
            cx - half,
            cx + half,
        )
        self.ylim = (
            cy - half,
            cy + half,
        )
        self.zlim = (
            min(
                -0.05,
                lo[2] - 0.05,
            ),
            hi[2] + 0.20,
        )

    def draw_wrist_axes(
        self,
        position,
        quat,
        label,
    ):
        position = np.asarray(
            position,
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(position)
        ):
            return

        self.ax.scatter(
            [position[0]],
            [position[1]],
            [position[2]],
            s=55,
            label=label,
        )

        R = quat_to_matrix(quat)
        axis_length = 0.08
        styles = [
            "-",
            "--",
            ":",
        ]

        for axis in range(3):
            end = (
                position
                + axis_length
                * R[:, axis]
            )

            self.ax.plot(
                [
                    position[0],
                    end[0],
                ],
                [
                    position[1],
                    end[1],
                ],
                [
                    position[2],
                    end[2],
                ],
                linestyle=styles[axis],
                linewidth=1.4,
            )

    def render_frame(self):
        i = self.i

        if self.debug:
            print(
                f"[render] frame "
                f"{i}/{self.n - 1}",
                flush=True,
            )

        try:
            self._render_frame_impl(i)
            self.update_style_display(i)
            self.fig.canvas.draw_idle()

        except Exception:
            self.paused = True

            print(
                "\n"
                "============================================================"
            )
            print(
                f"PLAYER ERROR while rendering frame {i} "
                f"of {self.n}"
            )
            print(
                "Playback has been paused so the exception is visible."
            )
            print(
                "============================================================"
            )
            traceback.print_exc()
            raise

    def _render_frame_impl(self, i):
        self.ax.cla()

        self.ax.set_xlim(
            *self.xlim
        )
        self.ax.set_ylim(
            *self.ylim
        )
        self.ax.set_zlim(
            *self.zlim
        )

        self.ax.set_xlabel(
            "command X [m]"
        )
        self.ax.set_ylabel(
            "command Y [m]"
        )
        self.ax.set_zlabel(
            "command Z [m]"
        )

        self.ax.set_title(
            "G1 sparse command player "
            "— explicit GUI timer"
        )

        # Ground z = 0.
        x0, x1 = self.xlim
        y0, y1 = self.ylim

        self.ax.plot(
            [
                x0,
                x1,
                x1,
                x0,
                x0,
            ],
            [
                y0,
                y0,
                y1,
                y1,
                y0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
            ],
            alpha=0.35,
            label="ground z=0",
        )

        # Retargeted source skeleton overlay.
        if (
            self.human_body is not None
            and self.human_indices is not None
        ):
            source_i = int(
                self.human_indices[i]
            )

            body = self.human_body[
                source_i
            ]

            for a, b in BODY_EDGES:
                pa = body[a]
                pb = body[b]

                if (
                    np.all(np.isfinite(pa))
                    and np.all(np.isfinite(pb))
                ):
                    self.ax.plot(
                        [
                            pa[0],
                            pb[0],
                        ],
                        [
                            pa[1],
                            pb[1],
                        ],
                        [
                            pa[2],
                            pb[2],
                        ],
                        alpha=0.25,
                        linewidth=1.0,
                    )

            for wrist_idx in (
                LEFT_WRIST,
                RIGHT_WRIST,
            ):
                p = body[wrist_idx]

                if np.all(
                    np.isfinite(p)
                ):
                    self.ax.scatter(
                        [p[0]],
                        [p[1]],
                        [p[2]],
                        s=20,
                    )

        left_wrist = np.asarray(
            self.cmd[
                "left_wrist_pos_world"
            ][i],
            dtype=np.float64,
        )
        right_wrist = np.asarray(
            self.cmd[
                "right_wrist_pos_world"
            ][i],
            dtype=np.float64,
        )

        self.draw_wrist_axes(
            left_wrist,
            self.cmd[
                "left_wrist_quat_world_wxyz"
            ][i],
            "left wrist command",
        )

        self.draw_wrist_axes(
            right_wrist,
            self.cmd[
                "right_wrist_quat_world_wxyz"
            ][i],
            "right wrist command",
        )

        torso_xy = np.asarray(
            self.cmd[
                "torso_center_xy_world"
            ][i],
            dtype=np.float64,
        )

        left_h = float(
            self.cmd[
                "left_shoulder_height"
            ][i]
        )
        right_h = float(
            self.cmd[
                "right_shoulder_height"
            ][i]
        )
        heading = float(
            self.cmd[
                "torso_heading_world"
            ][i]
        )

        if (
            np.all(
                np.isfinite(torso_xy)
            )
            and np.isfinite(left_h)
            and np.isfinite(right_h)
        ):
            torso_center = np.array([
                torso_xy[0],
                torso_xy[1],
                0.5
                * (
                    left_h
                    + right_h
                ),
            ])

            self.ax.scatter(
                [
                    torso_center[0]
                ],
                [
                    torso_center[1]
                ],
                [
                    torso_center[2]
                ],
                s=55,
                marker="s",
                label="torso preference",
            )

            if np.isfinite(heading):
                forward = np.array([
                    np.cos(heading),
                    np.sin(heading),
                    0.0,
                ])

                end = (
                    torso_center
                    + 0.22
                    * forward
                )

                self.ax.plot(
                    [
                        torso_center[0],
                        end[0],
                    ],
                    [
                        torso_center[1],
                        end[1],
                    ],
                    [
                        torso_center[2],
                        end[2],
                    ],
                    linewidth=2.0,
                    label="torso heading",
                )

        # Trails.
        if self.trail_frames > 0:
            start = max(
                0,
                i
                - self.trail_frames
                + 1,
            )
            sl = slice(
                start,
                i + 1,
            )

            for key in (
                "left_wrist_pos_world",
                "right_wrist_pos_world",
            ):
                trail = np.asarray(
                    self.cmd[key][sl],
                    dtype=np.float64,
                )

                valid = np.all(
                    np.isfinite(trail),
                    axis=1,
                )
                trail = trail[valid]

                if len(trail) >= 2:
                    self.ax.plot(
                        trail[:, 0],
                        trail[:, 1],
                        trail[:, 2],
                        alpha=0.35,
                    )

        scale = float(
            self.metadata.get(
                "morphology_scale",
                np.nan,
            )
        )

        verify_line = ""

        if (
            self.verification
            is not None
        ):
            left_error = (
                self.verification[
                    "left_error"
                ][i]
            )
            right_error = (
                self.verification[
                    "right_error"
                ][i]
            )

            verify_line = (
                "\n"
                f"scale check: "
                f"|Lxy err|="
                f"{np.linalg.norm(left_error[:2]):.7f} m, "
                f"|Rxy err|="
                f"{np.linalg.norm(right_error[:2]):.7f} m"
            )

        text = (
            f"frame {i + 1}/{self.n} "
            f"(index {i})\n"
            f"t="
            f"{float(self.cmd['timestamp'][i]):.3f} s  "
            f"speed={self.speed:.2f}x  "
            f"{'PAUSED' if self.paused else 'PLAYING'}\n"
            f"scale={scale:.5f}"
            f"{verify_line}\n"
            "SPACE pause | LEFT/RIGHT step | "
            "UP/DOWN speed | R reset | Q quit"
        )

        self.ax.text2D(
            0.02,
            0.98,
            text,
            transform=self.ax.transAxes,
            va="top",
            fontsize=9,
        )

        handles, labels = (
            self.ax.get_legend_handles_labels()
        )

        unique = {}

        for handle, label in zip(
            handles,
            labels,
        ):
            if (
                label
                and label
                not in unique
            ):
                unique[label] = handle

        self.ax.legend(
            unique.values(),
            unique.keys(),
            loc="upper right",
            fontsize=8,
        )

    def on_timer(self):
        if (
            self.closed
            or self.paused
        ):
            return

        next_i = self.i + 1

        if next_i >= self.n:
            next_i = 0

        self.i = next_i

        try:
            self.render_frame()
        except Exception:
            # render_frame already prints the traceback and pauses.
            return

    def on_key(self, event):
        key = event.key

        if key == " ":
            self.paused = (
                not self.paused
            )

        elif key == "left":
            self.paused = True
            self.i = (
                self.i - 1
            ) % self.n

        elif key == "right":
            self.paused = True
            self.i = (
                self.i + 1
            ) % self.n

        elif key == "up":
            self.speed = min(
                self.speed * 1.25,
                8.0,
            )
            self.update_timer_interval()

        elif key == "down":
            self.speed = max(
                self.speed / 1.25,
                0.125,
            )
            self.update_timer_interval()

        elif key in (
            "r",
            "R",
        ):
            self.paused = True
            self.i = 0

        elif key in (
            "q",
            "Q",
            "escape",
        ):
            plt.close(
                self.fig
            )
            return

        self.render_frame()

    def on_close(self, _):
        self.closed = True

        try:
            self.timer.stop()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        type=Path,
        help="Command NPZ file.",
    )

    parser.add_argument(
        "--source-episode",
        type=Path,
        default=None,
        help=(
            "Optional raw EgoSuite episode root. "
            "If omitted, metadata_json['episode'] is used."
        ),
    )

    style_selection = parser.add_mutually_exclusive_group()
    style_selection.add_argument(
        "--style",
        type=Path,
        default=None,
        help=(
            "Style descriptor NPZ. By default, COMMAND.style.npz is loaded "
            "automatically when present."
        ),
    )
    style_selection.add_argument(
        "--no-style",
        action="store_true",
        help="Disable descriptor auto-discovery and display.",
    )

    parser.add_argument(
        "--no-human",
        action="store_true",
        help="Disable source-human overlay and scaling verification.",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Playback FPS override.",
    )

    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Initial zero-based frame index.",
    )

    parser.add_argument(
        "--trail-frames",
        type=int,
        default=45,
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print every rendered frame index.",
    )

    args = parser.parse_args()

    command_path, cmd, metadata = (
        load_command(
            args.command
        )
    )

    print(
        f"Command: {command_path}"
    )
    print(
        f"Frames : "
        f"{len(cmd['timestamp'])}"
    )
    print(
        f"Interface: "
        f"{metadata.get('interface_name', 'unknown')} "
        f"v{metadata.get('interface_version', 'unknown')}"
    )

    style = None
    style_indices = None
    if not args.no_style:
        style_path = args.style
        if style_path is None:
            candidate = command_path.with_name(f"{command_path.stem}.style.npz")
            if candidate.exists():
                style_path = candidate
        if style_path is not None:
            style_path, style = load_style_descriptor(
                style_path,
                fallback_timestamps=cmd["timestamp"],
            )
            style_indices = nearest_indices(
                style["timestamp"],
                cmd["timestamp"],
            )
            print(
                f"Style  : {style_path}\n"
                f"         {style['descriptor'].shape[1]}D, "
                f"{len(style['descriptor'])} frames, "
                f"v{style['metadata'].get('descriptor_version', 'unknown')}"
            )
        elif args.style is None:
            print("Style  : no auto-discovered sidecar")

    human_body = None
    human_indices = None
    verification = None

    if not args.no_human:
        source = (
            args.source_episode
        )

        if (
            source is None
            and metadata.get(
                "episode"
            )
        ):
            source = Path(
                metadata["episode"]
            )

        if source is not None:
            source = (
                Path(source)
                .expanduser()
                .resolve()
            )

            if source.exists():
                raw = load_episode(
                    source
                )

                human_body = (
                    transform_body_to_command_world(
                        raw["body"],
                        metadata,
                    )
                )

                human_indices = nearest_indices(
                    raw["timestamps"],
                    cmd["timestamp"],
                )

                verification = (
                    verify_wrist_scaling(
                        raw["body"],
                        cmd,
                        metadata,
                        human_indices,
                    )
                )

                max_left = float(
                    np.nanmax(
                        np.abs(
                            verification[
                                "left_error"
                            ][:, :2]
                        )
                    )
                )
                max_right = float(
                    np.nanmax(
                        np.abs(
                            verification[
                                "right_error"
                            ][:, :2]
                        )
                    )
                )

                print(
                    f"Human overlay: {source}"
                )
                print(
                    "Max wrist XY verification error:"
                )
                print(
                    f"  left : {max_left:.9f} m"
                )
                print(
                    f"  right: {max_right:.9f} m"
                )

    print(
        f"Starting at frame index "
        f"{int(np.clip(args.start_frame, 0, len(cmd['timestamp']) - 1))}"
    )

    # IMPORTANT:
    # Keep a strong Python reference to the player for the entire lifetime
    # of the GUI window.  If the constructor result is discarded here,
    # the player <-> GUI timer reference cycle can be garbage-collected,
    # causing playback to stop after an apparently random small number
    # of frames.
    player = CommandPlayer(
        command_path=command_path,
        cmd=cmd,
        metadata=metadata,
        human_body=human_body,
        human_indices=human_indices,
        verification=verification,
        style=style,
        style_indices=style_indices,
        fps=args.fps,
        start_frame=args.start_frame,
        trail_frames=args.trail_frames,
        debug=args.debug,
    )

    # Also anchor both objects on the Matplotlib Figure.  This protects
    # against backends whose timer objects are otherwise only weakly held.
    player.fig._g1_command_player = player
    player.fig._g1_command_timer = player.timer

    plt.show()

    # Explicit cleanup after the window closes.
    try:
        player.timer.stop()
    except Exception:
        pass


if __name__ == "__main__":
    main()
