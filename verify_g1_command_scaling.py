#!/usr/bin/env python3
"""Numerically verify G1 command wrist-position scaling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from egosuite_to_g1_command import (
    LEFT_WRIST,
    RIGHT_WRIST,
    load_episode,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", type=Path)
    p.add_argument("--source-episode", type=Path, default=None)
    args = p.parse_args()

    with np.load(args.command, allow_pickle=False) as f:
        left = np.asarray(f["left_wrist_pos_world"], float)
        right = np.asarray(f["right_wrist_pos_world"], float)
        raw_meta = f["metadata_json"]
        if raw_meta.ndim == 0:
            raw_meta = raw_meta.item()
        meta = json.loads(str(raw_meta))

    source = args.source_episode
    if source is None:
        source = Path(meta["episode"])

    data = load_episode(source)
    body = np.asarray(data["body"], float)

    s = float(meta["morphology_scale"])
    origin = np.asarray(meta["source_origin_xy_world"], float)
    ground = float(meta["source_ground_z_reference"])

    expected_left = body[:, LEFT_WRIST].copy()
    expected_right = body[:, RIGHT_WRIST].copy()

    for x in (expected_left, expected_right):
        x[:,0] = s * (x[:,0] - origin[0])
        x[:,1] = s * (x[:,1] - origin[1])
        x[:,2] = s * (x[:,2] - ground)

    n = min(len(left), len(expected_left))
    le = left[:n] - expected_left[:n]
    re = right[:n] - expected_right[:n]

    print(f"interface version : {meta.get('interface_version')}")
    print(f"scale             : {s:.8f}")
    print(f"source origin XY  : {origin}")
    print(f"source ground z   : {ground:.8f}")
    print()
    print("Expected formula:")
    print("  cmd_xy = scale * (human_xy - source_origin_xy)")
    print("  cmd_z  = scale * (human_z  - source_ground_z)")
    print()
    print(f"max left  XY abs error: {np.max(np.abs(le[:,:2])):.10f} m")
    print(f"max right XY abs error: {np.max(np.abs(re[:,:2])):.10f} m")
    print(f"max left XYZ abs error: {np.max(np.abs(le)):.10f} m")
    print(f"max right XYZ abs error: {np.max(np.abs(re)):.10f} m")
    print()
    print("First-frame example:")
    print(f"  raw L wrist XY      : {body[0, LEFT_WRIST, :2]}")
    print(f"  raw-relative L XY   : {body[0, LEFT_WRIST, :2] - origin}")
    print(f"  expected scaled L XY: {expected_left[0,:2]}")
    print(f"  exported L XY       : {left[0,:2]}")


if __name__ == "__main__":
    main()
