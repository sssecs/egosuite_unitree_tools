#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import json
import numpy as np

from cross_embodiment_style_descriptor import (
    STYLE_DESCRIPTOR_VERSION,
    export_style_descriptor,
    extract_human_style_descriptor,
)
from dataset_filter import SittingFilterConfig, assess_sitting_episode
from egosuite_to_g1_command import (
    G1_HEIGHT_M,
    G1_WRIST_SPAN_M,
    extract_g1_command,
    export_g1_command,
    find_episode_roots,
    load_episode,
    write_control_interface_yaml,
)


def parse_quat(values):
    if values is None:
        return (1.0, 0.0, 0.0, 0.0)
    q = np.asarray(values, dtype=np.float64)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError("Invalid quaternion offset")
    q /= n
    return tuple(float(v) for v in q)


def label(root, ep):
    try:
        return str(ep.relative_to(root))
    except ValueError:
        return str(ep)


def resolve_episode(root, episodes, spec):
    try:
        idx = int(spec)
    except ValueError:
        idx = None

    if idx is not None:
        if idx < 0:
            idx += len(episodes)
        if not 0 <= idx < len(episodes):
            raise IndexError(idx)
        return episodes[idx]

    p = Path(spec).expanduser()
    if not p.is_absolute():
        p = root / p
    p = p.resolve()
    if not (p / "meta" / "info.json").exists():
        raise FileNotFoundError(p)
    return p


def output_path(root, ep, out_dir):
    rel = ep.relative_to(root)
    return out_dir / rel.parent / f"{rel.name}.npz"


def style_output_path(command_path):
    command_path = Path(command_path)
    return command_path.with_name(f"{command_path.stem}.style.npz")


def existing_interface_version(path):
    """Return exported interface version, or None if unreadable/legacy."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as f:
            if "metadata_json" not in f.files:
                return None
            raw = f["metadata_json"]
            if raw.ndim == 0:
                raw = raw.item()
            meta = json.loads(str(raw))
            return str(meta.get("interface_version", ""))
    except Exception:
        return None


def existing_style_descriptor_version(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as f:
            raw = f["metadata_json"]
            if raw.ndim == 0:
                raw = raw.item()
            metadata = json.loads(str(raw))
            return str(metadata.get("descriptor_version", ""))
    except Exception:
        return None


def write_filter_report(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(
        description=(
            "Convert EgoSuite human motion to G1-scaled sparse commands. "
            "Ground is z=0; all translational motion uses the same scale."
        )
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/mnt/hdd/humanoid_locomotion/datasets/data/"
            "EgoDemo/EgoStand-body/lerobot"
        ),
    )

    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--episode", action="append", metavar="INDEX_OR_PATH")
    sel.add_argument("--all", action="store_true")

    p.add_argument("--list", action="store_true")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./EgoDemo-G1-commands"),
    )

    p.add_argument(
        "--scale-mode",
        choices=["conservative", "height", "wrist_span"],
        default="conservative",
    )
    p.add_argument("--robot-height", type=float, default=G1_HEIGHT_M)
    p.add_argument("--robot-wrist-span", type=float, default=G1_WRIST_SPAN_M)
    p.add_argument("--scale", type=float, default=None)

    p.add_argument("--ground-window", type=int, default=5)
    p.add_argument("--origin-window-seconds", type=float, default=1.0)
    p.add_argument("--torso-filter-window", type=int, default=5)

    p.add_argument(
        "--left-q-offset",
        type=float,
        nargs=4,
        metavar=("W", "X", "Y", "Z"),
    )
    p.add_argument(
        "--right-q-offset",
        type=float,
        nargs=4,
        metavar=("W", "X", "Y", "Z"),
    )

    p.add_argument("--shoulder-min", type=float, default=None)
    p.add_argument("--shoulder-max", type=float, default=None)
    p.add_argument("--torso-xy-tolerance", type=float, default=0.10)
    p.add_argument("--shoulder-height-tolerance", type=float, default=0.05)
    p.add_argument("--torso-heading-tolerance", type=float, default=0.15)
    p.add_argument(
        "--locomotion-mode",
        choices=["auto", "feet_locked"],
        default="auto",
    )
    p.add_argument(
        "--sitting-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reject episodes dominated by persistent seated geometry (default: on)",
    )
    p.add_argument(
        "--sitting-reject-fraction",
        type=float,
        default=0.60,
        help="minimum persistent seated-frame fraction for episode rejection",
    )
    p.add_argument(
        "--filter-report",
        type=Path,
        default=None,
        help="JSONL audit report (default: OUTPUT_DIR/filter_report.jsonl)",
    )
    p.add_argument(
        "--style-descriptor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write independent *.style.npz descriptor sidecars (default: on)",
    )
    p.add_argument("--style-smoothing-window", type=int, default=5)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    if not 0.0 <= args.sitting_reject_fraction <= 1.0:
        p.error("--sitting-reject-fraction must be in [0, 1]")
    if args.style_smoothing_window < 1:
        p.error("--style-smoothing-window must be positive")

    root = args.root.expanduser().resolve()
    episodes = find_episode_roots(root)
    if not episodes:
        p.error(f"No episodes found under {root}")

    if args.list:
        for i, ep in enumerate(episodes):
            print(f"{i:5d}  {label(root, ep)}")
        return

    if args.all:
        selected = episodes
    else:
        specs = args.episode or ["0"]
        selected = [resolve_episode(root, episodes, s) for s in specs]

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_control_interface_yaml(out_dir / "g1_whole_body_command.yaml")

    converted = skipped = filtered = described = failed = 0
    filter_records = []
    filter_config = SittingFilterConfig(
        reject_fraction=args.sitting_reject_fraction,
    )

    for i, ep in enumerate(selected, 1):
        out_path = output_path(root, ep, out_dir)
        descriptor_path = style_output_path(out_path)
        print(f"[{i}/{len(selected)}] {label(root, ep)}")

        try:
            data = load_episode(ep)
            assessment = assess_sitting_episode(
                data["body"],
                data["timestamps"],
                fps=data["fps"],
                config=filter_config,
            )
            record = {
                "episode": label(root, ep),
                **{k: v for k, v in assessment.items() if not k.endswith("_mask")},
            }
            filter_records.append(record)
            if args.sitting_filter and assessment["reject"]:
                filtered += 1
                print(
                    "  filter: persistent seated pose "
                    f"({assessment['seated_fraction']:.1%} of frames)"
                )
                continue

            if args.style_descriptor:
                style_version = existing_style_descriptor_version(descriptor_path)
                if (
                    descriptor_path.exists()
                    and not args.overwrite
                    and style_version == STYLE_DESCRIPTOR_VERSION
                ):
                    print(f"  style: existing file kept: {descriptor_path}")
                else:
                    if descriptor_path.exists() and not args.overwrite:
                        print(
                            "  stale/incompatible style descriptor "
                            f"(version={style_version!r}); regenerating"
                        )
                    style = extract_human_style_descriptor(
                        data["body"],
                        data["timestamps"],
                        smoothing_window=args.style_smoothing_window,
                    )
                    export_style_descriptor(descriptor_path, style, ep)
                    described += 1
                    print(
                        f"  style descriptor : {style['descriptor'].shape} -> "
                        f"{descriptor_path}"
                    )

            old_version = existing_interface_version(out_path)
            if out_path.exists() and not args.overwrite and old_version == "3.2":
                print(f"  command: existing compatible v3.2 file kept: {out_path}")
                skipped += 1
                continue
            if out_path.exists() and not args.overwrite:
                print(
                    f"  stale/incompatible command (version={old_version!r}); "
                    f"regenerating: {out_path}"
                )

            cmd = extract_g1_command(
                data,
                scale_mode=args.scale_mode,
                robot_height=args.robot_height,
                robot_wrist_span=args.robot_wrist_span,
                explicit_scale=args.scale,
                ground_window=args.ground_window,
                origin_window_seconds=args.origin_window_seconds,
                torso_filter_window=args.torso_filter_window,
                left_q_offset=parse_quat(args.left_q_offset),
                right_q_offset=parse_quat(args.right_q_offset),
                shoulder_min=args.shoulder_min,
                shoulder_max=args.shoulder_max,
                torso_xy_tolerance=args.torso_xy_tolerance,
                shoulder_height_tolerance=args.shoulder_height_tolerance,
                torso_heading_tolerance=args.torso_heading_tolerance,
                locomotion_mode=args.locomotion_mode,
            )
            export_g1_command(out_path, cmd, ep)
            converted += 1

            if not args.quiet:
                print(f"  scale             : {cmd['scale']:.5f}")
                print(f"  source origin XY  : {cmd['source_origin_xy_world']}")
                print(f"  source ground z   : {cmd['source_ground_z_reference']:.5f}")
                print("  command ground z  : 0.0")
                print(f"  output            : {out_path}")

        except KeyboardInterrupt:
            raise
        except Exception as exc:
            failed += 1
            print(
                f"  ERROR: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            if len(selected) == 1:
                raise

    print(
        f"\nconverted={converted} described={described} filtered={filtered} "
        f"skipped={skipped} failed={failed}"
    )
    report_path = args.filter_report or (out_dir / "filter_report.jsonl")
    write_filter_report(report_path, filter_records)
    print(f"filter_report={report_path}")


if __name__ == "__main__":
    main()
