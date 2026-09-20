from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cross_embodiment_style_descriptor import (
    STYLE_DESCRIPTOR_NAMES,
    export_style_descriptor,
    extract_human_style_descriptor,
)
from dataset_filter import assess_sitting_episode


def make_body(*, seated: bool, frames: int = 60) -> np.ndarray:
    body = np.zeros((frames, 22, 3), dtype=np.float64)
    pelvis_z = 0.55 if seated else 1.0
    knee_x = 0.40 if seated else 0.02
    knee_z = 0.55 if seated else 0.52
    for side, y in ((0, 0.10), (1, -0.10)):
        hip, knee, ankle, foot = (
            (1, 4, 7, 10) if side == 0 else (2, 5, 8, 11)
        )
        body[:, hip] = (0.0, y, pelvis_z - 0.02)
        body[:, knee] = (knee_x, y, knee_z)
        body[:, ankle] = (knee_x if seated else 0.0, y, 0.06)
        body[:, foot] = (knee_x + 0.08 if seated else 0.10, y, 0.0)
    body[:, 0] = (0.0, 0.0, pelvis_z)
    body[:, 16] = (0.0, 0.22, pelvis_z + 0.55)
    body[:, 17] = (0.0, -0.22, pelvis_z + 0.55)
    return body


class DatasetProcessingTest(unittest.TestCase):
    def test_persistent_sitting_is_rejected_but_standing_is_kept(self):
        timestamps = np.arange(60) / 30.0
        seated = assess_sitting_episode(make_body(seated=True), timestamps)
        standing = assess_sitting_episode(make_body(seated=False), timestamps)
        self.assertTrue(seated["reject"])
        self.assertGreater(seated["seated_fraction"], 0.95)
        self.assertFalse(standing["reject"])

    def test_descriptor_schema_and_sidecar(self):
        timestamps = np.arange(60) / 30.0
        style = extract_human_style_descriptor(
            make_body(seated=False), timestamps
        )
        self.assertEqual(style["descriptor"].shape, (60, 17))
        self.assertEqual(len(STYLE_DESCRIPTOR_NAMES), 17)
        self.assertTrue(np.allclose(style["descriptor"][:, 0], 1.0))
        self.assertTrue(np.allclose(style["descriptor"][:, 5], 0.0))
        self.assertTrue(np.allclose(style["descriptor"][:, 8], 0.0))
        self.assertTrue(np.all(style["valid_mask"]))
        with tempfile.TemporaryDirectory() as directory:
            path = export_style_descriptor(
                Path(directory) / "episode.style.npz", style, "episode"
            )
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(data["timestamp"].shape, (60,))
                self.assertEqual(data["style_descriptor"].shape, (60, 17))
                self.assertEqual(data["style_descriptor_names"].shape, (17,))


if __name__ == "__main__":
    unittest.main()
