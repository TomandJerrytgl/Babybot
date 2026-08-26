"""Stable reader and validator for Babybot stereo recording datasets."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator, Optional
import zipfile

import cv2


@dataclass(frozen=True)
class StereoFramePair:
    index: int
    timestamp_ns: int
    monotonic_ns: int
    sync_delta_ns: int
    left_path: Optional[Path]
    right_path: Optional[Path]


class StereoDataset:
    """Iterate an on-disk recording without assuming a future ML framework."""

    def __init__(self, recording_directory):
        self.root = Path(recording_directory)
        self.metadata_path = self.root / "metadata.json"
        self.pairs_path = self.root / "pairs.csv"
        if not self.metadata_path.is_file() or not self.pairs_path.is_file():
            raise FileNotFoundError(
                f"Not a Babybot stereo dataset: {self.root}"
            )
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.schema = self.metadata.get("schema", "babybot.stereo-recording/v1")
        if self.schema == "babybot.stereo-recording/v1":
            self._ensure_frames_available()
        elif self.schema != "babybot.stereo-recording/v2":
            raise ValueError(f"Unsupported stereo dataset schema: {self.schema}")

    def _ensure_frames_available(self):
        """Restore archived frames after a dataset was cloned from TGLgeneral."""
        frames = self.root / "frames"
        if frames.is_dir():
            return
        archive_path = self.root / "frames.zip"
        if not archive_path.is_file():
            raise FileNotFoundError(f"Dataset has neither frames/ nor frames.zip: {self.root}")
        root = self.root.resolve()
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                destination = (self.root / member.filename).resolve()
                if root != destination and root not in destination.parents:
                    raise ValueError(f"Unsafe archive member: {member.filename}")
            archive.extractall(self.root)

    def pairs(self) -> Iterator[StereoFramePair]:
        with self.pairs_path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                yield StereoFramePair(
                    index=int(row["index"]),
                    timestamp_ns=int(row["timestamp_ns"]),
                    monotonic_ns=int(row["monotonic_ns"]),
                    sync_delta_ns=int(row.get("sync_delta_ns", 0)),
                    left_path=(self.root / row["left_path"] if row.get("left_path") else None),
                    right_path=(self.root / row["right_path"] if row.get("right_path") else None),
                )

    def images(self):
        """Yield ``(pair, left_bgr, right_bgr)`` for model training."""
        if self.schema == "babybot.stereo-recording/v2":
            yield from self._video_images()
            return
        for pair in self.pairs():
            left = cv2.imread(str(pair.left_path), cv2.IMREAD_COLOR)
            right = cv2.imread(str(pair.right_path), cv2.IMREAD_COLOR)
            if left is None or right is None:
                raise ValueError(f"Unreadable frame pair {pair.index} in {self.root}")
            yield pair, left, right

    def _video_images(self):
        captures = {
            eye: cv2.VideoCapture(str(self._video_path(eye)))
            for eye in ("left", "right")
        }
        try:
            if not all(capture.isOpened() for capture in captures.values()):
                raise ValueError(f"Unable to open paired videos in {self.root}")
            for pair in self.pairs():
                left_ok, left = captures["left"].read()
                right_ok, right = captures["right"].read()
                if not left_ok or not right_ok or left is None or right is None:
                    raise ValueError(f"Video ended before pair {pair.index} in {self.root}")
                yield pair, left, right
        finally:
            for capture in captures.values():
                capture.release()

    def _video_path(self, eye):
        for video in self.metadata.get("videos", []):
            if video.get("eye") == eye:
                path = self.root / video["path"]
                if not path.is_file():
                    raise FileNotFoundError(path)
                return path
        raise ValueError(f"Metadata has no {eye} video")

    def validate(self):
        errors = []
        count = 0
        expected = 0
        shape = None
        for pair, left, right in self.images():
            if pair.index != expected:
                errors.append(f"Expected pair {expected}, found {pair.index}")
            if left.shape != right.shape:
                errors.append(f"Pair {pair.index} has different eye shapes")
            if shape is None:
                shape = left.shape
            elif left.shape != shape:
                errors.append(f"Pair {pair.index} changed image shape")
            expected = pair.index + 1
            count += 1
        declared = int(self.metadata.get("paired_frame_count", -1))
        if declared != count:
            errors.append(f"Metadata declares {declared} pairs, found {count}")
        return {"valid": not errors, "pair_count": count, "errors": errors}
