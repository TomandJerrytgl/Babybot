"""Observation data structures and the short-term stereo memory."""

from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Deque, List

import cv2
import numpy as np


@dataclass(frozen=True)
class Observation:
    """Compressed full-resolution history plus small analysis images."""

    observation_id: int
    timestamp: float
    monotonic_timestamp: float
    left: np.ndarray
    right: np.ndarray
    left_jpeg: bytes = b""
    right_jpeg: bytes = b""
    full_shape: tuple = ()

    @classmethod
    def from_frames(
        cls,
        observation_id,
        timestamp,
        monotonic_timestamp,
        left,
        right,
        jpeg_quality=85,
        analysis_width=320,
    ):
        left_jpeg = cls._encode(left, jpeg_quality)
        right_jpeg = cls._encode(right, jpeg_quality)
        return cls(
            observation_id=observation_id,
            timestamp=timestamp,
            monotonic_timestamp=monotonic_timestamp,
            left=cls._analysis_copy(left, analysis_width),
            right=cls._analysis_copy(right, analysis_width),
            left_jpeg=left_jpeg,
            right_jpeg=right_jpeg,
            full_shape=left.shape,
        )

    def decode(self, eye):
        if eye not in ("left", "right"):
            raise ValueError("eye must be 'left' or 'right'")
        encoded = self.left_jpeg if eye == "left" else self.right_jpeg
        if not encoded:
            return getattr(self, eye).copy()
        image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to decode {eye} observation JPEG")
        return image

    def map_window_to_full(self, window):
        if not self.full_shape:
            return tuple(int(value) for value in window)
        analysis_height, analysis_width = self.left.shape[:2]
        full_height, full_width = self.full_shape[:2]
        x, y, width, height = window
        return (
            int(round(x * full_width / analysis_width)),
            int(round(y * full_height / analysis_height)),
            max(1, int(round(width * full_width / analysis_width))),
            max(1, int(round(height * full_height / analysis_height))),
        )

    @staticmethod
    def _analysis_copy(image, analysis_width):
        height, width = image.shape[:2]
        target_width = min(width, int(analysis_width))
        target_height = max(1, int(round(height * target_width / width)))
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _encode(image, quality):
        ok, encoded = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
        )
        if not ok:
            raise RuntimeError("Failed to encode observation JPEG")
        return encoded.tobytes()


class ObservationBuffer:
    """Thread-safe, time-bounded in-memory storage for observations."""

    def __init__(self, retention_seconds: float = 20.0):
        if retention_seconds <= 0:
            raise ValueError("retention_seconds must be positive")
        self.retention_seconds = float(retention_seconds)
        self._items: Deque[Observation] = deque()
        self._lock = Lock()

    def append(self, observation: Observation) -> None:
        with self._lock:
            self._items.append(observation)
            self._discard_expired(observation.monotonic_timestamp)

    def prune(self, now_monotonic: float) -> None:
        with self._lock:
            self._discard_expired(now_monotonic)

    def snapshot(self) -> List[Observation]:
        with self._lock:
            return list(self._items)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def _discard_expired(self, now_monotonic: float) -> None:
        cutoff = now_monotonic - self.retention_seconds
        while self._items and self._items[0].monotonic_timestamp < cutoff:
            self._items.popleft()
