"""Observation data structures and the short-term stereo memory."""

from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Deque, List

import numpy as np


@dataclass(frozen=True)
class Observation:
    """One approximately synchronized pair of full-resolution camera frames."""

    observation_id: int
    timestamp: float
    monotonic_timestamp: float
    left: np.ndarray
    right: np.ndarray


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
