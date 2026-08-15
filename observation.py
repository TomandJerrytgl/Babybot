"""Raw, short-lived sensory observations for Babybot."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Observation:
    """One raw stereo sensory snapshot; the runtime does not retain a history."""

    observation_id: int
    timestamp: float
    monotonic_timestamp: float
    left: np.ndarray
    right: np.ndarray

    @classmethod
    def from_frames(
        cls, observation_id, timestamp, monotonic_timestamp, left, right
    ):
        if left is None or right is None:
            raise ValueError("Observation requires both eye images")
        return cls(
            observation_id=int(observation_id),
            timestamp=float(timestamp),
            monotonic_timestamp=float(monotonic_timestamp),
            left=left,
            right=right,
        )
