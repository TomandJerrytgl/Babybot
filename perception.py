"""Processed sensory data used by Babybot to find attention."""

from dataclasses import dataclass

import cv2
import numpy as np

from observation import Observation


@dataclass(frozen=True)
class Perception:
    """A disposable 320x200 stereo view derived from one raw observation."""

    perception_id: int
    timestamp: float
    monotonic_timestamp: float
    left: np.ndarray
    right: np.ndarray
    observation_shape: tuple

    @property
    def observation_id(self):
        """Compatibility identifier used by the current attention interface."""
        return self.perception_id

    @classmethod
    def from_observation(cls, observation: Observation, width=320, height=200):
        if width <= 0 or height <= 0:
            raise ValueError("Perception dimensions must be positive")
        return cls(
            perception_id=observation.observation_id,
            timestamp=observation.timestamp,
            monotonic_timestamp=observation.monotonic_timestamp,
            left=cv2.resize(observation.left, (int(width), int(height)), interpolation=cv2.INTER_AREA),
            right=cv2.resize(observation.right, (int(width), int(height)), interpolation=cv2.INTER_AREA),
            observation_shape=observation.left.shape,
        )

    def map_window_to_observation(self, window):
        observation_height, observation_width = self.observation_shape[:2]
        perception_height, perception_width = self.left.shape[:2]
        x, y, width, height = window
        return (
            int(round(x * observation_width / perception_width)),
            int(round(y * observation_height / perception_height)),
            max(1, int(round(width * observation_width / perception_width))),
            max(1, int(round(height * observation_height / perception_height))),
        )
