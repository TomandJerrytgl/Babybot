import unittest

import numpy as np

from observation import Observation
from perception import Perception


class ObservationTests(unittest.TestCase):
    def test_observation_preserves_raw_stereo_frames(self):
        left = np.full((80, 128, 3), 120, dtype=np.uint8)
        right = np.full((80, 128, 3), 80, dtype=np.uint8)
        item = Observation.from_frames(1, 10.0, 20.0, left, right)
        self.assertIs(item.left, left)
        self.assertIs(item.right, right)
        self.assertEqual(item.left.shape, (80, 128, 3))

    def test_observation_requires_both_eyes(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            Observation.from_frames(1, 1.0, 1.0, frame, None)


class PerceptionTests(unittest.TestCase):
    def test_perception_is_320_by_200(self):
        frame = np.full((800, 1280, 3), 120, dtype=np.uint8)
        observation = Observation.from_frames(1, 1.0, 1.0, frame, frame.copy())
        perception = Perception.from_observation(observation)
        self.assertEqual(perception.left.shape, (200, 320, 3))
        self.assertEqual(perception.right.shape, (200, 320, 3))

    def test_perception_window_maps_to_raw_observation(self):
        frame = np.zeros((800, 1280, 3), dtype=np.uint8)
        observation = Observation.from_frames(1, 1.0, 1.0, frame, frame.copy())
        perception = Perception.from_observation(observation)
        self.assertEqual(
            perception.map_window_to_observation((8, 5, 16, 10)),
            (32, 20, 64, 40),
        )


if __name__ == "__main__":
    unittest.main()
