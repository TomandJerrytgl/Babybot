import unittest

import numpy as np

from observation import Observation, ObservationBuffer


def make_observation(identifier, monotonic_timestamp):
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    return Observation(
        observation_id=identifier,
        timestamp=1000.0 + identifier,
        monotonic_timestamp=monotonic_timestamp,
        left=image,
        right=image.copy(),
    )


class ObservationBufferTests(unittest.TestCase):
    def test_compressed_observation_keeps_full_resolution_jpeg_and_small_analysis(self):
        frame = np.full((80, 128, 3), 120, dtype=np.uint8)
        item = Observation.from_frames(1, 1.0, 1.0, frame, frame, jpeg_quality=85, analysis_width=32)
        self.assertEqual(item.left.shape[:2], (20, 32))
        self.assertTrue(item.left_jpeg.startswith(b"\xff\xd8"))
        self.assertEqual(item.decode("left").shape, frame.shape)
        self.assertEqual(item.map_window_to_full((8, 5, 16, 10)), (32, 20, 64, 40))

    def test_discards_items_older_than_retention_window(self):
        buffer = ObservationBuffer(retention_seconds=20.0)
        buffer.append(make_observation(1, 10.0))
        buffer.append(make_observation(2, 29.9))
        buffer.append(make_observation(3, 30.1))
        self.assertEqual(
            [item.observation_id for item in buffer.snapshot()],
            [2, 3],
        )

    def test_keeps_item_exactly_on_cutoff(self):
        buffer = ObservationBuffer(retention_seconds=20.0)
        buffer.append(make_observation(1, 10.0))
        buffer.append(make_observation(2, 30.0))
        self.assertEqual(len(buffer), 2)

    def test_rejects_non_positive_retention(self):
        with self.assertRaises(ValueError):
            ObservationBuffer(0)


if __name__ == "__main__":
    unittest.main()
