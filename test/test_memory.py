import tempfile
import unittest

import numpy as np

from conscious import visual_feature
from memory import MemoryStore


class MemoryStoreTests(unittest.TestCase):
    @staticmethod
    def sample(color):
        image = np.full((24, 32, 3), color, dtype=np.uint8)
        return image, visual_feature(image)

    @staticmethod
    def metadata(timestamp):
        return {
            "timestamp": timestamp,
            "disparity_pixels": 4.0,
            "left_window": (1, 2, 20, 16),
            "right_window": (2, 2, 20, 16),
            "pixel_size": (20, 16),
            "calibrated": False,
            "distance": None,
            "physical_size": None,
        }

    def test_similar_samples_share_one_object_and_keep_three(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(directory, similarity_threshold=.70)
            for index, color in enumerate(((10, 80, 200), (12, 82, 198),
                                           (15, 85, 195), (18, 88, 192))):
                image, feature = self.sample(color)
                store.learn(image, image, feature, self.metadata(index + 1))
            self.assertEqual(store.object_count, 1)
            self.assertEqual(len(store.manifest["objects"][0]["samples"]), 3)

    def test_twentieth_distinct_object_stops_learning(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(directory, maximum_objects=2, similarity_threshold=1.01)
            for index, color in enumerate(((0, 0, 0), (0, 0, 255), (0, 255, 0))):
                image, feature = self.sample(color)
                store.learn(image, image, feature, self.metadata(index + 1))
            self.assertEqual(store.object_count, 2)
            self.assertTrue(store.learning_stopped)


if __name__ == "__main__":
    unittest.main()
