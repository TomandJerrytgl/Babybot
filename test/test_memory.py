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

    def test_size_has_high_weight_but_position_has_low_weight(self):
        image, feature = self.sample((20, 90, 210))
        base = self.metadata(1)
        base.update({"width_fraction": .20, "height_fraction": .25,
                     "area_fraction": .05, "relative_x": .2, "relative_y": .2,
                     "geometry_confidence": .9})
        sample = {"feature": feature.tolist(), "metadata": base}
        moved = dict(base, relative_x=.8, relative_y=.8)
        different_size = dict(base, width_fraction=.55, height_fraction=.60,
                              area_fraction=.33)
        moved_score = MemoryStore.perceptual_similarity(feature, moved, sample)
        size_score = MemoryStore.perceptual_similarity(feature, different_size, sample)
        self.assertGreater(moved_score, .95)
        self.assertGreater(moved_score, size_score)


if __name__ == "__main__":
    unittest.main()
