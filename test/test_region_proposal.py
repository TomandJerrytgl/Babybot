import unittest

import cv2
import numpy as np

from region_proposal import RegionProposalConfig, StereoRegionProposer


class RegionProposalTests(unittest.TestCase):
    def setUp(self):
        self.proposer = StereoRegionProposer(RegionProposalConfig())

    def test_local_lab_growth_separates_strong_color_regions(self):
        image = np.zeros((60, 100, 3), np.uint8)
        image[:, :50] = (20, 40, 220)
        image[:, 50:] = (220, 60, 20)
        labels, regions, _edges = self.proposer.grow_regions(image)
        self.assertNotEqual(labels[30, 20], labels[30, 80])
        self.assertGreaterEqual(len(regions), 2)

    def test_interface_scale_rejects_small_object_against_large_table(self):
        small = {"area": 500, "lab_mean": np.array([100, 130, 130]),
                 "bbox": (20, 20, 50, 10)}
        table = {"area": 15000, "lab_mean": np.array([105, 130, 130]),
                 "bbox": (0, 30, 300, 100)}
        layer = {"area": 600, "lab_mean": np.array([170, 160, 90]),
                 "bbox": (20, 30, 52, 12)}
        boundary = {"shared": 48, "horizontal": 48, "vertical": 0,
                    "edge_sum": 6.0}
        table_score = self.proposer.merge_features(
            small, table, boundary, 64000
        )["scale_compatibility"]
        layer_score = self.proposer.merge_features(
            small, layer, boundary, 64000
        )["scale_compatibility"]
        self.assertGreater(layer_score, table_score)

    def test_regions_below_three_percent_are_not_proposed_alone(self):
        image = np.full((200, 320, 3), 120, np.uint8)
        image[20:40, 20:40] = (20, 80, 220)
        labels, regions, edges = self.proposer.grow_regions(image)
        windows, diagnostics, views = self.proposer.propose_eye(
            labels, regions, edges, None
        )
        self.assertTrue(all(width * height >= 1920 for _x, _y, width, height in windows))
        self.assertEqual(diagnostics["initial_region_count"], len(regions))
        self.assertEqual(set(views), {
            "initial_mask", "initial_overlay", "visual_merged_mask",
            "stereo_merged_mask", "stereo_merged_overlay",
        })
        self.assertTrue(all(view.shape == image.shape for view in views.values()))

    def test_initial_mask_fills_regions_with_distinct_pseudocolors(self):
        image = np.zeros((60, 100, 3), np.uint8)
        image[:, :50] = (20, 40, 220)
        image[:, 50:] = (220, 60, 20)
        labels, regions, edges = self.proposer.grow_regions(image)
        _windows, _diagnostics, views = self.proposer.propose_eye(
            labels, regions, edges, None, image
        )
        mask = views["initial_mask"]
        self.assertFalse(np.array_equal(mask, image))
        self.assertFalse(np.array_equal(mask[30, 20], mask[30, 80]))

    def test_merge_diagnostics_expose_every_score_component(self):
        first = {"area": 500, "lab_mean": np.array([100, 130, 130]),
                 "bbox": (20, 20, 50, 10)}
        second = {"area": 600, "lab_mean": np.array([105, 132, 128]),
                  "bbox": (20, 30, 52, 12)}
        boundary = {"shared": 48, "horizontal": 48, "vertical": 0,
                    "edge_sum": 6.0}
        scores = self.proposer.merge_features(first, second, boundary, 64000, .8)
        self.assertEqual(set(scores), {
            "color_similarity", "weak_boundary", "scale_compatibility",
            "merged_fill", "shape_continuity", "small_region_preference",
            "depth_similarity", "merge_score",
        })


if __name__ == "__main__":
    unittest.main()
