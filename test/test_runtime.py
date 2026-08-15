import pickle
import tempfile
import time
import unittest
from concurrent.futures import Future
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import urlopen

import cv2
import numpy as np

from attention import (
    Attention,
    AttentionReference,
    AttentionValidator,
    TemplateTracker,
    visual_signature,
)
from main import (
    LatestStereoFrame,
    PreviewStore,
    RuntimeConfig,
    StereoCognition,
    calculate_stereo_attention,
    encode_preview,
    locate_template,
    make_request_handler,
)
from conscious import Conscious, ConsciousObject, visual_feature
from memory import MemoryStore
from observation import Observation
from perception import Perception
from http.server import ThreadingHTTPServer
import threading


def observation(identifier, monotonic_timestamp):
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    return Observation(
        observation_id=identifier,
        timestamp=1000.0 + identifier,
        monotonic_timestamp=monotonic_timestamp,
        left=image,
        right=image.copy(),
    )


class AttentionTests(unittest.TestCase):
    def test_right_eye_is_processed_independently(self):
        left = np.zeros((120, 160, 3), dtype=np.uint8)
        right = left.copy()
        right[35:85, 80:130] = (0, 0, 255)
        item = Observation(1, time.time(), time.monotonic(), left, right)
        result = Attention(item, eye="right", verbose=False)
        self.assertTrue(result.candidates)
        self.assertIsNotNone(result.focus)

    def test_colored_single_frame_object_is_detected(self):
        image = np.full((200, 320, 3), 180, dtype=np.uint8)
        image[60:160, 150:280] = (40, 80, 220)
        item = Observation(1, time.time(), time.monotonic(), image, image.copy())
        result = Attention(item, eye="left", verbose=False)
        self.assertTrue(result.candidates)
        self.assertEqual(result.focus_source, "default")
        x, y, width, height = result.focus
        self.assertGreaterEqual(width, 16)
        self.assertGreaterEqual(height, 16)
        self.assertGreater(
            Attention.intersection_area((x, y, width, height), (150, 60, 130, 100)),
            0,
        )

    def test_same_single_frame_produces_same_default_attention(self):
        image = np.full((120, 160, 3), 160, dtype=np.uint8)
        image[30:90, 55:120] = (30, 100, 220)
        item = Observation(1, time.time(), time.monotonic(), image, image.copy())
        first = Attention(item, verbose=False)
        second = Attention(item, verbose=False)
        self.assertEqual(first.focus, second.focus)
        self.assertEqual(
            [candidate["score"] for candidate in first.candidates],
            [candidate["score"] for candidate in second.candidates],
        )

    def test_uniform_background_creates_no_candidate(self):
        image = np.full((120, 160, 3), 120, dtype=np.uint8)
        item = Observation(1, time.time(), time.monotonic(), image, image.copy())
        result = Attention(item, verbose=False)
        self.assertEqual(result.candidates, [])

    def test_center_preference_breaks_equal_visual_tie(self):
        attention = Attention.__new__(Attention)
        shape = (200, 320, 3)
        center = attention.center_preference((135, 75, 50, 50), shape)
        edge = attention.center_preference((0, 0, 50, 50), shape)
        self.assertGreater(center, edge)

    def test_default_output_allows_adaptive_rectangles(self):
        image = np.full((200, 320, 3), 170, dtype=np.uint8)
        image[75:125, 70:250] = (20, 80, 230)
        item = Observation(1, time.time(), time.monotonic(), image, image.copy())
        result = Attention(item, verbose=False)
        self.assertTrue(result.candidates)
        self.assertTrue(any(
            candidate["window"][2] > candidate["window"][3]
            for candidate in result.candidates
        ))


class TemplateTrackerTests(unittest.TestCase):
    @staticmethod
    def textured_frame(x, y):
        image = np.zeros((180, 280, 3), dtype=np.uint8)
        patch = image[y:y + 40, x:x + 50]
        patch[:] = (20, 120, 220)
        cv2.line(patch, (0, 0), (49, 39), (255, 255, 255), 3)
        cv2.circle(patch, (30, 15), 7, (0, 0, 0), -1)
        return image

    def test_relocates_template_near_previous_position(self):
        source = self.textured_frame(80, 60)
        current = self.textured_frame(94, 68)
        tracker = TemplateTracker(source, (80, 60, 50, 40))
        result = tracker.locate(current)
        self.assertIsNotNone(result.window)
        x, y, width, height = result.window
        self.assertAlmostEqual(x, 94, delta=2)
        self.assertAlmostEqual(y, 68, delta=2)
        self.assertGreaterEqual(result.confidence, 0.75)
        self.assertEqual(tracker.window, (x, y, width, height))

    def test_failed_match_does_not_update_search_center(self):
        source = self.textured_frame(80, 60)
        tracker = TemplateTracker(source, (80, 60, 50, 40))
        result = tracker.locate(np.zeros_like(source))
        self.assertIsNone(result.window)
        self.assertEqual(tracker.window, (80, 60, 50, 40))


class StereoPipelineTests(unittest.TestCase):
    def test_attention_process_payload_is_serializable(self):
        perception = Perception.from_observation(observation(1, 1.0))
        payload = pickle.dumps((calculate_stereo_attention, perception, "default", None))
        self.assertTrue(payload)

    def test_template_search_changes_width_and_height(self):
        template = np.zeros((20, 30, 3), np.uint8)
        template[:, :] = (10, 80, 210)
        cv2.line(template, (0, 0), (29, 19), (255, 255, 255), 2)
        image = np.zeros((100, 140, 3), np.uint8)
        image[30:54, 50:86] = cv2.resize(template, (36, 24))
        result = locate_template(image, template)
        self.assertGreater(result["similarity"], .85)
        self.assertNotEqual(result["window"][2], result["window"][3])

    def test_conscious_capacity_and_removal(self):
        conscious = Conscious(capacity=3)
        image = np.zeros((8, 8, 3), np.uint8)
        for index in range(4):
            item = ConsciousObject(str(index), image, image, np.zeros(99), (0, 0, 8, 8),
                                   (0, 0, 8, 8), 0.0)
            self.assertEqual(conscious.add(item), index < 3)
        self.assertEqual(len(conscious), 3)
        self.assertTrue(conscious.remove("1"))
        self.assertEqual(len(conscious), 2)

    def test_partial_overlap_keeps_higher_score_not_larger_area(self):
        attention = Attention.__new__(Attention)
        candidates = [
            {"window": (10, 10, 100, 100), "score": 0.6, "rank": 1},
            {"window": (90, 90, 50, 50), "score": 0.8, "rank": 3},
        ]
        selected = attention.suppress_overlaps(candidates)
        self.assertEqual([item["rank"] for item in selected], [3])


class AttentionValidatorTests(unittest.TestCase):
    def test_success_updates_next_search_center(self):
        source = TemplateTrackerTests.textured_frame(80, 60)
        moved = TemplateTrackerTests.textured_frame(105, 70)
        reference = AttentionReference(
            observation_id=1,
            timestamp=1.0,
            window=(80, 60, 50, 40),
            signature=visual_signature(source, (80, 60, 50, 40)),
        )
        validator = AttentionValidator(reference, similarity_threshold=0.70)
        result = validator.validate(moved)
        self.assertIsNotNone(result.window)
        self.assertEqual(validator.last_confirmed_window, result.window)
        self.assertAlmostEqual(result.window[0], 105, delta=2)
        self.assertAlmostEqual(result.window[1], 70, delta=2)

    def test_reference_replacement_resets_search_center(self):
        image = TemplateTrackerTests.textured_frame(80, 60)
        first = AttentionReference(1, 1.0, (80, 60, 50, 40), visual_signature(image, (80, 60, 50, 40)))
        second = AttentionReference(2, 2.0, (20, 30, 50, 40), visual_signature(image, (80, 60, 50, 40)))
        validator = AttentionValidator(first)
        validator.last_confirmed_window = (100, 70, 50, 40)
        validator.replace_reference(second)
        self.assertEqual(validator.reference.observation_id, 2)
        self.assertEqual(validator.last_confirmed_window, second.window)


class CandidateGenerationTests(unittest.TestCase):
    def test_coarse_windows_include_square_and_rectangular_shapes(self):
        attention = Attention.__new__(Attention)
        windows = attention.coarse_adaptive_windows((200, 320, 3), scales=(64,))
        self.assertTrue(any(width == height for _x, _y, width, height in windows))
        self.assertTrue(any(width > height for _x, _y, width, height in windows))
        self.assertTrue(any(width < height for _x, _y, width, height in windows))

    def test_coarse_windows_cover_far_image_edges(self):
        attention = Attention.__new__(Attention)
        windows = attention.coarse_adaptive_windows(
            (200, 320, 3), scales=(64,), aspect_ratios=(1.0,)
        )
        self.assertIn((0, 0, 64, 64), windows)
        self.assertIn((256, 136, 64, 64), windows)

    def test_refinement_changes_width_and_height_independently(self):
        attention = Attention.__new__(Attention)
        windows = attention.refined_windows((80, 60, 64, 64), (200, 320, 3))
        self.assertIn((80, 60, 72, 56), windows)
        self.assertIn((80, 60, 56, 72), windows)

    def test_surround_contrast_creates_candidate(self):
        attention = Attention.__new__(Attention)
        image = np.full((200, 320, 3), 165, dtype=np.uint8)
        image[68:132, 128:192] = (20, 80, 230)
        candidate = attention.evaluate_fixed_window(image, (128, 68, 64, 64))
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["window"], (128, 68, 64, 64))

    def test_uniform_window_is_not_attention(self):
        attention = Attention.__new__(Attention)
        image = np.full((200, 320, 3), 120, dtype=np.uint8)
        candidate = attention.evaluate_fixed_window(image, (128, 68, 64, 64))
        self.assertIsNone(candidate)

    def test_colorful_window_scores_above_white_window(self):
        attention = Attention.__new__(Attention)
        white = np.full((200, 320, 3), 120, dtype=np.uint8)
        colorful = white.copy()
        white[68:132, 128:192] = 245
        colorful[68:132, 128:192] = (20, 80, 230)
        white_candidate = attention.evaluate_fixed_window(white, (128, 68, 64, 64))
        colorful_candidate = attention.evaluate_fixed_window(colorful, (128, 68, 64, 64))
        self.assertIsNotNone(colorful_candidate)
        if white_candidate is not None:
            self.assertGreater(colorful_candidate["score"], white_candidate["score"])

    def test_internal_edges_raise_attention(self):
        attention = Attention.__new__(Attention)
        image = np.full((200, 320, 3), 120, dtype=np.uint8)
        for offset in range(0, 64, 8):
            cv2.line(image, (128 + offset, 68), (128 + offset, 131), (220, 220, 220), 2)
        candidate = attention.evaluate_fixed_window(image, (128, 68, 64, 64))
        self.assertIsNotNone(candidate)
        self.assertGreater(candidate["edge"], 0.0)

    def test_integral_window_scan_is_fast(self):
        image = np.full((200, 320, 3), 160, dtype=np.uint8)
        image[50:150, 100:220] = (20, 80, 230)
        item = Observation(1, time.time(), time.monotonic(), image, image.copy())
        started = time.perf_counter()
        Attention(item, verbose=False)
        self.assertLess(time.perf_counter() - started, 5.00)

    def test_higher_score_inner_box_is_kept_with_outer_box(self):
        attention = Attention.__new__(Attention)
        outer = {"window": (0, 0, 100, 100), "score": 0.6}
        inner = {"window": (20, 20, 30, 30), "score": 0.8}
        selected = attention.suppress_overlaps([outer, inner])
        self.assertEqual(len(selected), 2)

    def test_lower_score_inner_box_is_suppressed(self):
        attention = Attention.__new__(Attention)
        outer = {"window": (0, 0, 100, 100), "score": 0.8}
        inner = {"window": (20, 20, 30, 30), "score": 0.6}
        selected = attention.suppress_overlaps([outer, inner])
        self.assertEqual(selected, [outer])

class WebPreviewTests(unittest.TestCase):
    def test_default_web_host_is_loopback_only(self):
        self.assertEqual(RuntimeConfig().web_host, "127.0.0.1")

    def test_default_processing_rates(self):
        config = RuntimeConfig()
        self.assertFalse(hasattr(config, "perception_interval"))
        self.assertEqual(config.observation_preview_fps, 20.0)
        self.assertEqual(config.perception_width, 320)
        self.assertEqual(config.perception_height, 200)
        self.assertFalse(hasattr(config, "retention_seconds"))

    def test_latest_frame_buffer_overwrites_old_frame(self):
        store = LatestStereoFrame()
        first = np.zeros((2, 2, 3), dtype=np.uint8)
        second = np.full((2, 2, 3), 255, dtype=np.uint8)
        store.update(first, first)
        store.update(second, second)
        left, right, version = store.snapshot()
        self.assertEqual(version, 2)
        self.assertTrue(np.all(left == 255))
        self.assertTrue(np.all(right == 255))

    def setUp(self):
        self.store = PreviewStore()
        handler = make_request_handler(self.store)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_page_and_stereo_frames_are_served(self):
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        self.store.update("observation", image, image, [], [], 7, 80)
        self.store.update("perception", image, image, [], [], 3, 80)
        page = urlopen(self.base_url + "/", timeout=2).read().decode("utf-8")
        self.assertIn("Raw observation", page)
        self.assertIn("Perception", page)
        response = urlopen(self.base_url + "/frame/observation/right.jpg", timeout=2)
        body = response.read()
        self.assertEqual(response.headers["X-Frame-Id"], "7")
        self.assertIsNotNone(cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR))
        perception_response = urlopen(
            self.base_url + "/frame/perception/left.jpg", timeout=2
        )
        self.assertEqual(perception_response.headers["X-Frame-Id"], "3")

    def test_frame_is_unavailable_before_first_observation(self):
        with self.assertRaises(HTTPError) as context:
            urlopen(self.base_url + "/frame/observation/left.jpg", timeout=2)
        self.assertEqual(context.exception.code, 503)


if __name__ == "__main__":
    unittest.main()
