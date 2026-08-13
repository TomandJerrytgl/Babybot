import time
import unittest
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
from main import EyePipeline, PreviewStore, RuntimeConfig, encode_preview, make_request_handler
from observation import Observation, ObservationBuffer
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


class ObservationBufferTests(unittest.TestCase):
    def test_discards_items_older_than_retention_window(self):
        buffer = ObservationBuffer(retention_seconds=20.0)
        buffer.append(observation(1, 10.0))
        buffer.append(observation(2, 29.9))
        buffer.append(observation(3, 30.1))
        self.assertEqual([item.observation_id for item in buffer.snapshot()], [2, 3])

    def test_rejects_non_positive_retention(self):
        with self.assertRaises(ValueError):
            ObservationBuffer(0)


class AttentionTests(unittest.TestCase):
    def test_right_eye_is_processed_independently(self):
        left = np.zeros((120, 160, 3), dtype=np.uint8)
        right = left.copy()
        right[35:85, 80:130] = (0, 0, 255)
        item = Observation(1, time.time(), time.monotonic(), left, right)
        result = Attention(item, eye="right", verbose=False)
        self.assertTrue(result.candidates)
        self.assertIsNotNone(result.focus)

    def test_large_moving_object_uses_unified_scoring(self):
        previous = np.zeros((200, 320, 3), dtype=np.uint8)
        current = previous.copy()
        current[60:160, 150:280] = (40, 80, 220)
        item = Observation(1, time.time(), time.monotonic(), current, current.copy())
        result = Attention(
            item,
            eye="left",
            previous_image=previous,
            verbose=False,
        )
        self.assertTrue(result.candidates)
        self.assertEqual(result.focus_source, "mixed")
        self.assertGreater(result.candidates[0]["motion"], 0.0)
        x, y, width, height = result.focus
        self.assertLessEqual(x, 150)
        self.assertLessEqual(y, 60)
        self.assertGreaterEqual(x + width, 280)
        self.assertGreaterEqual(y + height, 160)

    def test_small_motion_below_half_percent_gets_no_motion_bonus(self):
        previous = np.zeros((200, 320, 3), dtype=np.uint8)
        current = previous.copy()
        current[90:100, 150:160] = (0, 0, 255)
        item = Observation(1, time.time(), time.monotonic(), current, current.copy())
        result = Attention(item, previous_image=previous, verbose=False)
        self.assertTrue(all(candidate["motion"] == 0.0 for candidate in result.candidates))

    def test_global_brightness_change_is_not_motion(self):
        previous = np.full((120, 160, 3), 40, dtype=np.uint8)
        current = np.full((120, 160, 3), 90, dtype=np.uint8)
        item = Observation(1, time.time(), time.monotonic(), current, current.copy())
        result = Attention(item, previous_image=previous, verbose=False)
        self.assertTrue(all(candidate["motion"] == 0.0 for candidate in result.candidates))

    def test_center_preference_breaks_equal_visual_tie(self):
        attention = Attention.__new__(Attention)
        shape = (200, 320, 3)
        center = attention.center_preference((135, 75, 50, 50), shape)
        edge = attention.center_preference((0, 0, 50, 50), shape)
        self.assertGreater(center, edge)

    def test_output_window_adds_ten_percent_padding(self):
        attention = Attention.__new__(Attention)
        window = attention.map_and_pad_window(
            (50, 40, 100, 80), 1.0, 1.0, (200, 300, 3), 0.10
        )
        self.assertEqual(window, (40, 32, 120, 96))


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


class EyePipelineTests(unittest.TestCase):
    def test_failed_validation_hides_candidate_immediately(self):
        source = TemplateTrackerTests.textured_frame(80, 60)
        reference = AttentionReference(
            observation_id=1,
            timestamp=1.0,
            window=(80, 60, 50, 40),
            signature=visual_signature(source, (80, 60, 50, 40)),
        )
        pipeline = EyePipeline("left")
        pipeline.validator = AttentionValidator(reference, similarity_threshold=0.95)
        pipeline.candidate = {"window": reference.window, "score": 1.0}
        pipeline.validate(np.full_like(source, 255))
        self.assertEqual(pipeline.candidates(), [])


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

    def test_reference_replacement_resets_search_center(self):
        image = TemplateTrackerTests.textured_frame(80, 60)
        first = AttentionReference(1, 1.0, (80, 60, 50, 40), visual_signature(image, (80, 60, 50, 40)))
        second = AttentionReference(2, 2.0, (20, 30, 50, 40), visual_signature(image, (80, 60, 50, 40)))
        validator = AttentionValidator(first)
        validator.last_confirmed_window = (100, 70, 50, 40)
        validator.replace_reference(second)
        self.assertEqual(validator.reference.observation_id, 2)
        self.assertEqual(validator.last_confirmed_window, second.window)


class WebPreviewTests(unittest.TestCase):
    def test_default_web_host_is_loopback_only(self):
        self.assertEqual(RuntimeConfig().web_host, "127.0.0.1")

    def test_default_processing_rates(self):
        config = RuntimeConfig()
        self.assertEqual(config.observation_interval, 0.1)
        self.assertEqual(config.attention_interval, 0.5)
        self.assertEqual(config.preview_fps, 10.0)

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
        self.store.update(image, image, [], [], 7, 80)
        page = urlopen(self.base_url + "/", timeout=2).read().decode("utf-8")
        self.assertIn("Left eye", page)
        response = urlopen(self.base_url + "/frame/right.jpg", timeout=2)
        body = response.read()
        self.assertEqual(response.headers["X-Observation-Id"], "7")
        self.assertIsNotNone(cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR))

    def test_frame_is_unavailable_before_first_observation(self):
        with self.assertRaises(HTTPError) as context:
            urlopen(self.base_url + "/frame/left.jpg", timeout=2)
        self.assertEqual(context.exception.code, 503)


if __name__ == "__main__":
    unittest.main()
