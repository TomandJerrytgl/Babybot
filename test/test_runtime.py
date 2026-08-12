import time
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

import cv2
import numpy as np

from attention import Attention
from main import PreviewStore, encode_preview, make_request_handler
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

    def test_expanded_region_stays_inside_image(self):
        item = observation(1, 1.0)
        attention = Attention(item, verbose=False)
        x, y, width, height = attention.expanded_region((90, 60, 20, 20), (80, 100, 3), 3.0)
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + width, 100)
        self.assertLessEqual(y + height, 80)


class WebPreviewTests(unittest.TestCase):
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
