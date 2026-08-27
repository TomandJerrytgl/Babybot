import json
from pathlib import Path
import tempfile
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer
import threading
from types import SimpleNamespace

import cv2
import numpy as np

from record import RecordingLibrary, make_record_handler
from stereo_camera import LatestStereoFrame
from stereo_dataset import StereoDataset
from stereo_recording import StereoRecorder, StereoRecordingConfig


def wait_until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("Timed out waiting for background recording work")


class StereoRecorderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.recorder = StereoRecorder(StereoRecordingConfig(
            data_root=self.temporary.name,
            camera_fps=10.0,
            jpeg_quality=90,
            queue_capacity=8,
        ))

    def tearDown(self):
        self.recorder.shutdown()
        self.temporary.cleanup()

    def test_recording_creates_directly_readable_paired_dataset(self):
        left = np.zeros((24, 32, 3), dtype=np.uint8)
        right = np.zeros((24, 32, 3), dtype=np.uint8)
        self.assertTrue(self.recorder.start(left.shape))
        initial_status = self.recorder.status()
        self.assertEqual(initial_status["queue_capacity"], 8)
        self.assertEqual(initial_status["queue_percent"], 0.0)
        for index in range(3):
            left[:] = index * 30
            right[:] = index * 30 + 5
            self.assertTrue(self.recorder.submit(
                left, right, timestamp_ns=1_000_000_000 + index * 100_000_000,
                monotonic_ns=2_000_000_000 + index * 100_000_000,
                sync_delta_ns=100 + index,
            ))
        self.assertTrue(self.recorder.stop_async())
        status = wait_until(
            lambda: self.recorder.status()
            if self.recorder.status()["state"] == "idle" else None
        )
        self.assertEqual(status["upload_state"], "failed")
        batch = Path(self.temporary.name) / status["batch"]
        dataset = StereoDataset(batch)
        validation = dataset.validate()
        self.assertTrue(validation["valid"], validation["errors"])
        self.assertEqual(validation["pair_count"], 3)
        inspection = RecordingLibrary(self.temporary.name).inspect(batch)
        self.assertTrue(inspection["valid"], inspection)
        self.assertEqual(inspection["videos"]["left"]["frames"], 3)
        self.assertEqual(inspection["videos"]["right"]["frames"], 3)
        library = RecordingLibrary(self.temporary.name)
        paired_jpeg = library.paired_frame_jpeg(batch.name, 1)
        paired_image = cv2.imdecode(np.frombuffer(paired_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(paired_image.shape[:2], (24, 64))
        with self.assertRaises(IndexError):
            library.paired_frame_jpeg(batch.name, 99)
        library.close()
        self.assertFalse((batch / "frames").exists())
        self.assertFalse((batch / "frames.zip").exists())
        self.assertTrue(any((batch / "videos").glob("left.*")))
        self.assertTrue(any((batch / "videos").glob("right.*")))
        for eye in ("left", "right"):
            video_path = next((batch / "videos").glob(f"{eye}.*"))
            capture = cv2.VideoCapture(str(video_path))
            try:
                self.assertTrue(capture.isOpened())
                self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 10.0, places=2)
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
            finally:
                capture.release()
        metadata = json.loads((batch / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["schema"], "babybot.stereo-recording/v2")
        self.assertEqual(metadata["paired_frame_count"], 3)
        self.assertEqual(metadata["pairing"], "same grab/retrieve capture cycle")
        self.assertEqual(metadata["sync_delta_max_ns"], 102)
        self.assertAlmostEqual(metadata["capture_duration_seconds"], 0.2)
        self.assertAlmostEqual(metadata["effective_capture_fps"], 10.0)
        self.assertAlmostEqual(metadata["video_fps"], 10.0)
        self.assertEqual(metadata["videos"][0]["codec"], "MJPG")

    def test_status_reports_encoder_and_queue_progress(self):
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        self.assertTrue(self.recorder.start(image.shape))
        self.assertTrue(self.recorder.submit(image, image))
        status = wait_until(
            lambda: self.recorder.status()
            if self.recorder.status()["paired_frame_count"] == 1 else None
        )
        self.assertEqual(status["encoder"], "MJPG")
        self.assertEqual(status["submitted_frame_count"], 1)
        self.assertEqual(status["queue_capacity"], 8)

    def test_recent_capture_rate_is_available_for_video_encoding(self):
        frames = LatestStereoFrame()
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        frames.update(image, image, monotonic_time=10.0)
        frames.update(image, image, monotonic_time=10.05)
        frames.update(image, image, monotonic_time=10.10)
        self.assertAlmostEqual(frames.capture_fps(60.0), 20.0)

    def test_state_rejects_a_second_start_and_second_stop(self):
        shape = (12, 16, 3)
        self.assertTrue(self.recorder.start(shape))
        self.assertFalse(self.recorder.start(shape))
        self.assertTrue(self.recorder.stop_async())
        self.assertFalse(self.recorder.stop_async())
        wait_until(lambda: self.recorder.status()["state"] == "idle")


class RecordingWebApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.recorder = StereoRecorder(StereoRecordingConfig(
            data_root=self.temporary.name, camera_fps=5.0
        ))
        self.frames = LatestStereoFrame()
        shape = (12, 16, 3)
        image = np.zeros(shape, dtype=np.uint8)
        self.frames.update(image, image)
        self.assertTrue(self.recorder.start(shape))
        self.assertTrue(self.recorder.submit(image, image))
        self.assertTrue(self.recorder.stop_async())
        saved = wait_until(
            lambda: self.recorder.status()
            if self.recorder.status()["state"] == "idle" else None
        )
        self.saved_batch = saved["batch"]
        self.library = RecordingLibrary(self.temporary.name)
        runtime = SimpleNamespace(
            recorder=self.recorder,
            frames=self.frames,
            library=self.library,
            start_recording=lambda: self.recorder.start(shape),
        )
        handler = make_record_handler(runtime)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.recorder.shutdown()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.library.close()
        self.temporary.cleanup()

    def post(self, path):
        return urlopen(Request(self.base_url + path, method="POST"), timeout=2)

    def test_page_and_recording_actions_are_available(self):
        page = urlopen(self.base_url + "/", timeout=2).read().decode("utf-8")
        self.assertIn("Babybot Record mode", page)
        self.assertIn("Saved recording inspector", page)
        self.assertIn("/paired-frame/", page)
        self.assertIn("stopButton.onclick=", page)
        self.assertNotIn("stop.onclick=", page)
        paired = urlopen(
            f"{self.base_url}/paired-frame/{self.saved_batch}/0.jpg", timeout=2
        ).read()
        image = cv2.imdecode(np.frombuffer(paired, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape[:2], (12, 32))
        with self.assertRaises(HTTPError) as missing:
            urlopen(
                f"{self.base_url}/paired-frame/{self.saved_batch}/99.jpg", timeout=2
            )
        self.assertEqual(missing.exception.code, 404)
        self.assertEqual(self.post("/action/start").status, 202)
        status = json.loads(urlopen(
            self.base_url + "/status.json", timeout=2
        ).read())
        self.assertTrue(status["recording"])
        with self.assertRaises(HTTPError) as context:
            self.post("/action/start")
        self.assertEqual(context.exception.code, 409)
        self.assertEqual(self.post("/action/stop").status, 202)


if __name__ == "__main__":
    unittest.main()
