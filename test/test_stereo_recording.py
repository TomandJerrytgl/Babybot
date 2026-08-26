import json
from pathlib import Path
import tempfile
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer
import threading

import cv2
import numpy as np

from main import PreviewStore, make_request_handler
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
        for index in range(3):
            left[:] = index * 30
            right[:] = index * 30 + 5
            self.assertTrue(self.recorder.submit(
                left, right, timestamp_ns=1000 + index, monotonic_ns=2000 + index,
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
        self.assertTrue((batch / "frames.zip").is_file())
        self.assertTrue(any((batch / "videos").glob("left.*")))
        self.assertTrue(any((batch / "videos").glob("right.*")))
        metadata = json.loads((batch / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["schema"], "babybot.stereo-recording/v1")
        self.assertEqual(metadata["paired_frame_count"], 3)
        self.assertEqual(metadata["pairing"], "same grab/retrieve capture cycle")
        self.assertEqual(metadata["sync_delta_max_ns"], 102)

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
        shape = (12, 16, 3)
        handler = make_request_handler(
            PreviewStore(),
            recording_status=self.recorder.status,
            start_recording=lambda: self.recorder.start(shape),
            stop_recording=self.recorder.stop_async,
            retry_recording_upload=self.recorder.retry_upload,
        )
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.recorder.shutdown()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temporary.cleanup()

    def post(self, path):
        return urlopen(Request(self.base_url + path, method="POST"), timeout=2)

    def test_page_and_recording_actions_are_available(self):
        page = urlopen(self.base_url + "/", timeout=2).read().decode("utf-8")
        self.assertIn("Stereo training recorder", page)
        self.assertEqual(self.post("/action/record/start").status, 202)
        status = json.loads(urlopen(
            self.base_url + "/status/recording.json", timeout=2
        ).read())
        self.assertTrue(status["recording"])
        with self.assertRaises(HTTPError) as context:
            self.post("/action/record/start")
        self.assertEqual(context.exception.code, 409)
        self.assertEqual(self.post("/action/record/stop").status, 202)


if __name__ == "__main__":
    unittest.main()
