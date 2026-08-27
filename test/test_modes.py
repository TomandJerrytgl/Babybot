import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from dreaming import DreamingController, pair_attention_objects
from feature_training import FeatureTrainer
from shared_memory import SharedMemory, SOURCE_PRIORITY
from stereo_recording import StereoRecorder, StereoRecordingConfig


class SharedMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.memory = SharedMemory(self.temporary.name)
        self.image = np.full((20, 20, 3), 80, dtype=np.uint8)
        self.feature = np.linspace(0, 1, 99, dtype=np.float32)

    def tearDown(self):
        self.temporary.cleanup()

    def test_memory_has_no_capacity_limit_and_awake_priority_is_preserved(self):
        identifiers = [
            self.memory.add_object(
                self.image, self.image, self.feature + index,
                {"timestamp": index}, source_mode="dreaming",
            ) for index in range(25)
        ]
        self.assertEqual(self.memory.counts()["objects"], 25)
        awake = self.memory.add_object(
            self.image, self.image, self.feature,
            {"timestamp": 100}, source_mode="awake",
        )
        self.memory.add_sample(
            awake, self.image, self.image, self.feature,
            {"timestamp": 101}, source_mode="dreaming",
        )
        with self.memory.connection() as connection:
            priority = connection.execute(
                "SELECT source_priority FROM objects WHERE object_id=?", (awake,)
            ).fetchone()[0]
        self.assertEqual(priority, SOURCE_PRIORITY["awake"])
        self.assertEqual(len(identifiers), 25)

    def test_user_decisions_are_persistent(self):
        first = self.memory.add_object(self.image, None, self.feature, {}, eye_mode="left")
        second = self.memory.add_object(None, self.image, self.feature + 1, {}, eye_mode="right")
        self.memory.record_decision(first, second, "different")
        self.assertEqual(self.memory.counts()["decisions"], 1)


class DreamingFrameworkTests(unittest.TestCase):
    def test_pairing_retains_stereo_and_single_eye_objects(self):
        result = {
            "left": [{"window": (10, 10, 20, 20)}, {"window": (70, 70, 10, 10)}],
            "right": [{"window": (12, 11, 21, 20)}, {"window": (5, 90, 8, 8)}],
        }
        objects = pair_attention_objects(None, result)
        self.assertEqual(len(objects), 3)
        self.assertEqual({item["eye_mode"] for item in objects}, {"stereo", "left", "right"})

    def test_neural_training_is_explicitly_unconfigured(self):
        status = FeatureTrainer().status()
        self.assertFalse(status["configured"])
        self.assertIn("not configured", status["message"])

    def test_dreaming_processes_video_without_camera_hardware(self):
        with tempfile.TemporaryDirectory() as root:
            recordings = Path(root) / "recordings"
            recorder = StereoRecorder(StereoRecordingConfig(
                data_root=str(recordings), camera_fps=10,
            ))
            image = np.zeros((24, 32, 3), dtype=np.uint8)
            self.assertTrue(recorder.start(image.shape, video_fps=10))
            recorder.submit(image, image, timestamp_ns=1_000_000_000,
                            monotonic_ns=2_000_000_000)
            recorder.stop_async()
            deadline = time.monotonic() + 5
            while recorder.status()["state"] != "idle" and time.monotonic() < deadline:
                time.sleep(0.02)
            batch = recorder.status()["batch"]
            controller = DreamingController(recordings, Path(root) / "memory")
            self.assertTrue(controller.start(batch))
            deadline = time.monotonic() + 10
            while controller.status()["state"] not in ("complete", "error") and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(controller.status()["state"], "complete", controller.status())
            self.assertEqual(controller.status()["processed_frames"], 1)


if __name__ == "__main__":
    unittest.main()
