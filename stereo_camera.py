"""Shared stereo-camera primitives for Awake and Record modes."""

from __future__ import annotations

from collections import deque
import logging
import threading
import time

import cv2


LOGGER = logging.getLogger("babybot.camera")


class LatestStereoFrame:
    """Capacity-one stereo buffer with a recent capture-rate estimate."""

    def __init__(self):
        self._lock = threading.Lock()
        self._left = None
        self._right = None
        self._version = 0
        self._capture_times = deque(maxlen=120)

    def update(self, left, right, monotonic_time=None):
        with self._lock:
            self._left = left
            self._right = right
            self._version += 1
            self._capture_times.append(
                time.monotonic() if monotonic_time is None else float(monotonic_time)
            )

    def snapshot(self, copy=True):
        with self._lock:
            if self._left is None:
                return None
            left = self._left.copy() if copy else self._left
            right = self._right.copy() if copy else self._right
            return left, right, self._version

    def capture_fps(self, fallback):
        with self._lock:
            if len(self._capture_times) < 2:
                return float(fallback)
            elapsed = self._capture_times[-1] - self._capture_times[0]
            return (
                (len(self._capture_times) - 1) / elapsed
                if elapsed > 0 else float(fallback)
            )


class StereoCamera:
    """Own two OpenCV captures and return one software-synchronized pair."""

    def __init__(self, left_index=0, right_index=2, width=1280, height=800,
                 fps=60.0, retry_delay=0.1):
        self.left_index = int(left_index)
        self.right_index = int(right_index)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.retry_delay = float(retry_delay)
        self.left = None
        self.right = None

    def open_until_ready(self, stop_event):
        last_log = 0.0
        while not stop_event.is_set():
            self.release()
            self.left = self._open(self.left_index)
            self.right = self._open(self.right_index)
            if self.left.isOpened() and self.right.isOpened():
                LOGGER.info("Both cameras opened")
                return
            if time.monotonic() - last_log >= 5.0:
                LOGGER.error("Camera open failed; retrying")
                last_log = time.monotonic()
            stop_event.wait(self.retry_delay)
        raise InterruptedError("Stopped before cameras opened")

    def warm_up(self, seconds, stop_event):
        LOGGER.info("Warming cameras for %.1f seconds", seconds)
        deadline = time.monotonic() + float(seconds)
        while time.monotonic() < deadline and not stop_event.is_set():
            if self.capture_pair() is None:
                stop_event.wait(self.retry_delay)
        LOGGER.info("Camera warm-up complete")

    def capture_pair(self):
        if self.left is None or self.right is None or not self.left.grab():
            return None
        left_grab_ns = time.monotonic_ns()
        if not self.right.grab():
            return None
        right_grab_ns = time.monotonic_ns()
        left_ok, left = self.left.retrieve()
        right_ok, right = self.right.retrieve()
        if not left_ok or not right_ok or left is None or right is None:
            return None
        return left, right, abs(right_grab_ns - left_grab_ns)

    def release(self):
        for camera in (self.left, self.right):
            if camera is not None:
                camera.release()
        self.left = None
        self.right = None

    def _open(self, index):
        camera = cv2.VideoCapture(index, cv2.CAP_V4L2)
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        camera.set(cv2.CAP_PROP_FPS, self.fps)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return camera
