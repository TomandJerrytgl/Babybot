"""Babybot runtime: stereo capture, attention, short-term memory and web view."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import html
import logging
import signal
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Tuple

import cv2
import numpy as np

from attention import (
    Attention,
    AttentionValidator,
    ValidationResult,
    make_attention_reference,
)
from observation import Observation, ObservationBuffer


LOGGER = logging.getLogger("babybot")
Window = Tuple[int, int, int, int]


@dataclass(frozen=True)
class RuntimeConfig:
    left_camera: int = 0
    right_camera: int = 2
    width: int = 1280
    height: int = 800
    camera_fps: int = 60
    warmup_seconds: float = 6.0
    observation_interval: float = 0.1
    attention_interval: float = 0.5
    retention_seconds: float = 20.0
    retry_delay: float = 0.1
    web_host: str = "127.0.0.1"
    web_port: int = 8080
    jpeg_quality: int = 80
    preview_fps: float = 10.0


class PreviewStore:
    """Latest annotated frames shared safely with HTTP request threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._left: Optional[bytes] = None
        self._right: Optional[bytes] = None
        self._observation_id = -1

    def update(
        self,
        left: np.ndarray,
        right: np.ndarray,
        left_candidates: list,
        right_candidates: list,
        observation_id: int,
        jpeg_quality: int,
    ) -> None:
        left_jpeg = encode_preview(left, left_candidates, jpeg_quality)
        right_jpeg = encode_preview(right, right_candidates, jpeg_quality)
        with self._lock:
            self._left = left_jpeg
            self._right = right_jpeg
            self._observation_id = observation_id

    def get(self, eye: str) -> Tuple[Optional[bytes], int]:
        with self._lock:
            image = self._left if eye == "left" else self._right
            return image, self._observation_id


def encode_preview(image: np.ndarray, candidates: list, quality: int) -> bytes:
    annotated = image.copy()
    colors = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0)]
    for rank, candidate in enumerate(candidates[:5], start=1):
        x, y, width, height = candidate["window"]
        color = colors[(rank - 1) % len(colors)]
        cv2.rectangle(annotated, (x, y), (x + width, y + height), color, 4 if rank == 1 else 2)
        cv2.putText(
            annotated,
            f"#{rank} {candidate['score']:.3f}",
            (x, max(22, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to encode preview image")
    return encoded.tobytes()


def make_request_handler(previews: PreviewStore):
    class PreviewHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/":
                body = (
                    "<!doctype html><html><head><meta charset='utf-8'>"
                    "<meta name='viewport' content='width=device-width'>"
                    "<title>Babybot Attention</title>"
                    "<style>body{background:#111;color:#eee;font-family:sans-serif;margin:20px}"
                    ".eyes{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}"
                    "img{width:100%;height:auto;background:#222}h1,h2{font-weight:500}</style></head>"
                    "<body><h1>Babybot Attention</h1><div class='eyes'>"
                    "<section><h2>Left eye</h2><img id='left'></section>"
                    "<section><h2>Right eye</h2><img id='right'></section></div>"
                    "<script>function refresh(){let pending=2;const done=()=>{if(--pending===0)"
                    "setTimeout(refresh,100)};left.onload=left.onerror=done;right.onload=right.onerror=done;"
                    "const t=Date.now();left.src='/frame/left.jpg?t='+t;right.src='/frame/right.jpg?t='+t}"
                    "refresh()</script>"
                    "</body></html>"
                ).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path in ("/frame/left.jpg", "/frame/right.jpg"):
                eye = "left" if "left" in path else "right"
                image, observation_id = previews.get(eye)
                if image is None:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "Waiting for first observation")
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Observation-Id", str(observation_id))
                self.send_header("Content-Length", str(len(image)))
                self.end_headers()
                self.wfile.write(image)
                return

            self.send_error(HTTPStatus.NOT_FOUND, html.escape(path))

        def log_message(self, message: str, *args) -> None:
            LOGGER.debug("Web client: " + message, *args)

    return PreviewHandler


class EyePipeline:
    """Independent 2 Hz attention reference and 10 Hz validation for one eye."""

    def __init__(self, eye: str):
        self.eye = eye
        self.future: Optional[Future] = None
        self.source_observation: Optional[Observation] = None
        self.validator: Optional[AttentionValidator] = None
        self.candidate = None
        self.previous_attention_image = None

    def submit_attention(self, executor, observation: Observation) -> bool:
        if self.future is not None:
            return False
        self.source_observation = observation
        self.future = executor.submit(
            Attention,
            observation,
            eye=self.eye,
            previous_image=self.previous_attention_image,
            verbose=False,
        )
        self.previous_attention_image = getattr(observation, self.eye)
        return True

    def collect_attention(self) -> bool:
        if self.future is None or not self.future.done():
            return False
        future = self.future
        observation = self.source_observation
        self.future = None
        self.source_observation = None
        try:
            attention = future.result()
            if observation is None:
                return False
            reference = make_attention_reference(observation, self.eye, attention)
            if reference is None:
                self.validator = None
                self.candidate = None
                return False
            self.validator = AttentionValidator(reference)
            self.candidate = None
            LOGGER.info(
                "%s attention replaced (observation=%d, age=%.0fms, source=%s, %.0fms)",
                self.eye, observation.observation_id,
                (time.monotonic() - observation.monotonic_timestamp) * 1000,
                attention.focus_source, attention.elapsed_time * 1000,
            )
            return True
        except Exception:
            LOGGER.exception("%s attention calculation failed", self.eye)
            return False

    def validate(self, image) -> Optional[ValidationResult]:
        if self.validator is None:
            self.candidate = None
            return None
        result = self.validator.validate(image)
        if result.window is None:
            self.candidate = None
        else:
            self.candidate = {
                "window": result.window,
                "score": result.similarity,
                "source": "validated",
            }
        return result

    def candidates(self):
        return [self.candidate] if self.candidate is not None else []


class BabybotRuntime:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.observations = ObservationBuffer(config.retention_seconds)
        self.previews = PreviewStore()
        self.stop_event = threading.Event()
        self.left_camera = None
        self.right_camera = None
        self.web_server: Optional[ThreadingHTTPServer] = None
        self.web_thread: Optional[threading.Thread] = None
        self.left_pipeline = EyePipeline("left")
        self.right_pipeline = EyePipeline("right")
        self.attention_pool = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="attention",
        )

    def run(self) -> None:
        self._start_web_server()
        try:
            self._open_cameras_until_ready()
            self._warm_up()
            self._capture_loop()
        finally:
            self.shutdown()

    def request_stop(self, *_args) -> None:
        LOGGER.info("Stop requested")
        self.stop_event.set()

    def _open_camera(self, index: int):
        camera = cv2.VideoCapture(index, cv2.CAP_V4L2)
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        camera.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return camera

    def _open_cameras_until_ready(self) -> None:
        last_log = 0.0
        while not self.stop_event.is_set():
            self._release_cameras()
            self.left_camera = self._open_camera(self.config.left_camera)
            self.right_camera = self._open_camera(self.config.right_camera)
            if self.left_camera.isOpened() and self.right_camera.isOpened():
                LOGGER.info("Both cameras opened")
                return
            now = time.monotonic()
            if now - last_log >= 5.0:
                LOGGER.error("Camera open failed; retrying")
                last_log = now
            self.stop_event.wait(self.config.retry_delay)
        raise InterruptedError("Stopped before cameras opened")

    def _capture_pair(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self.left_camera.grab() or not self.right_camera.grab():
            return None
        left_ok, left = self.left_camera.retrieve()
        right_ok, right = self.right_camera.retrieve()
        if not left_ok or not right_ok or left is None or right is None:
            return None
        return left, right

    def _warm_up(self) -> None:
        LOGGER.info("Warming cameras for %.1f seconds", self.config.warmup_seconds)
        deadline = time.monotonic() + self.config.warmup_seconds
        while time.monotonic() < deadline and not self.stop_event.is_set():
            if self._capture_pair() is None:
                LOGGER.warning("Frame capture failed during warm-up")
                self.stop_event.wait(self.config.retry_delay)
        LOGGER.info("Camera warm-up complete")

    def _capture_loop(self) -> None:
        observation_id = 0
        next_observation = time.monotonic()
        next_attention = time.monotonic()
        last_failure_log = 0.0
        while not self.stop_event.is_set():
            pair = self._capture_pair()
            if pair is None:
                now = time.monotonic()
                if now - last_failure_log >= 2.0:
                    LOGGER.error("Stereo frame failed; retrying")
                    last_failure_log = now
                self.stop_event.wait(self.config.retry_delay)
                continue

            left, right = pair
            now = time.monotonic()

            if now >= next_observation:
                observation = Observation(
                    observation_id=observation_id,
                    timestamp=time.time(),
                    monotonic_timestamp=now,
                    left=left.copy(),
                    right=right.copy(),
                )
                self.observations.append(observation)
                attention_due = now >= next_attention
                left_submitted = right_submitted = False
                if attention_due:
                    left_submitted = self.left_pipeline.submit_attention(
                        self.attention_pool, observation
                    )
                    right_submitted = self.right_pipeline.submit_attention(
                        self.attention_pool, observation
                    )
                    next_attention = max(
                        next_attention + self.config.attention_interval,
                        now,
                    )

                # A new result immediately replaces the old reference. Neither
                # polling call waits unless its future already reports done.
                self.left_pipeline.collect_attention()
                self.right_pipeline.collect_attention()
                left_result = self.left_pipeline.validate(left)
                right_result = self.right_pipeline.validate(right)
                self.previews.update(
                    left,
                    right,
                    self.left_pipeline.candidates(),
                    self.right_pipeline.candidates(),
                    observation_id,
                    self.config.jpeg_quality,
                )
                LOGGER.debug(
                    "Observation %d stored (buffer=%d, attention left=%s right=%s, "
                    "validation left=%s right=%s)",
                    observation_id, len(self.observations), left_submitted,
                    right_submitted, self._validation_summary(left_result),
                    self._validation_summary(right_result),
                )
                observation_id += 1
                next_observation = max(
                    next_observation + self.config.observation_interval,
                    now,
                )

    @staticmethod
    def _validation_summary(result):
        if result is None:
            return "no-reference"
        state = "shown" if result.window is not None else "hidden"
        return f"{state}/{result.similarity:.3f}/{result.elapsed_time * 1000:.0f}ms"

    def _start_web_server(self) -> None:
        handler = make_request_handler(self.previews)
        self.web_server = ThreadingHTTPServer((self.config.web_host, self.config.web_port), handler)
        self.web_thread = threading.Thread(target=self.web_server.serve_forever, name="preview-web", daemon=True)
        self.web_thread.start()
        LOGGER.info(
            "Preview listens locally at http://127.0.0.1:%d; use an SSH port forward to view it",
            self.config.web_port,
        )

    def _release_cameras(self) -> None:
        for camera in (self.left_camera, self.right_camera):
            if camera is not None:
                camera.release()
        self.left_camera = None
        self.right_camera = None

    def shutdown(self) -> None:
        self.stop_event.set()
        self.attention_pool.shutdown(wait=True, cancel_futures=True)
        self._release_cameras()
        if self.web_server is not None:
            self.web_server.shutdown()
            self.web_server.server_close()
            self.web_server = None
        if self.web_thread is not None and self.web_thread is not threading.current_thread():
            self.web_thread.join(timeout=2.0)
        LOGGER.info("Babybot stopped cleanly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Babybot vision loop")
    parser.add_argument("--left-camera", type=int, default=0)
    parser.add_argument("--right-camera", type=int, default=2)
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    runtime = BabybotRuntime(RuntimeConfig(left_camera=args.left_camera, right_camera=args.right_camera, web_port=args.port))
    signal.signal(signal.SIGINT, runtime.request_stop)
    signal.signal(signal.SIGTERM, runtime.request_stop)
    try:
        runtime.run()
    except InterruptedError:
        runtime.shutdown()
    except Exception:
        LOGGER.exception("Babybot stopped because of an unexpected error")
        runtime.shutdown()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
