"""Babybot runtime: real-time observation and event-driven visual cognition."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import html
import logging
import multiprocessing
import os
import signal
import threading
import time
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from attention import Attention
from conscious import Conscious, ConsciousObject, crop_window, feature_similarity, visual_feature
from memory import MemoryStore
from observation import Observation
from perception import Perception


LOGGER = logging.getLogger("babybot")
Window = Tuple[int, int, int, int]


def lower_attention_process_priority():
    cv2.setNumThreads(1)
    try:
        os.nice(5)
    except (AttributeError, OSError):
        pass


def locate_template(image, template):
    """Find a template while allowing independent width and height changes."""
    best = {"window": None, "similarity": 0.0}
    ih, iw = image.shape[:2]
    th, tw = template.shape[:2]
    for sx in (0.55, 0.7, 0.85, 1.0, 1.2, 1.45, 1.75):
        for sy in (0.55, 0.7, 0.85, 1.0, 1.2, 1.45, 1.75):
            width, height = max(8, round(tw * sx)), max(8, round(th * sy))
            if width > iw or height > ih:
                continue
            resized = cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA)
            result = cv2.matchTemplate(image, resized, cv2.TM_SQDIFF_NORMED)
            minimum, _, location, _ = cv2.minMaxLoc(result)
            similarity = float(np.clip(1.0 - minimum, 0.0, 1.0))
            if similarity > best["similarity"]:
                best = {"window": (location[0], location[1], width, height), "similarity": similarity}
    return best


def calculate_stereo_attention(perception: Perception, mode: str, payload=None):
    """Process-safe, atomic stereo attention calculation."""
    started = time.perf_counter()
    if mode == "conscious":
        return {
            "mode": mode,
            "left": locate_template(perception.left, payload["left_template"]),
            "right": locate_template(perception.right, payload["right_template"]),
            "object_id": payload["object_id"],
            "elapsed_time": time.perf_counter() - started,
        }
    left = Attention(perception, eye="left", verbose=False)
    right = Attention(perception, eye="right", verbose=False)
    return {
        "mode": "default",
        "left": [item.copy() for item in left.candidates[:5]],
        "right": [item.copy() for item in right.candidates[:5]],
        "elapsed_time": time.perf_counter() - started,
    }


@dataclass(frozen=True)
class RuntimeConfig:
    left_camera: int = 0
    right_camera: int = 2
    width: int = 1280
    height: int = 800
    camera_fps: int = 60
    warmup_seconds: float = 6.0
    retry_delay: float = 0.1
    web_host: str = "127.0.0.1"
    web_port: int = 8080
    jpeg_quality: int = 80
    perception_width: int = 320
    perception_height: int = 200
    observation_preview_fps: float = 20.0
    memory_root: str = "memory"


class PreviewStore:
    def __init__(self):
        self._lock, self._images, self._identifiers = threading.Lock(), {}, {}

    def update(self, kind, left, right, left_candidates, right_candidates, identifier, jpeg_quality):
        if kind not in ("observation", "perception"):
            raise ValueError("Preview kind must be observation or perception")
        encoded = (encode_preview(left, left_candidates, jpeg_quality),
                   encode_preview(right, right_candidates, jpeg_quality))
        with self._lock:
            self._images[(kind, "left")], self._images[(kind, "right")] = encoded
            self._identifiers[kind] = int(identifier)

    def get(self, kind, eye):
        with self._lock:
            return self._images.get((kind, eye)), self._identifiers.get(kind, -1)


class LatestStereoFrame:
    def __init__(self):
        self._lock, self._left, self._right, self._version = threading.Lock(), None, None, 0

    def update(self, left, right):
        with self._lock:
            self._left, self._right, self._version = left, right, self._version + 1

    def snapshot(self, copy=True):
        with self._lock:
            if self._left is None:
                return None
            return ((self._left.copy() if copy else self._left),
                    (self._right.copy() if copy else self._right), self._version)


def encode_preview(image, candidates, quality):
    annotated = image.copy()
    colors = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0)]
    for rank, candidate in enumerate(candidates[:5], 1):
        x, y, width, height = candidate["window"]
        color = colors[(rank - 1) % len(colors)]
        cv2.rectangle(annotated, (x, y), (x + width, y + height), color, 4 if rank == 1 else 2)
        score = candidate.get("attention_score", candidate.get("score", 0.0))
        cv2.putText(annotated, f"#{rank} A:{score:.2f}", (x, max(22, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to encode preview image")
    return encoded.tobytes()


def make_request_handler(previews):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                body = ("<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
                        "<title>Babybot</title><style>body{background:#111;color:#eee;font-family:sans-serif;margin:20px}"
                        ".eyes{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}"
                        "img{width:100%;height:auto;background:#222}h1,h2{font-weight:500}</style></head><body>"
                        "<h1>Babybot</h1><h2>Raw observation - 20 Hz - no attention overlay</h2><div class='eyes'>"
                        "<section><h3>Left eye</h3><img id='observation-left'></section><section><h3>Right eye</h3><img id='observation-right'></section></div>"
                        "<h2>Perception - event driven - 320x200</h2><div class='eyes'><section><h3>Left eye</h3><img id='perception-left'></section>"
                        "<section><h3>Right eye</h3><img id='perception-right'></section></div><script>"
                        "function refresh(k,d){let p=2,l=document.getElementById(k+'-left'),r=document.getElementById(k+'-right');"
                        "const done=()=>{if(--p===0)setTimeout(()=>refresh(k,d),d)};l.onload=l.onerror=done;r.onload=r.onerror=done;"
                        "let t=Date.now();l.src='/frame/'+k+'/left.jpg?t='+t;r.src='/frame/'+k+'/right.jpg?t='+t}"
                        "refresh('observation',50);refresh('perception',100)</script></body></html>").encode("utf-8")
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
            parts = path.strip("/").split("/")
            if len(parts) == 3 and parts[0] == "frame" and parts[1] in ("observation", "perception") and parts[2] in ("left.jpg", "right.jpg"):
                image, identifier = previews.get(parts[1], "left" if parts[2] == "left.jpg" else "right")
                if image is None:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "Waiting for first frame"); return
                self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("Cache-Control", "no-store")
                self.send_header("X-Frame-Id", str(identifier)); self.send_header("Content-Length", str(len(image)))
                self.end_headers(); self.wfile.write(image); return
            self.send_error(HTTPStatus.NOT_FOUND, html.escape(path))

        def log_message(self, message, *args):
            LOGGER.debug("Web client: " + message, *args)
    return Handler


class StereoCognition:
    """Owns the confirmed default/conscious state transition."""
    def __init__(self, memory):
        self.conscious = Conscious(capacity=3)
        self.memory = memory

    @staticmethod
    def _pair_compatible(left, right, image_shape):
        lx, ly, lw, lh = left["window"]; rx, ry, rw, rh = right["window"]
        vertical = abs((ly + lh / 2) - (ry + rh / 2)) / image_shape[0]
        area_ratio = max(lw * lh, rw * rh) / max(1, min(lw * lh, rw * rh))
        return vertical <= .20 and area_ratio <= 2.5

    def consume_default(self, perception, result):
        pairs = []
        for left in result["left"]:
            for right in result["right"]:
                if not self._pair_compatible(left, right, perception.left.shape):
                    continue
                lf, rf = visual_feature(perception.left, left["window"]), visual_feature(perception.right, right["window"])
                similarity = feature_similarity(lf, rf)
                if similarity < .60:
                    continue
                feature = (lf + rf) / 2
                center = (left.get("center", 0) + right.get("center", 0)) / 2
                score = (left["score"] + right["score"]) / 2 + .15 * similarity + .05 * center
                pairs.append((score, feature, left, right, similarity))
        pairs.sort(key=lambda item: item[0], reverse=True)
        for _, feature, left, right, similarity in pairs:
            familiar, _ = self.memory.find_similar(feature)
            if familiar is not None:
                continue
            item = ConsciousObject(str(uuid.uuid4()), crop_window(perception.left, left["window"]),
                                   crop_window(perception.right, right["window"]), feature,
                                   left["window"], right["window"], time.time())
            self.conscious.add(item)
            return ([left], [right])
        return ([], [])

    def consume_conscious(self, perception, result):
        item = self.conscious.active()
        if item is None or result.get("object_id") != item.object_id:
            return [], []
        left, right = result["left"], result["right"]
        if (left["window"] is None or right["window"] is None or
                left["similarity"] < .68 or right["similarity"] < .68 or
                not self._pair_compatible(left, right, perception.left.shape)):
            item.missed(); return [], []
        similarity = (left["similarity"] + right["similarity"]) / 2
        item.matched(left["window"], right["window"], similarity)
        boxes = ([{"window": left["window"], "score": similarity}],
                 [{"window": right["window"], "score": similarity}])
        lf, rf = visual_feature(perception.left, left["window"]), visual_feature(perception.right, right["window"])
        feature = (lf + rf) / 2
        known, _ = self.memory.find_similar(feature)
        if item.consecutive_matches >= 3 and not self.memory.learning_stopped:
            lx, _, lw, lh = left["window"]; rx, _, rw, rh = right["window"]
            metadata = {"timestamp": time.time(), "disparity_pixels": abs((lx + lw / 2) - (rx + rw / 2)),
                        "left_window": left["window"], "right_window": right["window"],
                        "pixel_size": [(lw + rw) / 2, (lh + rh) / 2], "calibrated": False,
                        "distance": None, "physical_size": None}
            memory_id, _, _ = self.memory.learn(crop_window(perception.left, left["window"]),
                                                 crop_window(perception.right, right["window"]), feature, metadata)
            item.memory_id = memory_id
            known = memory_id is not None
        if known is not None or item.memory_id is not None or self.memory.learning_stopped:
            item.reduce_curiosity(.2)
            if item.curiosity <= .2:
                self.conscious.remove(item.object_id)
        return boxes


class BabybotRuntime:
    def __init__(self, config):
        self.config, self.previews, self.latest_frames = config, PreviewStore(), LatestStereoFrame()
        self.stop_event = threading.Event(); self.left_camera = self.right_camera = None
        self.web_server = self.web_thread = self.capture_thread = self.observation_preview_thread = None
        self.cognition = StereoCognition(MemoryStore(config.memory_root))
        self.attention_pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"), initializer=lower_attention_process_priority)

    def run(self):
        self._start_web_server()
        try:
            self._open_cameras_until_ready(); self._warm_up()
            self.capture_thread = threading.Thread(target=self._camera_loop, daemon=True)
            self.observation_preview_thread = threading.Thread(target=self._observation_preview_loop, daemon=True)
            self.capture_thread.start(); self.observation_preview_thread.start(); self._perception_loop()
        finally: self.shutdown()

    def request_stop(self, *_): LOGGER.info("Stop requested"); self.stop_event.set()

    def _open_camera(self, index):
        camera = cv2.VideoCapture(index, cv2.CAP_V4L2); camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width); camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        camera.set(cv2.CAP_PROP_FPS, self.config.camera_fps); camera.set(cv2.CAP_PROP_BUFFERSIZE, 1); return camera

    def _open_cameras_until_ready(self):
        last_log = 0.
        while not self.stop_event.is_set():
            self._release_cameras(); self.left_camera = self._open_camera(self.config.left_camera); self.right_camera = self._open_camera(self.config.right_camera)
            if self.left_camera.isOpened() and self.right_camera.isOpened(): LOGGER.info("Both cameras opened"); return
            if time.monotonic() - last_log >= 5: LOGGER.error("Camera open failed; retrying"); last_log = time.monotonic()
            self.stop_event.wait(self.config.retry_delay)
        raise InterruptedError("Stopped before cameras opened")

    def _capture_pair(self):
        if not self.left_camera.grab() or not self.right_camera.grab(): return None
        lok, left = self.left_camera.retrieve(); rok, right = self.right_camera.retrieve()
        return (left, right) if lok and rok and left is not None and right is not None else None

    def _warm_up(self):
        LOGGER.info("Warming cameras for %.1f seconds", self.config.warmup_seconds); deadline = time.monotonic() + self.config.warmup_seconds
        while time.monotonic() < deadline and not self.stop_event.is_set():
            if self._capture_pair() is None: self.stop_event.wait(self.config.retry_delay)
        LOGGER.info("Camera warm-up complete")

    def _camera_loop(self):
        last_log = 0.
        while not self.stop_event.is_set():
            pair = self._capture_pair()
            if pair is not None: self.latest_frames.update(*pair); continue
            if time.monotonic() - last_log >= 2: LOGGER.error("Stereo frame failed; retrying"); last_log = time.monotonic()
            self.stop_event.wait(self.config.retry_delay)

    def _perception_loop(self):
        perception_id = 0
        while not self.stop_event.is_set():
            snapshot = self.latest_frames.snapshot()
            if snapshot is None: self.stop_event.wait(.01); continue
            left, right, _ = snapshot; now = time.monotonic()
            observation = Observation.from_frames(perception_id, time.time(), now, left, right)
            perception = Perception.from_observation(observation, self.config.perception_width, self.config.perception_height)
            active = self.cognition.conscious.active(); mode = "conscious" if active else "default"
            payload = active.worker_payload() if active else None
            future = self.attention_pool.submit(calculate_stereo_attention, perception, mode, payload)
            try:
                while not self.stop_event.is_set():
                    try: result = future.result(timeout=.1); break
                    except TimeoutError: continue
                else: future.cancel(); break
                boxes = (self.cognition.consume_conscious(perception, result) if mode == "conscious"
                         else self.cognition.consume_default(perception, result))
                self.previews.update("perception", perception.left, perception.right, *boxes, perception_id, self.config.jpeg_quality)
                LOGGER.info("Perception %d refreshed (mode=%s, %.0fms, conscious=%d, memory=%d%s)", perception_id, mode,
                            result["elapsed_time"] * 1000, len(self.cognition.conscious), self.cognition.memory.object_count,
                            ", learning stopped" if self.cognition.memory.learning_stopped else "")
                perception_id += 1
            except Exception: LOGGER.exception("Stereo attention calculation failed")

    def _observation_preview_loop(self):
        interval = 1 / self.config.observation_preview_fps
        while not self.stop_event.is_set():
            started = time.monotonic(); snapshot = self.latest_frames.snapshot()
            if snapshot is not None:
                left, right, version = snapshot
                self.previews.update("observation", left, right, [], [], version, self.config.jpeg_quality)
            self.stop_event.wait(max(0., interval - (time.monotonic() - started)))

    def _start_web_server(self):
        self.web_server = ThreadingHTTPServer((self.config.web_host, self.config.web_port), make_request_handler(self.previews))
        self.web_thread = threading.Thread(target=self.web_server.serve_forever, daemon=True); self.web_thread.start()
        LOGGER.info("Preview listens locally at http://127.0.0.1:%d; use an SSH port forward", self.config.web_port)

    def _release_cameras(self):
        for camera in (self.left_camera, self.right_camera):
            if camera is not None: camera.release()
        self.left_camera = self.right_camera = None

    def shutdown(self):
        self.stop_event.set(); self.attention_pool.shutdown(wait=True, cancel_futures=True)
        for worker in (self.capture_thread, self.observation_preview_thread):
            if worker is not None and worker is not threading.current_thread(): worker.join(timeout=2)
        self._release_cameras()
        if self.web_server is not None: self.web_server.shutdown(); self.web_server.server_close(); self.web_server = None
        if self.web_thread is not None and self.web_thread is not threading.current_thread(): self.web_thread.join(timeout=2)
        LOGGER.info("Babybot stopped cleanly")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Babybot vision loop")
    parser.add_argument("--left-camera", type=int, default=0); parser.add_argument("--right-camera", type=int, default=2); parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def main():
    args = parse_args(); logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    runtime = BabybotRuntime(RuntimeConfig(left_camera=args.left_camera, right_camera=args.right_camera, web_port=args.port))
    signal.signal(signal.SIGINT, runtime.request_stop); signal.signal(signal.SIGTERM, runtime.request_stop)
    try: runtime.run()
    except InterruptedError: runtime.shutdown()
    except Exception: LOGGER.exception("Babybot stopped because of an unexpected error"); runtime.shutdown(); return 1
    return 0


if __name__ == "__main__": raise SystemExit(main())
