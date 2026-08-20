"""Babybot visual-front-end runtime: Observation -> Perception -> Attention."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from dataclasses import dataclass
import html
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import multiprocessing
import os
from pathlib import Path
import signal
import threading
import time
import traceback
from typing import Optional

import cv2
import numpy as np

from attention import Attention
from observation import Observation
from perception import Perception


LOGGER = logging.getLogger("babybot")
SCORE_FIELDS = (
    ("ranking_score", "ranking score"),
    ("objectness", "objectness"),
    ("boundary", "boundary fit"),
    ("contrast", "Lab surround contrast"),
    ("contrast_top", "Lab contrast — top"),
    ("contrast_bottom", "Lab contrast — bottom"),
    ("contrast_left", "Lab contrast — left"),
    ("contrast_right", "Lab contrast — right"),
    ("color", "vividness"),
    ("edge", "internal edges"),
    ("coherence", "internal coherence"),
    ("center", "center preference"),
)


def lower_attention_process_priority():
    """Keep expensive attention work behind camera and preview work."""
    cv2.setNumThreads(1)
    try:
        os.nice(5)
    except (AttributeError, OSError):
        pass


def calculate_attention_pair(perception: Perception, settings: dict):
    """Process-safe entry point that computes both eyes for one perception."""
    started = time.perf_counter()
    arguments = {
        "verbose": False,
        "objectness_threshold": settings["minimum_objectness"],
        "maximum_candidates": settings["maximum_candidates"],
        "partial_overlap_iou": settings["partial_overlap_iou"],
    }
    result = {"left": [], "right": [], "left_elapsed": 0.0, "right_elapsed": 0.0,
              "left_error": None, "right_error": None}
    for eye in ("left", "right"):
        eye_started = time.perf_counter()
        try:
            attention = Attention(perception, eye=eye, **arguments)
            result[eye] = [candidate.copy() for candidate in attention.candidates]
            result[f"{eye}_elapsed"] = attention.elapsed_time
        except Exception:
            result[f"{eye}_elapsed"] = time.perf_counter() - eye_started
            result[f"{eye}_error"] = traceback.format_exc()
    result["elapsed_time"] = time.perf_counter() - started
    return result


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
    minimum_objectness: float = 0.45
    maximum_candidates_per_eye: int = 10
    partial_overlap_iou: float = 0.30
    candidate_crop_scale: float = 1.10
    attention_report_path: str = "debug/attention_report.html"

    def attention_settings(self):
        return {
            "minimum_objectness": self.minimum_objectness,
            "maximum_candidates": self.maximum_candidates_per_eye,
            "partial_overlap_iou": self.partial_overlap_iou,
        }


class PreviewStore:
    """Thread-safe latest-only JPEG and diagnostic storage."""

    def __init__(self):
        self._lock = threading.Lock()
        self._images = {}
        self._identifiers = {}
        self._candidate_images = {}
        self._attention_status = {"ready": False, "message": "Waiting for first attention result"}

    def update(self, kind, left, right, left_candidates, right_candidates, identifier, jpeg_quality):
        if kind not in ("observation", "perception"):
            raise ValueError("Unknown preview kind")
        encoded = (
            encode_preview(left, left_candidates, jpeg_quality),
            encode_preview(right, right_candidates, jpeg_quality),
        )
        with self._lock:
            self._images[(kind, "left")], self._images[(kind, "right")] = encoded
            self._identifiers[kind] = int(identifier)

    def update_attention(self, perception, result, identifier, jpeg_quality, crop_scale):
        left_jpeg = encode_preview(perception.left, result["left"], jpeg_quality)
        right_jpeg = encode_preview(perception.right, result["right"], jpeg_quality)
        candidate_images = {}
        for eye in ("left", "right"):
            image = getattr(perception, eye)
            for candidate in result[eye]:
                crop, _ = make_candidate_crop(image, candidate, crop_scale)
                candidate_images[(eye, candidate["rank"])] = encode_jpeg(crop, jpeg_quality)
        status = {
            "ready": True,
            "perception_id": int(identifier),
            "observation_id": int(perception.observation_id),
            "timestamp": float(perception.timestamp),
            "left_elapsed_ms": round(result["left_elapsed"] * 1000, 1),
            "right_elapsed_ms": round(result["right_elapsed"] * 1000, 1),
            "total_elapsed_ms": round(result["elapsed_time"] * 1000, 1),
            "left_error": result.get("left_error"),
            "right_error": result.get("right_error"),
            "left": [candidate_details(item) for item in result["left"]],
            "right": [candidate_details(item) for item in result["right"]],
        }
        with self._lock:
            self._images[("perception", "left")] = left_jpeg
            self._images[("perception", "right")] = right_jpeg
            self._identifiers["perception"] = int(identifier)
            self._candidate_images = candidate_images
            self._attention_status = status

    def get(self, kind, eye):
        with self._lock:
            return self._images.get((kind, eye)), self._identifiers.get(kind, -1)

    def get_candidate(self, eye, rank):
        with self._lock:
            return self._candidate_images.get((eye, int(rank))), self._identifiers.get("perception", -1)

    def attention_status(self):
        with self._lock:
            return dict(self._attention_status)


class LatestStereoFrame:
    """Capacity-one stereo buffer; every successful capture replaces the old pair."""

    def __init__(self):
        self._lock = threading.Lock()
        self._left = None
        self._right = None
        self._version = 0

    def update(self, left, right):
        with self._lock:
            self._left = left
            self._right = right
            self._version += 1

    def snapshot(self, copy=True):
        with self._lock:
            if self._left is None:
                return None
            left = self._left.copy() if copy else self._left
            right = self._right.copy() if copy else self._right
            return left, right, self._version


def encode_jpeg(image, quality):
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode preview image")
    return encoded.tobytes()


def encode_preview(image, candidates, quality):
    annotated = image.copy()
    colors = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0)]
    for rank, candidate in enumerate(candidates[:10], start=1):
        x, y, width, height = candidate["window"]
        color = colors[(rank - 1) % len(colors)]
        cv2.rectangle(annotated, (x, y), (x + width, y + height), color, 3 if rank == 1 else 2)
        cv2.putText(
            annotated,
            f"#{candidate.get('rank', rank)} R:{candidate.get('ranking_score', candidate.get('score', 0.0)):.2f} O:{candidate.get('objectness', 0.0):.2f}",
            (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            color, 1, cv2.LINE_AA,
        )
    return encode_jpeg(annotated, quality)


def make_candidate_crop(image, candidate, scale=1.10):
    """Return a candidate crop about 10% larger and its box inside the crop."""
    x, y, width, height = candidate["window"]
    image_height, image_width = image.shape[:2]
    padding_x = int(round(width * (float(scale) - 1.0) / 2.0))
    padding_y = int(round(height * (float(scale) - 1.0) / 2.0))
    crop_x1 = max(0, x - padding_x)
    crop_y1 = max(0, y - padding_y)
    crop_x2 = min(image_width, x + width + padding_x)
    crop_y2 = min(image_height, y + height + padding_y)
    crop = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    relative = (x - crop_x1, y - crop_y1, width, height)
    rx, ry, rw, rh = relative
    cv2.rectangle(crop, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)
    cv2.putText(crop, f"#{candidate.get('rank', '?')}", (rx + 3, max(15, ry + 15)),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 255), 1, cv2.LINE_AA)
    return crop, relative


def candidate_details(candidate):
    details = {
        "rank": int(candidate.get("rank", 0)),
        "window": [int(value) for value in candidate["window"]],
        "area_fraction": round(float(candidate.get("area_fraction", 0.0)), 5),
    }
    for key, _label in SCORE_FIELDS:
        value = candidate.get(key)
        details[key] = None if value is None else round(float(value), 4)
    return details


def jpeg_data_uri(data):
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def write_attention_report(path, perception, result, identifier, jpeg_quality, crop_scale):
    """Atomically replace one self-contained HTML report for the latest result."""
    sections = []
    for eye, title in (("left", "Left eye"), ("right", "Right eye")):
        candidates = result[eye]
        full = encode_preview(getattr(perception, eye), candidates, jpeg_quality)
        cards = []
        for candidate in candidates:
            crop, _ = make_candidate_crop(getattr(perception, eye), candidate, crop_scale)
            details = candidate_details(candidate)
            score_lines = "".join(
                f"<tr><th>{html.escape(label)}</th><td>"
                f"{'—' if details[key] is None else f'{details[key]:.4f}'}</td></tr>"
                for key, label in SCORE_FIELDS
            )
            window = details["window"]
            cards.append(
                "<article class='card'>"
                f"<h3>#{details['rank']} — {window[0]}, {window[1]}, {window[2]}×{window[3]}</h3>"
                f"<img src='{jpeg_data_uri(encode_jpeg(crop, jpeg_quality))}'>"
                f"<table>{score_lines}<tr><th>area fraction</th><td>{details['area_fraction']:.5f}</td></tr></table>"
                "</article>"
            )
        sections.append(
            f"<section><h2>{title}: {len(candidates)} candidates</h2>"
            f"<img class='full' src='{jpeg_data_uri(full)}'><div class='cards'>{''.join(cards)}</div></section>"
        )
    document = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Babybot attention report</title>"
        "<style>body{font-family:sans-serif;background:#111;color:#eee;margin:24px}.full{max-width:900px;width:100%}"
        ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}.card{background:#222;padding:12px}"
        ".card img{width:100%;height:auto}table{width:100%;border-collapse:collapse}th{text-align:left}"
        "th,td{padding:3px;border-bottom:1px solid #444}</style></head><body>"
        f"<h1>Babybot attention report</h1><p>Perception {int(identifier)} | observation {int(perception.observation_id)} | "
        f"total {result['elapsed_time'] * 1000:.1f} ms | left {result['left_elapsed'] * 1000:.1f} ms | "
        f"right {result['right_elapsed'] * 1000:.1f} ms</p>{''.join(sections)}</body></html>"
    )
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(report_path)


def make_request_handler(previews, report_path="debug/attention_report.html"):
    class PreviewHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                body = (
                    "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
                    "<title>Babybot visual front end</title><style>body{background:#111;color:#eee;font-family:sans-serif;margin:20px}"
                    ".eyes{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}img{width:100%;height:auto;background:#222}"
                    ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}.card{background:#222;padding:10px}"
                    "pre{white-space:pre-wrap;font-size:12px}h1,h2{font-weight:500}</style></head><body><h1>Babybot visual front end</h1>"
                    "<h2>Raw observation — target 20 Hz — no overlay</h2><div class='eyes'><section><h3>Left eye</h3><img id='observation-left'></section>"
                    "<section><h3>Right eye</h3><img id='observation-right'></section></div><h2>Perception + Attention — event driven — 320×200</h2>"
                    "<p id='timing'>Waiting for first attention result</p><p><a href='/report/attention' target='_blank'>Open latest self-contained report</a></p>"
                    "<div class='eyes'><section><h3>Left eye</h3><img id='perception-left'></section><section><h3>Right eye</h3>"
                    "<img id='perception-right'></section></div><h2>Left candidates</h2><div id='left-candidates' class='cards'></div>"
                    "<h2>Right candidates</h2><div id='right-candidates' class='cards'></div><script>"
                    "function refresh(k,d){let p=2,l=document.getElementById(k+'-left'),r=document.getElementById(k+'-right');"
                    "const done=()=>{if(--p===0)setTimeout(()=>refresh(k,d),d)};l.onload=l.onerror=done;r.onload=r.onerror=done;"
                    "let t=Date.now();l.src='/frame/'+k+'/left.jpg?t='+t;r.src='/frame/'+k+'/right.jpg?t='+t}"
                    "function cards(eye,items,id){let root=document.getElementById(eye+'-candidates');root.replaceChildren();items.forEach(c=>{"
                    "let a=document.createElement('article');a.className='card';let h=document.createElement('h3');h.textContent='#'+c.rank+' window '+c.window.join(', ');"
                    "let i=document.createElement('img');i.src='/candidate/'+eye+'/'+c.rank+'.jpg?t='+id;let p=document.createElement('pre');"
                    "p.textContent=JSON.stringify(c,null,2);a.append(h,i,p);root.append(a)})}"
                    "async function status(){try{let r=await fetch('/status/attention.json?t='+Date.now()),s=await r.json();if(s.ready){"
                    "document.getElementById('timing').textContent='Perception '+s.perception_id+' | left '+s.left_elapsed_ms+' ms | right '+s.right_elapsed_ms+' ms | total '+s.total_elapsed_ms+' ms';"
                    "cards('left',s.left,s.perception_id);cards('right',s.right,s.perception_id)}}catch(e){}setTimeout(status,300)}"
                    "refresh('observation',50);refresh('perception',150);status()</script></body></html>"
                ).encode("utf-8")
                self._send_bytes(body, "text/html; charset=utf-8")
                return
            if path == "/status/attention.json":
                self._send_bytes(
                    json.dumps(previews.attention_status()).encode("utf-8"),
                    "application/json", no_store=True,
                )
                return
            if path == "/report/attention":
                file_path = Path(report_path)
                if not file_path.is_file():
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "Waiting for first attention report")
                    return
                self._send_bytes(file_path.read_bytes(), "text/html; charset=utf-8", no_store=True)
                return
            parts = path.strip("/").split("/")
            if len(parts) == 3 and parts[0] == "frame" and parts[1] in ("observation", "perception") and parts[2] in ("left.jpg", "right.jpg"):
                eye = "left" if parts[2] == "left.jpg" else "right"
                image, identifier = previews.get(parts[1], eye)
                self._send_image_or_wait(image, identifier)
                return
            if len(parts) == 3 and parts[0] == "candidate" and parts[1] in ("left", "right") and parts[2].endswith(".jpg"):
                try:
                    rank = int(parts[2][:-4])
                except ValueError:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                image, identifier = previews.get_candidate(parts[1], rank)
                self._send_image_or_wait(image, identifier)
                return
            self.send_error(HTTPStatus.NOT_FOUND, html.escape(path))

        def _send_image_or_wait(self, image, identifier):
            if image is None:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "Waiting for image")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Frame-Id", str(identifier))
            self.send_header("Content-Length", str(len(image)))
            self.end_headers()
            self.wfile.write(image)

        def _send_bytes(self, body, content_type, no_store=False):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            if no_store:
                self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, message, *args):
            LOGGER.debug("Web client: " + message, *args)

    return PreviewHandler


class BabybotRuntime:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.previews = PreviewStore()
        self.latest_frames = LatestStereoFrame()
        self.stop_event = threading.Event()
        self.left_camera = None
        self.right_camera = None
        self.web_server: Optional[ThreadingHTTPServer] = None
        self.web_thread: Optional[threading.Thread] = None
        self.capture_thread: Optional[threading.Thread] = None
        self.observation_preview_thread: Optional[threading.Thread] = None
        self.attention_pool = ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=lower_attention_process_priority,
        )

    def run(self):
        self._start_web_server()
        try:
            self._open_cameras_until_ready()
            self._warm_up()
            self.capture_thread = threading.Thread(target=self._camera_loop, name="camera-capture", daemon=True)
            self.observation_preview_thread = threading.Thread(
                target=self._observation_preview_loop, name="observation-preview", daemon=True
            )
            self.capture_thread.start()
            self.observation_preview_thread.start()
            self._perception_loop()
        finally:
            self.shutdown()

    def request_stop(self, *_args):
        LOGGER.info("Stop requested")
        self.stop_event.set()

    def _open_camera(self, index):
        camera = cv2.VideoCapture(index, cv2.CAP_V4L2)
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        camera.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return camera

    def _open_cameras_until_ready(self):
        last_log = 0.0
        while not self.stop_event.is_set():
            self._release_cameras()
            self.left_camera = self._open_camera(self.config.left_camera)
            self.right_camera = self._open_camera(self.config.right_camera)
            if self.left_camera.isOpened() and self.right_camera.isOpened():
                LOGGER.info("Both cameras opened")
                return
            if time.monotonic() - last_log >= 5.0:
                LOGGER.error("Camera open failed; retrying")
                last_log = time.monotonic()
            self.stop_event.wait(self.config.retry_delay)
        raise InterruptedError("Stopped before cameras opened")

    def _capture_pair(self):
        if not self.left_camera.grab() or not self.right_camera.grab():
            return None
        left_ok, left = self.left_camera.retrieve()
        right_ok, right = self.right_camera.retrieve()
        if not left_ok or not right_ok or left is None or right is None:
            return None
        return left, right

    def _warm_up(self):
        LOGGER.info("Warming cameras for %.1f seconds", self.config.warmup_seconds)
        deadline = time.monotonic() + self.config.warmup_seconds
        while time.monotonic() < deadline and not self.stop_event.is_set():
            if self._capture_pair() is None:
                self.stop_event.wait(self.config.retry_delay)
        LOGGER.info("Camera warm-up complete")

    def _camera_loop(self):
        last_log = 0.0
        while not self.stop_event.is_set():
            pair = self._capture_pair()
            if pair is not None:
                self.latest_frames.update(*pair)
                continue
            if time.monotonic() - last_log >= 2.0:
                LOGGER.error("Stereo frame failed; retrying")
                last_log = time.monotonic()
            self.stop_event.wait(self.config.retry_delay)

    def _perception_loop(self):
        perception_id = 0
        settings = self.config.attention_settings()
        while not self.stop_event.is_set():
            snapshot = self.latest_frames.snapshot()
            if snapshot is None:
                self.stop_event.wait(0.01)
                continue
            left, right, _version = snapshot
            observation = Observation.from_frames(
                perception_id, time.time(), time.monotonic(), left, right
            )
            perception = Perception.from_observation(
                observation, self.config.perception_width, self.config.perception_height
            )
            future = self.attention_pool.submit(calculate_attention_pair, perception, settings)
            try:
                while not self.stop_event.is_set():
                    try:
                        result = future.result(timeout=0.1)
                        break
                    except TimeoutError:
                        continue
                else:
                    future.cancel()
                    break
                for eye in ("left", "right"):
                    if result.get(f"{eye}_error"):
                        LOGGER.error("%s attention failed\n%s", eye, result[f"{eye}_error"])
                self.previews.update_attention(
                    perception, result, perception_id, self.config.jpeg_quality,
                    self.config.candidate_crop_scale,
                )
                try:
                    write_attention_report(
                        self.config.attention_report_path, perception, result,
                        perception_id, self.config.jpeg_quality,
                        self.config.candidate_crop_scale,
                    )
                except Exception:
                    LOGGER.exception("Attention report generation failed")
                LOGGER.info(
                    "Perception %d refreshed (left=%d right=%d, left=%.0fms right=%.0fms total=%.0fms)",
                    perception_id, len(result["left"]), len(result["right"]),
                    result["left_elapsed"] * 1000, result["right_elapsed"] * 1000,
                    result["elapsed_time"] * 1000,
                )
                perception_id += 1
            except Exception:
                LOGGER.exception("Attention calculation failed")

    def _observation_preview_loop(self):
        interval = 1.0 / self.config.observation_preview_fps
        while not self.stop_event.is_set():
            started = time.monotonic()
            snapshot = self.latest_frames.snapshot()
            if snapshot is not None:
                left, right, version = snapshot
                self.previews.update(
                    "observation", left, right, [], [], version, self.config.jpeg_quality
                )
            self.stop_event.wait(max(0.0, interval - (time.monotonic() - started)))

    def _start_web_server(self):
        handler = make_request_handler(self.previews, self.config.attention_report_path)
        self.web_server = ThreadingHTTPServer((self.config.web_host, self.config.web_port), handler)
        self.web_thread = threading.Thread(
            target=self.web_server.serve_forever, name="preview-web", daemon=True
        )
        self.web_thread.start()
        LOGGER.info(
            "Preview listens locally at http://127.0.0.1:%d; use an SSH port forward",
            self.config.web_port,
        )

    def _release_cameras(self):
        for camera in (self.left_camera, self.right_camera):
            if camera is not None:
                camera.release()
        self.left_camera = None
        self.right_camera = None

    def shutdown(self):
        self.stop_event.set()
        self.attention_pool.shutdown(wait=True, cancel_futures=True)
        for worker in (self.capture_thread, self.observation_preview_thread):
            if worker is not None and worker is not threading.current_thread():
                worker.join(timeout=2.0)
        self._release_cameras()
        if self.web_server is not None:
            self.web_server.shutdown()
            self.web_server.server_close()
            self.web_server = None
        if self.web_thread is not None and self.web_thread is not threading.current_thread():
            self.web_thread.join(timeout=2.0)
        LOGGER.info("Babybot stopped cleanly")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Babybot visual front end")
    parser.add_argument("--left-camera", type=int, default=0)
    parser.add_argument("--right-camera", type=int, default=2)
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    runtime = BabybotRuntime(RuntimeConfig(
        left_camera=args.left_camera,
        right_camera=args.right_camera,
        web_port=args.port,
    ))
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
