"""Hardware-only Record mode: capture, inspect, play, and upload stereo data."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import mimetypes
import os
from pathlib import Path
import re
import signal
import threading
import time
from urllib.parse import unquote

import cv2

from stereo_camera import LatestStereoFrame, StereoCamera
from stereo_recording import StereoRecorder, StereoRecordingConfig


LOGGER = logging.getLogger("babybot.record")
BATCH_PATTERN = re.compile(r"recording_[A-Za-z0-9_-]+$")


def encode_jpeg(image, quality=80):
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode Record preview")
    return data.tobytes()


class RecordingLibrary:
    def __init__(self, root):
        self.root = Path(root)
        self._cache = {}
        self._playback_lock = threading.Lock()
        self._playback_batch = None
        self._playback_captures = {}
        self._playback_next_frame = 0

    def list(self):
        self.root.mkdir(parents=True, exist_ok=True)
        return [self.inspect(path) for path in sorted(self.root.glob("recording_*"), reverse=True)
                if path.is_dir() and (path / "metadata.json").is_file()]

    def inspect(self, path):
        metadata_path = path / "metadata.json"
        signature = metadata_path.stat().st_mtime_ns
        cached = self._cache.get(path.name)
        if cached and cached[0] == signature:
            return cached[1]
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            videos = {}
            for item in metadata.get("videos", []):
                eye = item.get("eye")
                video_path = path / item.get("path", "")
                if eye not in ("left", "right") or not video_path.is_file():
                    continue
                capture = cv2.VideoCapture(str(video_path))
                try:
                    videos[eye] = {
                        "file": video_path.name,
                        "bytes": video_path.stat().st_size,
                        "decodable": bool(capture.isOpened()),
                        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                        "fps": round(float(capture.get(cv2.CAP_PROP_FPS)), 4),
                        "duration": round(
                            capture.get(cv2.CAP_PROP_FRAME_COUNT)
                            / max(capture.get(cv2.CAP_PROP_FPS), 1e-9), 4
                        ),
                    }
                finally:
                    capture.release()
            left, right = videos.get("left"), videos.get("right")
            result = {
                "batch": path.name,
                "schema": metadata.get("schema"),
                "paired_frame_count": metadata.get("paired_frame_count", 0),
                "capture_duration_seconds": metadata.get(
                    "capture_duration_seconds", metadata.get("duration_seconds", 0)
                ),
                "sync_delta_mean_ns": metadata.get("sync_delta_mean_ns"),
                "sync_delta_max_ns": metadata.get("sync_delta_max_ns"),
                "videos": videos,
                "valid": bool(left and right and left["decodable"] and right["decodable"]
                              and left["frames"] == right["frames"]),
                "error": None,
            }
        except Exception as error:
            result = {"batch": path.name, "valid": False, "error": str(error), "videos": {}}
        self._cache[path.name] = (signature, result)
        return result

    def video_path(self, batch, eye):
        if not BATCH_PATTERN.fullmatch(batch) or eye not in ("left", "right"):
            raise FileNotFoundError
        batch_path = (self.root / batch).resolve()
        if self.root.resolve() not in batch_path.parents:
            raise FileNotFoundError
        metadata = json.loads((batch_path / "metadata.json").read_text(encoding="utf-8"))
        for item in metadata.get("videos", []):
            if item.get("eye") == eye:
                path = (batch_path / item["path"]).resolve()
                if batch_path not in path.parents or not path.is_file():
                    break
                return path
        raise FileNotFoundError

    def paired_frame_jpeg(self, batch, frame_index, quality=85):
        """Decode one matched left/right frame and return a browser-safe JPEG."""
        frame_index = int(frame_index)
        if frame_index < 0:
            raise ValueError("Frame index must be non-negative")
        with self._playback_lock:
            if self._playback_batch != batch:
                self._close_playback_locked()
                captures = {
                    eye: cv2.VideoCapture(str(self.video_path(batch, eye)))
                    for eye in ("left", "right")
                }
                if not all(capture.isOpened() for capture in captures.values()):
                    for capture in captures.values():
                        capture.release()
                    raise ValueError("Saved stereo video cannot be decoded")
                self._playback_batch = batch
                self._playback_captures = captures
                self._playback_next_frame = 0
            if frame_index != self._playback_next_frame:
                for capture in self._playback_captures.values():
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            frames = []
            for eye in ("left", "right"):
                ok, frame = self._playback_captures[eye].read()
                if not ok:
                    raise IndexError("Frame is outside the saved recording")
                frames.append(frame)
            self._playback_next_frame = frame_index + 1
            return encode_jpeg(cv2.hconcat(frames), quality)

    def close(self):
        with self._playback_lock:
            self._close_playback_locked()

    def _close_playback_locked(self):
        for capture in self._playback_captures.values():
            capture.release()
        self._playback_captures = {}
        self._playback_batch = None
        self._playback_next_frame = 0


RECORD_PAGE = """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>Babybot Record</title><style>body{font-family:sans-serif;background:#111;color:#eee;margin:20px}button,select{font-size:16px;padding:9px}.eyes{display:grid;grid-template-columns:1fr 1fr;gap:12px}.eyes img,.eyes video{width:100%;background:#222}pre{white-space:pre-wrap}.panel{background:#202020;padding:14px;margin:14px 0}</style></head>
<body><h1>Babybot Record mode</h1><section class='panel'><p id='status'>Loading...</p><button id='start'>Start recording</button> <button id='stop'>Stop recording</button> <button id='retry'>Retry upload</button></section>
<h2>Live stereo preview</h2><div class='eyes'><img id='live-left'><img id='live-right'></div>
<section class='panel'><h2>Saved recording inspector</h2><select id='batches'></select> <button id='reload'>Reload</button><pre id='details'></pre>
<p>Paired playback: left eye | right eye</p><img id='playback' style='width:100%;background:#222'>
<p><button id='play'>Play</button> <button id='pause'>Pause</button> <button id='back'>Previous frame</button> <button id='forward'>Next frame</button></p>
<input id='timeline' type='range' min='0' max='0' value='0' style='width:100%'><p id='frame-label'></p></section>
<script>async function post(p){let r=await fetch(p,{method:'POST'});if(!r.ok)throw Error(await r.text())}
const startButton=document.getElementById('start'),stopButton=document.getElementById('stop'),retryButton=document.getElementById('retry'),batchSelect=document.getElementById('batches'),detailsElement=document.getElementById('details');
async function status(){try{let s=await(await fetch('/status.json?t='+Date.now())).json();let encoder=s.encoder||'waiting for first frame';document.getElementById('status').textContent=`${s.message}\nState: ${s.state} | Encoder: ${encoder}\nFrames: ${s.paired_frame_count}/${s.submitted_frame_count} | Queue: ${s.queued_pairs}/${s.queue_capacity} (${s.queue_percent}%)\nUpload: ${s.upload_state}`;startButton.disabled=s.state!=='idle';stopButton.disabled=!s.recording;retryButton.disabled=s.upload_state!=='failed'}catch(e){}setTimeout(status,500)}
function preview(e){let i=document.getElementById('live-'+e);i.onload=i.onerror=()=>setTimeout(()=>preview(e),60);i.src='/preview/'+e+'.jpg?t='+Date.now()}
let records=[],playing=false,playTimer=null,playback=document.getElementById('playback'),timeline=document.getElementById('timeline'),frameLabel=document.getElementById('frame-label');async function load(){records=await(await fetch('/recordings.json?t='+Date.now())).json();batchSelect.replaceChildren(...records.map(x=>new Option(x.batch,x.batch)));choose()}
function selected(){return records.find(x=>x.batch===batchSelect.value)}
function showFrame(n){let x=selected();if(!x)return;let total=Math.max(0,x.paired_frame_count||0);n=Math.max(0,Math.min(Number(n)||0,Math.max(0,total-1)));timeline.value=n;frameLabel.textContent=`Frame ${n+1} / ${total}`;playback.src=`/paired-frame/${x.batch}/${n}.jpg?t=${Date.now()}`}
function choose(){playing=false;clearTimeout(playTimer);let x=selected();if(!x)return;detailsElement.textContent=JSON.stringify(x,null,2);timeline.max=Math.max(0,(x.paired_frame_count||0)-1);timeline.value=0;showFrame(0)}
function playbackStep(){if(!playing)return;let x=selected(),next=Number(timeline.value)+1;if(!x||next>=x.paired_frame_count){playing=false;return}showFrame(next);let fps=(x.videos.left&&x.videos.left.fps)||20;playTimer=setTimeout(playbackStep,1000/Math.max(1,fps))}
startButton.onclick=()=>post('/action/start');stopButton.onclick=()=>post('/action/stop');retryButton.onclick=()=>post('/action/retry-upload');document.getElementById('reload').onclick=load;batchSelect.onchange=choose;
document.getElementById('play').onclick=()=>{if(!playing){playing=true;playbackStep()}};document.getElementById('pause').onclick=()=>{playing=false;clearTimeout(playTimer)};document.getElementById('back').onclick=()=>{playing=false;showFrame(Number(timeline.value)-1)};document.getElementById('forward').onclick=()=>{playing=false;showFrame(Number(timeline.value)+1)};timeline.oninput=()=>{playing=false;clearTimeout(playTimer);showFrame(timeline.value)};preview('left');preview('right');status();load()</script></body></html>"""


def make_record_handler(runtime):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = unquote(self.path.split("?", 1)[0])
            if path == "/":
                return self.send_bytes(RECORD_PAGE.encode(), "text/html; charset=utf-8")
            if path == "/status.json":
                return self.send_json(runtime.recorder.status())
            if path == "/recordings.json":
                return self.send_json(runtime.library.list())
            parts = path.strip("/").split("/")
            if len(parts) == 2 and parts[0] == "preview" and parts[1] in ("left.jpg", "right.jpg"):
                snapshot = runtime.frames.snapshot(copy=False)
                if snapshot is None:
                    return self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "Waiting for camera")
                image = snapshot[0 if parts[1] == "left.jpg" else 1]
                return self.send_bytes(encode_jpeg(image), "image/jpeg")
            if len(parts) == 3 and parts[0] == "media":
                try:
                    return self.send_file(runtime.library.video_path(parts[1], parts[2]))
                except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
                    return self.send_error(HTTPStatus.NOT_FOUND)
            if (len(parts) == 3 and parts[0] == "paired-frame"
                    and parts[2].endswith(".jpg")):
                try:
                    frame_index = int(parts[2][:-4])
                    body = runtime.library.paired_frame_jpeg(parts[1], frame_index)
                    return self.send_bytes(body, "image/jpeg")
                except (FileNotFoundError, OSError, ValueError, IndexError,
                        json.JSONDecodeError):
                    return self.send_error(HTTPStatus.NOT_FOUND)
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            action = {"/action/start": runtime.start_recording,
                      "/action/stop": runtime.recorder.stop_async,
                      "/action/retry-upload": runtime.recorder.retry_upload}.get(path)
            if action is None:
                return self.send_error(HTTPStatus.NOT_FOUND)
            if not action():
                return self.send_error(HTTPStatus.CONFLICT, "Invalid state")
            self.send_bytes(b'{"accepted":true}', "application/json", HTTPStatus.ACCEPTED)

        def send_json(self, value):
            self.send_bytes(json.dumps(value).encode(), "application/json")

        def send_bytes(self, body, content_type, status=HTTPStatus.OK):
            self.send_response(status); self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)

        def send_file(self, path):
            size = path.stat().st_size
            start, end = 0, size - 1
            header = self.headers.get("Range")
            status = HTTPStatus.OK
            if header and header.startswith("bytes="):
                first, _, last = header[6:].partition("-")
                start = int(first or 0); end = min(int(last) if last else end, end)
                status = HTTPStatus.PARTIAL_CONTENT
            if start < 0 or start > end:
                return self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_response(status)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as stream:
                stream.seek(start); remaining = end - start + 1
                while remaining:
                    block = stream.read(min(1024 * 1024, remaining))
                    if not block: break
                    self.wfile.write(block); remaining -= len(block)

        def log_message(self, message, *args):
            LOGGER.debug(message, *args)
    return Handler


class RecordRuntime:
    def __init__(self, args):
        self.stop_event = threading.Event()
        self.frames = LatestStereoFrame()
        self.camera = StereoCamera(args.left_camera, args.right_camera, args.width,
                                   args.height, args.camera_fps)
        self.recorder = StereoRecorder(StereoRecordingConfig(
            data_root=args.recording_root, camera_fps=args.camera_fps,
            upload_repository=args.upload_repo,
        ))
        self.library = RecordingLibrary(args.recording_root)
        self.server = ThreadingHTTPServer((args.host, args.port), make_record_handler(self))
        self.camera_thread = None

    def start_recording(self):
        snapshot = self.frames.snapshot(copy=False)
        if snapshot is None or snapshot[0].shape != snapshot[1].shape:
            return False
        return self.recorder.start(
            snapshot[0].shape, self.frames.capture_fps(self.camera.fps)
        )

    def run(self):
        self.camera.open_until_ready(self.stop_event)
        self.camera.warm_up(2.0, self.stop_event)
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
        LOGGER.info("Record mode listens at http://127.0.0.1:%d", self.server.server_port)
        self.server.serve_forever()

    def camera_loop(self):
        while not self.stop_event.is_set():
            pair = self.camera.capture_pair()
            if pair is None:
                self.stop_event.wait(0.1); continue
            left, right, delta = pair
            wall_ns, mono_ns = time.time_ns(), time.monotonic_ns()
            self.frames.update(left, right, mono_ns / 1e9)
            self.recorder.submit(left, right, wall_ns, mono_ns, delta)

    def stop(self, *_args):
        self.stop_event.set()
        if self.recorder.status()["recording"]:
            self.recorder.stop_async()
            LOGGER.info("Shutdown requested; no new frames will be recorded")
        threading.Thread(target=self.server.shutdown, daemon=True).start()

    def close(self):
        self.recorder.shutdown(); self.library.close(); self.camera.release(); self.server.server_close()
        if self.camera_thread: self.camera_thread.join(timeout=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Babybot Record mode")
    parser.add_argument("--left-camera", type=int, default=0); parser.add_argument("--right-camera", type=int, default=2)
    parser.add_argument("--width", type=int, default=1280); parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--camera-fps", type=float, default=60); parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081); parser.add_argument("--recording-root", default="recordings")
    parser.add_argument("--upload-repo", default=os.environ.get("BABYBOT_TGLGENERAL_REPO", ""))
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    runtime = RecordRuntime(parse_args())
    signal.signal(signal.SIGINT, runtime.stop); signal.signal(signal.SIGTERM, runtime.stop)
    try: runtime.run()
    finally: runtime.close()


if __name__ == "__main__":
    main()
