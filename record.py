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


RECORD_PAGE = """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>Babybot Record</title><style>body{font-family:sans-serif;background:#111;color:#eee;margin:20px}button,select{font-size:16px;padding:9px}.eyes{display:grid;grid-template-columns:1fr 1fr;gap:12px}.eyes img,.eyes video{width:100%;background:#222}pre{white-space:pre-wrap}.panel{background:#202020;padding:14px;margin:14px 0}</style></head>
<body><h1>Babybot Record mode</h1><section class='panel'><p id='status'>Loading...</p><button id='start'>Start recording</button> <button id='stop'>Stop recording</button> <button id='retry'>Retry upload</button></section>
<h2>Live stereo preview</h2><div class='eyes'><img id='live-left'><img id='live-right'></div>
<section class='panel'><h2>Saved recording inspector</h2><select id='batches'></select> <button id='reload'>Reload</button><pre id='details'></pre>
<div class='eyes'><video id='video-left' controls></video><video id='video-right' controls></video></div>
<p><button id='back'>Previous frame</button> <button id='forward'>Next frame</button></p></section>
<script>async function post(p){let r=await fetch(p,{method:'POST'});if(!r.ok)throw Error(await r.text())}
async function status(){try{let s=await(await fetch('/status.json?t='+Date.now())).json();document.getElementById('status').textContent=JSON.stringify(s);start.disabled=s.state!=='idle';stop.disabled=!s.recording;retry.disabled=s.upload_state!=='failed'}catch(e){}setTimeout(status,500)}
function preview(e){let i=document.getElementById('live-'+e);i.onload=i.onerror=()=>setTimeout(()=>preview(e),60);i.src='/preview/'+e+'.jpg?t='+Date.now()}
let records=[];async function load(){records=await(await fetch('/recordings.json?t='+Date.now())).json();batches.replaceChildren(...records.map(x=>new Option(x.batch,x.batch)));choose()}
function choose(){let x=records.find(x=>x.batch===batches.value);if(!x)return;details.textContent=JSON.stringify(x,null,2);['left','right'].forEach(e=>{let v=document.getElementById('video-'+e);v.src='/media/'+x.batch+'/'+e;v.dataset.fps=(x.videos[e]&&x.videos[e].fps)||20})}
function sync(source,target){source.addEventListener('play',()=>target.play());source.addEventListener('pause',()=>target.pause());source.addEventListener('seeked',()=>{if(Math.abs(target.currentTime-source.currentTime)>.03)target.currentTime=source.currentTime});source.addEventListener('ratechange',()=>target.playbackRate=source.playbackRate)}
start.onclick=()=>post('/action/start');stop.onclick=()=>post('/action/stop');retry.onclick=()=>post('/action/retry-upload');reload.onclick=load;batches.onchange=choose;
let l=document.getElementById('video-left'),r=document.getElementById('video-right');sync(l,r);function frameStep(){return 1/Math.max(1,parseFloat(l.dataset.fps)||20)}back.onclick=()=>{l.pause();l.currentTime=Math.max(0,l.currentTime-frameStep());r.currentTime=l.currentTime};forward.onclick=()=>{l.pause();l.currentTime+=frameStep();r.currentTime=l.currentTime};preview('left');preview('right');status();load()</script></body></html>"""


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
        threading.Thread(target=self.server.shutdown, daemon=True).start()

    def close(self):
        self.recorder.shutdown(); self.camera.release(); self.server.server_close()
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
