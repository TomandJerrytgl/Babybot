"""Cross-platform offline Dreaming mode for algorithm and Memory experiments."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import signal
import threading
import time

import cv2
import numpy as np

from conscious import crop_window, visual_feature
from feature_training import FeatureTrainer
from observation import Observation
from perception import Perception
from shared_memory import SharedMemory
from stereo_dataset import StereoDataset
from vision_pipeline import calculate_attention_pair, default_attention_settings, encode_jpeg, encode_preview


LOGGER = logging.getLogger("babybot.dreaming")


def pair_attention_objects(perception, result):
    """Pair compatible candidates greedily; retain every unmatched single-eye object."""
    left_items = list(result.get("left", []))
    right_items = list(result.get("right", []))
    available = set(range(len(right_items)))
    objects = []
    for left in left_items:
        lx, ly, lw, lh = left["window"]
        best = None
        for index in available:
            right = right_items[index]
            rx, ry, rw, rh = right["window"]
            vertical = abs((ly + lh / 2) - (ry + rh / 2)) / max(lh, rh, 1)
            size = min(lw * lh, rw * rh) / max(lw * lh, rw * rh, 1)
            score = 0.65 * (1.0 - min(vertical, 1.0)) + 0.35 * size
            if vertical <= 0.6 and size >= 0.35 and (best is None or score > best[0]):
                best = (score, index)
        if best is None:
            objects.append({"eye_mode": "left", "left": left, "right": None})
        else:
            available.remove(best[1])
            objects.append({"eye_mode": "stereo", "left": left,
                            "right": right_items[best[1]]})
    objects.extend({"eye_mode": "right", "left": None, "right": right_items[index]}
                   for index in sorted(available))
    return objects


class DreamingController:
    def __init__(self, recordings_root, memory_root):
        self.recordings_root = Path(recordings_root)
        self.memory = SharedMemory(memory_root)
        self.trainer = FeatureTrainer()
        self._condition = threading.Condition()
        self._thread = None
        self._stop = False
        self._paused = False
        self._step = False
        self._pending_payload = None
        self._images = {}
        self._status = {
            "state": "idle", "message": "Select a recording", "batch": None,
            "frame": 0, "total_frames": 0, "stride": 1, "new_objects": 0,
            "duplicates": 0, "pending_count": 0, "processed_frames": 0,
            "memory": self.memory.counts(), "trainer": self.trainer.status(),
            "error": None, "pending": None,
        }

    def recordings(self):
        self.recordings_root.mkdir(parents=True, exist_ok=True)
        output = []
        for path in sorted(self.recordings_root.glob("recording_*"), reverse=True):
            try:
                dataset = StereoDataset(path)
                output.append({"batch": path.name, "schema": dataset.schema,
                               "pairs": dataset.metadata.get("paired_frame_count", 0)})
            except Exception as error:
                output.append({"batch": path.name, "error": str(error)})
        return output

    def status(self):
        with self._condition:
            result = dict(self._status)
            result["memory"] = self.memory.counts()
            return result

    def image(self, name):
        with self._condition:
            return self._images.get(name)

    def start(self, batch, stride=1):
        if Path(batch).name != batch or not batch.startswith("recording_"):
            return False
        path = self.recordings_root / batch
        try:
            dataset = StereoDataset(path)
        except Exception as error:
            with self._condition:
                self._status.update({"state": "error", "error": str(error), "message": "Dataset failed validation"})
            return False
        with self._condition:
            if self._status["state"] in ("running", "paused", "waiting_confirmation"):
                return False
            self._stop = self._paused = self._step = False
            self._pending_payload = None
            self._status.update({
                "state": "running", "message": "Processing as fast as possible",
                "batch": batch, "frame": 0,
                "total_frames": int(dataset.metadata.get("paired_frame_count", 0)),
                "stride": max(1, int(stride)), "new_objects": 0, "duplicates": 0,
                "pending_count": 0, "processed_frames": 0, "error": None, "pending": None,
            })
            self._thread = threading.Thread(
                target=self._run, args=(dataset,), name="dreaming-worker", daemon=True
            )
            self._thread.start()
            return True

    def pause(self):
        with self._condition:
            if self._status["state"] != "running": return False
            self._paused = True; self._status.update({"state": "paused", "message": "Paused"})
            return True

    def resume(self):
        with self._condition:
            if self._status["state"] != "paused" or self._pending_payload is not None: return False
            self._paused = False; self._status.update({"state": "running", "message": "Processing"})
            self._condition.notify_all(); return True

    def step(self):
        with self._condition:
            if self._status["state"] != "paused" or self._pending_payload is not None: return False
            self._step = True; self._condition.notify_all(); return True

    def stop(self):
        with self._condition:
            if self._status["state"] not in ("running", "paused", "waiting_confirmation"): return False
            self._stop = True; self._condition.notify_all(); return True

    def decide(self, decision):
        if decision not in ("same", "different", "unsure"):
            return False
        with self._condition:
            payload = self._pending_payload
            if payload is None: return False
        if decision == "same":
            self.memory.add_sample(payload["similar"]["object_id"], payload["left"], payload["right"],
                                   payload["feature"], payload["metadata"], eye_mode=payload["eye_mode"])
            self.memory.record_decision(payload["similar"]["object_id"], None, "same",
                                        payload["metadata"])
            self.trainer.add_object_sample(payload["similar"]["object_id"], payload["feature"])
        elif decision == "different":
            identifier = self.memory.add_object(payload["left"], payload["right"], payload["feature"],
                                                payload["metadata"], eye_mode=payload["eye_mode"])
            self.memory.record_decision(payload["similar"]["object_id"], identifier, "different",
                                        payload["metadata"])
            self.trainer.add_object_sample(identifier, payload["feature"])
        else:
            self.memory.record_decision(payload["similar"]["object_id"], None, "unsure",
                                        payload["metadata"])
        with self._condition:
            self._pending_payload = None
            self._status.update({"state": "running", "message": "Processing", "pending": None})
            self._condition.notify_all()
        return True

    def _run(self, dataset):
        settings = default_attention_settings()
        try:
            for pair, left, right in dataset.images():
                with self._condition:
                    while (self._paused or self._pending_payload is not None) and not self._stop:
                        if self._step and self._pending_payload is None:
                            self._step = False; break
                        self._condition.wait(0.2)
                    if self._stop: break
                    stride = self._status["stride"]
                if pair.index % stride:
                    continue
                observation = Observation.from_frames(
                    pair.index, pair.timestamp_ns / 1e9, pair.monotonic_ns / 1e9,
                    left, right,
                )
                perception = Perception.from_observation(observation)
                result = calculate_attention_pair(perception, settings)
                self._update_images(perception, result)
                for item in pair_attention_objects(perception, result):
                    if not self._handle_object(dataset.root.name, pair.index, perception, item):
                        break
                with self._condition:
                    self._status["frame"] = pair.index
                    self._status["processed_frames"] += 1
                    if self._paused and self._pending_payload is None:
                        self._status.update({"state": "paused", "message": "Paused after one frame"})
            with self._condition:
                if self._stop:
                    self._status.update({"state": "stopped", "message": "Stopped"})
                elif self._pending_payload is None:
                    self._status.update({"state": "complete", "message": "Dreaming complete"})
        except Exception as error:
            LOGGER.exception("Dreaming failed")
            with self._condition:
                self._status.update({"state": "error", "message": "Dreaming failed", "error": str(error)})

    def _handle_object(self, batch, frame_index, perception, item):
        left_crop = crop_window(perception.left, item["left"]["window"]) if item["left"] else None
        right_crop = crop_window(perception.right, item["right"]["window"]) if item["right"] else None
        features = [visual_feature(image) for image in (left_crop, right_crop) if image is not None]
        feature = np.mean(features, axis=0).astype(np.float32)
        metadata = {
            "source_recording": batch, "source_frame": frame_index,
            "eye_mode": item["eye_mode"], "left_window": item["left"]["window"] if item["left"] else None,
            "right_window": item["right"]["window"] if item["right"] else None,
            "depth": None, "physical_size": None,
            "neighbor_stereo_objects": [], "algorithm_version": "dreaming-v1",
        }
        similar = self.memory.find_similar(feature)
        if similar is None or similar["similarity"] < self.memory.similarity_threshold:
            identifier = self.memory.add_object(
                left_crop, right_crop, feature, metadata, eye_mode=item["eye_mode"]
            )
            self.trainer.add_object_sample(identifier, feature)
            with self._condition: self._status["new_objects"] += 1
            return True
        if similar["similarity"] >= self.memory.duplicate_threshold:
            with self._condition: self._status["duplicates"] += 1
            return True
        payload = {"left": left_crop, "right": right_crop, "feature": feature,
                   "metadata": metadata, "eye_mode": item["eye_mode"], "similar": similar}
        with self._condition:
            self._pending_payload = payload
            candidate_image = left_crop if left_crop is not None else right_crop
            if candidate_image is not None:
                self._images["pending_candidate"] = encode_jpeg(candidate_image, 85)
            existing_path = similar.get("left_image") or similar.get("right_image")
            if existing_path:
                existing_image = cv2.imread(str(self.memory.root / existing_path))
                if existing_image is not None:
                    self._images["pending_existing"] = encode_jpeg(existing_image, 85)
            self._status["pending_count"] += 1
            self._status.update({
                "state": "waiting_confirmation", "message": "Possible repeated object needs confirmation",
                "pending": {"existing_object_id": similar["object_id"],
                            "similarity": round(similar["similarity"], 4),
                            "eye_mode": item["eye_mode"]},
            })
        return False

    def _update_images(self, perception, result):
        with self._condition:
            self._images = {
                "left": encode_preview(perception.left, result.get("left", []), 80),
                "right": encode_preview(perception.right, result.get("right", []), 80),
            }


DREAM_PAGE = """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>Babybot Dreaming</title><style>body{font-family:sans-serif;background:#111;color:#eee;margin:20px}.eyes{display:grid;grid-template-columns:1fr 1fr;gap:12px}.eyes img{width:100%;background:#222}button,select{font-size:16px;padding:9px}pre{white-space:pre-wrap}.panel{background:#222;padding:14px;margin:14px 0}</style></head><body><h1>Babybot Dreaming mode</h1><section class='panel'><select id='batch'></select><select id='stride'><option value='1'>Every frame</option><option value='2'>Every 2 frames</option><option value='5'>Every 5 frames</option><option value='10'>Every 10 frames</option></select><button id='start'>Start</button><button id='pause'>Pause</button><button id='resume'>Resume</button><button id='step'>Single frame</button><button id='stop'>Stop</button></section><pre id='status'></pre><div id='decisions' class='panel' hidden><h2>Possible repeated object</h2><div class='eyes'><img id='pending-candidate'><img id='pending-existing'></div><button data-d='same'>Same object</button><button data-d='different'>Different object</button><button data-d='unsure'>Unsure</button></div><div class='eyes'><img id='left'><img id='right'></div><script>async function post(path,data={}){let r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});if(!r.ok)throw Error(await r.text())}async function load(){let x=await(await fetch('/recordings.json')).json();batch.replaceChildren(...x.filter(v=>!v.error).map(v=>new Option(v.batch,v.batch)))}async function poll(){try{let s=await(await fetch('/status.json?t='+Date.now())).json();status.textContent=JSON.stringify(s,null,2);decisions.hidden=!s.pending;if(s.pending){document.getElementById('pending-candidate').src='/frame/pending_candidate.jpg?t='+Date.now();document.getElementById('pending-existing').src='/frame/pending_existing.jpg?t='+Date.now()}if(s.processed_frames){left.src='/frame/left.jpg?t='+s.processed_frames;right.src='/frame/right.jpg?t='+s.processed_frames}}catch(e){}setTimeout(poll,400)}start.onclick=()=>post('/action/start',{batch:batch.value,stride:+stride.value});pause.onclick=()=>post('/action/pause');resume.onclick=()=>post('/action/resume');step.onclick=()=>post('/action/step');stop.onclick=()=>post('/action/stop');decisions.onclick=e=>{if(e.target.dataset.d)post('/action/decision',{decision:e.target.dataset.d})};load();poll()</script></body></html>"""


def make_handler(controller):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/": return self.send_data(DREAM_PAGE.encode(), "text/html; charset=utf-8")
            if path == "/status.json": return self.send_json(controller.status())
            if path == "/recordings.json": return self.send_json(controller.recordings())
            if path.startswith("/frame/") and path.endswith(".jpg"):
                name = path.rsplit("/", 1)[-1][:-4]
                if name not in ("left", "right", "pending_candidate", "pending_existing"):
                    return self.send_error(HTTPStatus.NOT_FOUND)
                data = controller.image(name)
                if data is None: return self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
                return self.send_data(data, "image/jpeg")
            self.send_error(HTTPStatus.NOT_FOUND)
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0)); data = json.loads(self.rfile.read(length) or b"{}")
            path = self.path.split("?", 1)[0]
            actions = {"/action/start": lambda: controller.start(data.get("batch", ""), data.get("stride", 1)),
                       "/action/pause": controller.pause, "/action/resume": controller.resume,
                       "/action/step": controller.step, "/action/stop": controller.stop,
                       "/action/decision": lambda: controller.decide(data.get("decision"))}
            action = actions.get(path)
            if action is None: return self.send_error(HTTPStatus.NOT_FOUND)
            if not action(): return self.send_error(HTTPStatus.CONFLICT, "Invalid state")
            self.send_data(b'{"accepted":true}', "application/json", HTTPStatus.ACCEPTED)
        def send_json(self, value): self.send_data(json.dumps(value).encode(), "application/json")
        def send_data(self, body, content_type, status=HTTPStatus.OK):
            self.send_response(status); self.send_header("Content-Type", content_type); self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def log_message(self, message, *args): LOGGER.debug(message, *args)
    return Handler


def parse_args():
    parser = argparse.ArgumentParser(description="Run cross-platform Babybot Dreaming mode")
    parser.add_argument("--recordings", default="recordings"); parser.add_argument("--memory", default="memory")
    parser.add_argument("--host", default="127.0.0.1"); parser.add_argument("--port", type=int, default=8082)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args(); controller = DreamingController(args.recordings, args.memory)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(controller))
    stopping = threading.Event()
    def stop(*_args):
        if not stopping.is_set(): stopping.set(); controller.stop(); threading.Thread(target=server.shutdown).start()
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    LOGGER.info("Dreaming mode listens at http://127.0.0.1:%d", args.port)
    try: server.serve_forever()
    finally: server.server_close()


if __name__ == "__main__":
    main()
