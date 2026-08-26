"""Babybot visual-front-end runtime: Observation -> Perception -> Attention."""

from __future__ import annotations

import argparse
import base64
from collections import deque
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
from region_proposal import RegionProposalConfig, StereoRegionProposer
from stereo_recording import StereoRecorder, StereoRecordingConfig


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
    proposal_config = RegionProposalConfig(
        minimum_region_fraction=settings["minimum_region_fraction"],
        growth_threshold=settings["region_growth_threshold"],
        minimum_merge_score=settings["minimum_merge_score"],
        stereo_merge_weight=settings["stereo_merge_weight"],
    )
    try:
        region_result = StereoRegionProposer(proposal_config).propose(
            perception.left, perception.right
        )
    except Exception:
        region_result = {
            "left": [], "right": [], "visualizations": {},
            "diagnostics": {
                "mode": "multiscale_fallback",
                "proposal_error": traceback.format_exc(),
            },
        }
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
            attention = Attention(
                perception, eye=eye,
                proposal_windows=region_result[eye], **arguments
            )
            result[eye] = [candidate.copy() for candidate in attention.candidates]
            result[f"{eye}_elapsed"] = attention.elapsed_time
        except Exception:
            result[f"{eye}_elapsed"] = time.perf_counter() - eye_started
            result[f"{eye}_error"] = traceback.format_exc()
    result["elapsed_time"] = time.perf_counter() - started
    result["region_diagnostics"] = region_result["diagnostics"]
    result["region_visualizations"] = region_result["visualizations"]
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
    minimum_region_fraction: float = 0.03
    region_growth_threshold: float = 14.0
    minimum_merge_score: float = 0.65
    stereo_merge_weight: float = 0.12
    recording_root: str = "recordings"
    recording_jpeg_quality: int = 95
    recording_queue_capacity: int = 512
    upload_repository: str = ""
    upload_subdirectory: str = "babybot/stereo_test_data"

    def attention_settings(self):
        return {
            "minimum_objectness": self.minimum_objectness,
            "maximum_candidates": self.maximum_candidates_per_eye,
            "partial_overlap_iou": self.partial_overlap_iou,
            "minimum_region_fraction": self.minimum_region_fraction,
            "region_growth_threshold": self.region_growth_threshold,
            "minimum_merge_score": self.minimum_merge_score,
            "stereo_merge_weight": self.stereo_merge_weight,
        }


class PreviewStore:
    """Thread-safe latest-only JPEG and diagnostic storage."""

    def __init__(self):
        self._lock = threading.Lock()
        self._images = {}
        self._identifiers = {}
        self._candidate_images = {}
        self._region_images = {}
        self._attention_status = {
            "ready": False, "calculating": False, "trigger_pending": False,
            "message": "Press Capture and calculate to create the first result",
        }

    def request_capture(self):
        with self._lock:
            if (self._attention_status.get("calculating")
                    or self._attention_status.get("trigger_pending")):
                return False
            self._attention_status["trigger_pending"] = True
            self._attention_status["message"] = "Capture requested"
            return True

    def begin_capture(self):
        with self._lock:
            self._attention_status["trigger_pending"] = False
            self._attention_status["calculating"] = True
            self._attention_status["message"] = "Calculating perception and attention"

    def fail_capture(self, message):
        with self._lock:
            self._attention_status["trigger_pending"] = False
            self._attention_status["calculating"] = False
            self._attention_status["message"] = str(message)

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
        region_images = {
            name: encode_jpeg(image, jpeg_quality)
            for name, image in result.get("region_visualizations", {}).items()
        }
        for eye in ("left", "right"):
            image = getattr(perception, eye)
            for candidate in result[eye]:
                crop, _ = make_candidate_crop(image, candidate, crop_scale)
                candidate_images[(eye, candidate["rank"])] = encode_jpeg(crop, jpeg_quality)
        status = {
            "ready": True,
            "calculating": False,
            "trigger_pending": False,
            "message": "Result ready; press Capture and calculate for the next result",
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
            self._region_images = region_images
            self._attention_status = status

    def get(self, kind, eye):
        with self._lock:
            return self._images.get((kind, eye)), self._identifiers.get(kind, -1)

    def get_candidate(self, eye, rank):
        with self._lock:
            return self._candidate_images.get((eye, int(rank))), self._identifiers.get("perception", -1)

    def get_region(self, name):
        with self._lock:
            return self._region_images.get(name), self._identifiers.get("perception", -1)

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
            if elapsed <= 0:
                return float(fallback)
            return (len(self._capture_times) - 1) / elapsed


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
        "source": str(candidate.get("source", "default")),
    }
    for key, _label in SCORE_FIELDS:
        value = candidate.get(key)
        details[key] = None if value is None else round(float(value), 4)
    return details


def jpeg_data_uri(data):
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def merge_diagnostics_html(region_diagnostics):
    """Render accepted and rejected region merges as inspectable score tables."""
    score_fields = (
        "merge_score", "color_similarity", "weak_boundary",
        "scale_compatibility", "merged_fill", "shape_continuity",
        "small_region_preference", "depth_similarity",
    )
    headers = "".join(f"<th>{html.escape(field)}</th>" for field in score_fields)
    sections = []
    for eye in ("left", "right"):
        eye_data = region_diagnostics.get(eye, {})
        records = list(eye_data.get("visual_merges", [])) + list(
            eye_data.get("stereo_merges", [])
        )
        for accepted, title in ((True, "Accepted merges"), (False, "Rejected merges")):
            selected = [item for item in records if bool(item.get("accepted")) is accepted]
            if accepted:
                selected.sort(key=lambda item: (item.get("stage", ""), item.get("order", 0)))
            else:
                selected.sort(key=lambda item: item.get("merge_score", 0.0), reverse=True)
            rows = []
            for item in selected:
                values = "".join(
                    f"<td>{'—' if item.get(field) is None else f'{float(item[field]):.4f}'}</td>"
                    for field in score_fields
                )
                rows.append(
                    f"<tr><td>{html.escape(str(item.get('stage', '')))}</td>"
                    f"<td>{int(item.get('order', 0))}</td>"
                    f"<td>{html.escape(str(item.get('component_a', item.get('regions', []))))}</td>"
                    f"<td>{html.escape(str(item.get('component_b', [])))}</td>"
                    f"<td>{html.escape(str(item.get('protected', [])))}</td>{values}</tr>"
                )
            table = (
                "<div class='table-wrap'><table><thead><tr><th>stage</th><th>order</th>"
                f"<th>component A</th><th>component B</th><th>protected</th>{headers}"
                f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
            )
            if accepted:
                sections.append(f"<h3>{eye.title()} eye — {title} ({len(selected)})</h3>{table}")
            else:
                sections.append(
                    f"<details><summary>{eye.title()} eye — {title} ({len(selected)})</summary>"
                    f"{table}</details>"
                )
    return "".join(sections)


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
                f"<table><tr><th>source</th><td>{html.escape(details['source'])}</td></tr>"
                f"{score_lines}<tr><th>area fraction</th><td>{details['area_fraction']:.5f}</td></tr></table>"
                "</article>"
            )
        sections.append(
            f"<section><h2>{title}: {len(candidates)} candidates</h2>"
            f"<img class='full' src='{jpeg_data_uri(full)}'><div class='cards'>{''.join(cards)}</div></section>"
        )
    region_diagnostics = html.escape(json.dumps(
        result.get("region_diagnostics", {}), ensure_ascii=False, indent=2
    ))
    merge_tables = merge_diagnostics_html(result.get("region_diagnostics", {}))
    visualizations = result.get("region_visualizations", {})
    visualization_cards = "".join(
        f"<article class='card'><h3>{html.escape(name.replace('_', ' '))}</h3>"
        f"<img src='{jpeg_data_uri(encode_jpeg(image, jpeg_quality))}'></article>"
        for name, image in visualizations.items()
    )
    document = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Babybot attention report</title>"
        "<style>body{font-family:sans-serif;background:#111;color:#eee;margin:24px}.full{max-width:900px;width:100%}"
        ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}.card{background:#222;padding:12px}"
        ".card img{width:100%;height:auto}table{width:100%;border-collapse:collapse}th{text-align:left}"
        "th,td{padding:3px;border-bottom:1px solid #444;white-space:nowrap}.table-wrap{overflow:auto;max-height:620px}"
        "details{margin:16px 0}summary{cursor:pointer;font-size:1.1rem}</style></head><body>"
        f"<h1>Babybot attention report</h1><p>Perception {int(identifier)} | observation {int(perception.observation_id)} | "
        f"total {result['elapsed_time'] * 1000:.1f} ms | left {result['left_elapsed'] * 1000:.1f} ms | "
        f"right {result['right_elapsed'] * 1000:.1f} ms</p>{''.join(sections)}"
        f"<section><h2>Region proposal stages</h2><div class='cards'>{visualization_cards}</div>"
        f"<h2>Merge score details</h2>{merge_tables}"
        f"<details><summary>Raw region diagnostics JSON</summary><pre>{region_diagnostics}</pre></details></section>"
        "</body></html>"
    )
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(report_path)


def make_request_handler(previews, report_path="debug/attention_report.html",
                         request_capture=None, recording_status=None,
                         start_recording=None, stop_recording=None,
                         retry_recording_upload=None):
    class PreviewHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                body = (
                    "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
                    "<title>Babybot visual front end</title><style>body{background:#111;color:#eee;font-family:sans-serif;margin:20px}"
                    ".eyes{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}img{width:100%;height:auto;background:#222}"
                    ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}.card{background:#222;padding:10px}"
                    "pre{white-space:pre-wrap;font-size:12px}h1,h2{font-weight:500}button{font-size:18px;padding:12px 20px;cursor:pointer}"
                    "button:disabled{opacity:.55;cursor:wait}.recording{background:#301818;padding:14px;margin:16px 0}</style></head><body><h1>Babybot visual front end</h1>"
                    "<section class='recording'><h2>Stereo training recorder</h2><p id='recording-status'>Loading recorder status...</p>"
                    "<p><button id='record-start'>Start recording</button> <button id='record-stop'>Stop recording</button> "
                    "<button id='upload-retry'>Retry upload</button></p></section>"
                    "<h2>Raw observation — target 20 Hz — no overlay</h2><div class='eyes'><section><h3>Left eye</h3><img id='observation-left'></section>"
                    "<section><h3>Right eye</h3><img id='observation-right'></section></div>"
                    "<p><button id='capture'>Capture and calculate</button></p>"
                    "<h2>Perception + Attention — manually captured — 320×200</h2>"
                    "<p id='timing'>Press Capture and calculate</p><p><a href='/report/attention' target='_blank'>Open latest self-contained report</a></p>"
                    "<div class='eyes'><section><h3>Left eye</h3><img id='perception-left'></section><section><h3>Right eye</h3>"
                    "<img id='perception-right'></section></div>"
                    "<h2>Initial Lab region mask</h2><div class='eyes'><img id='region-left-initial_mask'><img id='region-right-initial_mask'></div>"
                    "<h2>Original perception + Lab boundaries</h2><div class='eyes'><img id='region-left-initial_overlay'><img id='region-right-initial_overlay'></div>"
                    "<h2>Visual merged region mask</h2><div class='eyes'><img id='region-left-visual_merged_mask'><img id='region-right-visual_merged_mask'></div>"
                    "<h2>Stereo merged region mask</h2><div class='eyes'><img id='region-left-stereo_merged_mask'><img id='region-right-stereo_merged_mask'></div>"
                    "<h2>Stereo merged boundaries</h2><div class='eyes'><img id='region-left-stereo_merged_overlay'><img id='region-right-stereo_merged_overlay'></div>"
                    "<h2>Left candidates</h2><div id='left-candidates' class='cards'></div>"
                    "<h2>Right candidates</h2><div id='right-candidates' class='cards'></div><script>"
                    "function refresh(k,d){let p=2,l=document.getElementById(k+'-left'),r=document.getElementById(k+'-right');"
                    "const done=()=>{if(--p===0)setTimeout(()=>refresh(k,d),d)};l.onload=l.onerror=done;r.onload=r.onerror=done;"
                    "let t=Date.now();l.src='/frame/'+k+'/left.jpg?t='+t;r.src='/frame/'+k+'/right.jpg?t='+t}"
                    "function cards(eye,items,id){let root=document.getElementById(eye+'-candidates');root.replaceChildren();items.forEach(c=>{"
                    "let a=document.createElement('article');a.className='card';let h=document.createElement('h3');h.textContent='#'+c.rank+' window '+c.window.join(', ');"
                    "let i=document.createElement('img');i.src='/candidate/'+eye+'/'+c.rank+'.jpg?t='+id;let p=document.createElement('pre');"
                    "p.textContent=JSON.stringify(c,null,2);a.append(h,i,p);root.append(a)})}"
                    "let shown=-1;function showResult(s){let id=s.perception_id,t=Date.now();['left','right'].forEach(e=>{"
                    "document.getElementById('perception-'+e).src='/frame/perception/'+e+'.jpg?t='+t;"
                    "['initial_mask','initial_overlay','visual_merged_mask','stereo_merged_mask','stereo_merged_overlay'].forEach(k=>{"
                    "document.getElementById('region-'+e+'-'+k).src='/region/'+e+'_'+k+'.jpg?t='+t})});"
                    "cards('left',s.left,id);cards('right',s.right,id);shown=id}"
                    "async function status(){try{let r=await fetch('/status/attention.json?t='+Date.now()),s=await r.json(),b=document.getElementById('capture');"
                    "b.disabled=!!(s.calculating||s.trigger_pending);document.getElementById('timing').textContent=s.message||'Waiting';"
                    "if(s.ready&&s.perception_id!==shown){showResult(s);document.getElementById('timing').textContent='Perception '+s.perception_id+' | left '+s.left_elapsed_ms+' ms | right '+s.right_elapsed_ms+' ms | total '+s.total_elapsed_ms+' ms'}}catch(e){}setTimeout(status,300)}"
                    "document.getElementById('capture').onclick=async()=>{let b=document.getElementById('capture');b.disabled=true;"
                    "try{await fetch('/action/capture',{method:'POST'})}catch(e){b.disabled=false}};"
                    "async function recordingStatus(){try{let r=await fetch('/status/recording.json?t='+Date.now()),s=await r.json();"
                    "let text=s.message+' | pairs '+s.paired_frame_count+' | '+Number(s.duration_seconds).toFixed(1)+' s';"
                    "if(s.batch)text+=' | '+s.batch;if(s.upload_message)text+=' | '+s.upload_message;if(s.error)text+=' | ERROR: '+s.error;"
                    "if(s.upload_error)text+=' | UPLOAD ERROR: '+s.upload_error;document.getElementById('recording-status').textContent=text;"
                    "document.getElementById('record-start').disabled=s.state!=='idle';document.getElementById('record-stop').disabled=!s.recording;"
                    "document.getElementById('upload-retry').disabled=s.upload_state!=='failed'}catch(e){}setTimeout(recordingStatus,500)}"
                    "async function postAction(path){try{await fetch(path,{method:'POST'})}catch(e){}}"
                    "document.getElementById('record-start').onclick=()=>postAction('/action/record/start');"
                    "document.getElementById('record-stop').onclick=()=>postAction('/action/record/stop');"
                    "document.getElementById('upload-retry').onclick=()=>postAction('/action/record/retry-upload');"
                    "refresh('observation',50);status();recordingStatus()</script></body></html>"
                ).encode("utf-8")
                self._send_bytes(body, "text/html; charset=utf-8")
                return
            if path == "/status/attention.json":
                self._send_bytes(
                    json.dumps(previews.attention_status()).encode("utf-8"),
                    "application/json", no_store=True,
                )
                return
            if path == "/status/recording.json":
                if recording_status is None:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "Recording unavailable")
                    return
                self._send_bytes(
                    json.dumps(recording_status()).encode("utf-8"),
                    "application/json", no_store=True,
                )
                return
            if path == "/report/attention":
                file_path = Path(report_path)
                if (not previews.attention_status().get("ready")
                        or not file_path.is_file()):
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
            if len(parts) == 2 and parts[0] == "region" and parts[1].endswith(".jpg"):
                image, identifier = previews.get_region(parts[1][:-4])
                self._send_image_or_wait(image, identifier)
                return
            self.send_error(HTTPStatus.NOT_FOUND, html.escape(path))

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            actions = {
                "/action/capture": (request_capture, "Capture control unavailable"),
                "/action/record/start": (start_recording, "Recording control unavailable"),
                "/action/record/stop": (stop_recording, "Recording control unavailable"),
                "/action/record/retry-upload": (retry_recording_upload, "Upload retry unavailable"),
            }
            if path not in actions:
                self.send_error(HTTPStatus.NOT_FOUND, html.escape(path))
                return
            action, unavailable_message = actions[path]
            if action is None:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, unavailable_message)
                return
            if not action():
                self.send_error(HTTPStatus.CONFLICT, "Action is not valid in the current state")
                return
            body = b'{"accepted":true}'
            self.send_response(HTTPStatus.ACCEPTED)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

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
        self.recorder = StereoRecorder(StereoRecordingConfig(
            data_root=config.recording_root,
            camera_fps=config.camera_fps,
            jpeg_quality=config.recording_jpeg_quality,
            queue_capacity=config.recording_queue_capacity,
            upload_repository=config.upload_repository,
            upload_subdirectory=config.upload_subdirectory,
        ))
        self.stop_event = threading.Event()
        self.capture_request_event = threading.Event()
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
        self.capture_request_event.set()

    def request_attention_capture(self):
        if not self.previews.request_capture():
            return False
        self.capture_request_event.set()
        return True

    def request_recording_start(self):
        snapshot = self.latest_frames.snapshot(copy=False)
        if snapshot is None:
            return False
        left, right, _version = snapshot
        if left.shape != right.shape:
            return False
        estimated_fps = self.latest_frames.capture_fps(self.config.camera_fps)
        return self.recorder.start(left.shape, video_fps=estimated_fps)

    def request_recording_stop(self):
        return self.recorder.stop_async()

    def request_recording_upload_retry(self):
        return self.recorder.retry_upload()

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
        if not self.left_camera.grab():
            return None
        left_grab_ns = time.monotonic_ns()
        if not self.right_camera.grab():
            return None
        right_grab_ns = time.monotonic_ns()
        left_ok, left = self.left_camera.retrieve()
        right_ok, right = self.right_camera.retrieve()
        if not left_ok or not right_ok or left is None or right is None:
            return None
        return left, right, abs(right_grab_ns - left_grab_ns)

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
                left, right, sync_delta_ns = pair
                capture_wall_ns = time.time_ns()
                capture_monotonic_ns = time.monotonic_ns()
                self.latest_frames.update(
                    left, right, monotonic_time=capture_monotonic_ns / 1e9
                )
                self.recorder.submit(
                    left, right, timestamp_ns=capture_wall_ns,
                    monotonic_ns=capture_monotonic_ns,
                    sync_delta_ns=sync_delta_ns,
                )
                continue
            if time.monotonic() - last_log >= 2.0:
                LOGGER.error("Stereo frame failed; retrying")
                last_log = time.monotonic()
            self.stop_event.wait(self.config.retry_delay)

    def _perception_loop(self):
        perception_id = 0
        settings = self.config.attention_settings()
        while not self.stop_event.is_set():
            if not self.capture_request_event.wait(0.1):
                continue
            self.capture_request_event.clear()
            if self.stop_event.is_set():
                break
            self.previews.begin_capture()
            snapshot = self.latest_frames.snapshot()
            if snapshot is None:
                self.previews.fail_capture("No camera frame is available yet")
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
                self.previews.fail_capture(
                    "Calculation failed; previous successful result was retained"
                )

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
        handler = make_request_handler(
            self.previews, self.config.attention_report_path,
            self.request_attention_capture,
            self.recorder.status,
            self.request_recording_start,
            self.request_recording_stop,
            self.request_recording_upload_retry,
        )
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
        self.capture_request_event.set()
        self.recorder.shutdown()
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
    parser.add_argument("--recording-root", default="recordings")
    parser.add_argument(
        "--upload-repo", default=os.environ.get("BABYBOT_TGLGENERAL_REPO", ""),
        help="Path to an existing private TGLgeneral Git clone",
    )
    parser.add_argument(
        "--upload-subdirectory", default="babybot/stereo_test_data",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    runtime = BabybotRuntime(RuntimeConfig(
        left_camera=args.left_camera,
        right_camera=args.right_camera,
        web_port=args.port,
        recording_root=args.recording_root,
        upload_repository=args.upload_repo,
        upload_subdirectory=args.upload_subdirectory,
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
