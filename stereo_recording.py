"""Stereo recording, finalization, and private Git repository synchronization."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time

import cv2


LOGGER = logging.getLogger("babybot.recording")


@dataclass(frozen=True)
class StereoRecordingConfig:
    data_root: str = "recordings"
    camera_fps: float = 60.0
    jpeg_quality: int = 95
    queue_capacity: int = 512
    minimum_free_bytes: int = 2 * 1024 * 1024 * 1024
    upload_repository: str = ""
    upload_subdirectory: str = "babybot/stereo_test_data"


class GitLfsUploader:
    """Copy one finalized batch into an existing private clone and push it."""

    def __init__(self, repository, subdirectory):
        self.repository = Path(repository).expanduser() if repository else None
        self.subdirectory = Path(subdirectory)

    @property
    def configured(self):
        return self.repository is not None

    def upload(self, recording_directory):
        if not self.configured:
            raise RuntimeError("TGLgeneral clone path is not configured")
        if not (self.repository / ".git").exists():
            raise RuntimeError(f"Upload repository is not a Git clone: {self.repository}")
        source = Path(recording_directory)
        target = self.repository / self.subdirectory / source.name
        target.mkdir(parents=True, exist_ok=True)
        for name in ("videos", "pairs.csv", "metadata.json"):
            item = source / name
            if item.exists():
                shutil.copytree(item, target / name) if item.is_dir() else shutil.copy2(item, target / name)
        relative = target.relative_to(self.repository)
        self._git("lfs", "install", "--local")
        self._git("lfs", "track", "*.mp4", "*.avi", "*.zip")
        self._git("add", ".gitattributes", str(relative))
        if self._git_changed(".gitattributes", str(relative)):
            self._git(
                "commit", "-m", f"Add Babybot stereo recording {source.name}",
                "--", ".gitattributes", str(relative),
            )
        self._git("push")
        return str(target)

    def _git_changed(self, *paths):
        process = subprocess.run(
            ["git", "diff", "--cached", "--quiet", "--", *paths],
            cwd=self.repository, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=60,
        )
        if process.returncode not in (0, 1):
            raise RuntimeError("Unable to inspect staged upload changes")
        return process.returncode == 1

    def _git(self, *arguments):
        process = subprocess.run(
            ["git", *arguments], cwd=self.repository, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600,
        )
        if process.returncode:
            raise RuntimeError(f"git {' '.join(arguments)} failed: {process.stdout.strip()}")


class StereoRecorder:
    """Write every submitted stereo pair and finalize it into a training dataset."""

    def __init__(self, config: StereoRecordingConfig):
        self.config = config
        self.uploader = GitLfsUploader(
            config.upload_repository, config.upload_subdirectory
        )
        self._lock = threading.Lock()
        self._upload_lock = threading.Lock()
        self._queue = None
        self._writer = None
        self._finalizer = None
        self._last_upload_batch = None
        self._upload_threads = []
        self._batch = None
        self._csv_stream = None
        self._csv_writer = None
        self._started_wall_ns = None
        self._started_monotonic = None
        self._frame_shape = None
        self._video_fps = config.camera_fps
        self._video_writers = {}
        self._video_outputs = []
        self._count = 0
        self._submitted = 0
        self._sync_delta_total_ns = 0
        self._sync_delta_max_ns = 0
        self._first_frame_monotonic_ns = None
        self._last_frame_monotonic_ns = None
        self._stopped_wall_ns = None
        self._stopped_monotonic_ns = None
        self._status = {
            "state": "idle", "recording": False, "message": "Ready to record",
            "paired_frame_count": 0, "duration_seconds": 0.0,
            "batch": None, "upload_configured": self.uploader.configured,
            "upload_state": "idle",
        }

    def status(self):
        with self._lock:
            result = dict(self._status)
            if result["recording"] and self._started_monotonic is not None:
                result["duration_seconds"] = round(
                    time.monotonic() - self._started_monotonic, 1
                )
            result["queued_pairs"] = 0 if self._queue is None else self._queue.qsize()
            return result

    def start(self, frame_shape, video_fps=None):
        with self._lock:
            if self._status["state"] != "idle":
                return False
            now = datetime.now(timezone.utc)
            base = Path(self.config.data_root)
            base.mkdir(parents=True, exist_ok=True)
            name = now.strftime("recording_%Y%m%dT%H%M%S_%fZ")
            batch = base / name
            (batch / "videos").mkdir(parents=True)
            stream = (batch / "pairs.csv").open("w", encoding="utf-8", newline="")
            writer = csv.DictWriter(stream, fieldnames=(
                "index", "timestamp_ns", "monotonic_ns", "sync_delta_ns"
            ))
            writer.writeheader()
            self._batch = batch
            self._csv_stream = stream
            self._csv_writer = writer
            self._started_wall_ns = time.time_ns()
            self._started_monotonic = time.monotonic()
            self._frame_shape = tuple(int(value) for value in frame_shape)
            requested_fps = self.config.camera_fps if video_fps is None else float(video_fps)
            self._video_fps = max(1.0, requested_fps)
            self._video_writers = {}
            self._video_outputs = []
            self._count = 0
            self._submitted = 0
            self._sync_delta_total_ns = 0
            self._sync_delta_max_ns = 0
            self._first_frame_monotonic_ns = None
            self._last_frame_monotonic_ns = None
            self._stopped_wall_ns = None
            self._stopped_monotonic_ns = None
            self._queue = queue.Queue(maxsize=self.config.queue_capacity)
            self._status.update({
                "state": "recording", "recording": True, "message": "Recording",
                "paired_frame_count": 0, "duration_seconds": 0.0,
                "batch": name, "error": None, "upload_path": None,
                "video_fps": round(self._video_fps, 3),
            })
            self._writer = threading.Thread(
                target=self._writer_loop, name="stereo-frame-writer", daemon=True
            )
            self._writer.start()
            return True

    def submit(self, left, right, timestamp_ns=None, monotonic_ns=None,
               sync_delta_ns=0):
        with self._lock:
            if not self._status["recording"]:
                return False
            item = (
                self._submitted,
                int(time.time_ns() if timestamp_ns is None else timestamp_ns),
                int(time.monotonic_ns() if monotonic_ns is None else monotonic_ns),
                int(sync_delta_ns),
                left.copy(), right.copy(),
            )
            self._submitted += 1
            work_queue = self._queue
        try:
            work_queue.put_nowait(item)
            return True
        except queue.Full:
            self._fail_and_stop("Recording stopped because the frame writer could not keep up")
            return False

    def stop_async(self):
        with self._lock:
            if not self._status["recording"]:
                return False
            self._status.update({
                "state": "stopping", "recording": False,
                "message": "Finishing queued frames",
            })
            self._stopped_wall_ns = time.time_ns()
            self._stopped_monotonic_ns = time.monotonic_ns()
            work_queue = self._queue
            self._finalizer = threading.Thread(
                target=self._finish, args=(work_queue,), name="stereo-finalizer", daemon=True
            )
            self._finalizer.start()
            return True

    def retry_upload(self):
        with self._lock:
            if self._status["upload_state"] != "failed" or self._last_upload_batch is None:
                return False
            batch = self._last_upload_batch
            self._status.update({"upload_state": "queued", "upload_message": "Retry queued", "upload_error": None})
        self._start_upload_thread(batch)
        return True

    def shutdown(self, timeout=30.0):
        self.stop_async()
        worker = self._finalizer
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=timeout)
        deadline = time.monotonic() + timeout
        with self._lock:
            upload_threads = list(self._upload_threads)
        for upload_thread in upload_threads:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            upload_thread.join(timeout=remaining)

    def _writer_loop(self):
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            index, wall_ns, monotonic_ns, sync_delta_ns, left, right = item
            try:
                if index % 60 == 0:
                    free = shutil.disk_usage(self._batch).free
                    if free < self.config.minimum_free_bytes:
                        raise OSError(
                            f"Only {free} bytes remain; the recording reserve is "
                            f"{self.config.minimum_free_bytes} bytes"
                        )
                if not self._video_writers:
                    self._open_video_writers()
                self._video_writers["left"].write(left)
                self._video_writers["right"].write(right)
                self._csv_writer.writerow({
                    "index": index, "timestamp_ns": wall_ns, "monotonic_ns": monotonic_ns,
                    "sync_delta_ns": sync_delta_ns,
                })
                self._csv_stream.flush()
            except Exception as error:
                self._queue.task_done()
                self._fail_and_stop(str(error))
                return
            with self._lock:
                self._count += 1
                self._sync_delta_total_ns += sync_delta_ns
                self._sync_delta_max_ns = max(self._sync_delta_max_ns, sync_delta_ns)
                if self._first_frame_monotonic_ns is None:
                    self._first_frame_monotonic_ns = monotonic_ns
                self._last_frame_monotonic_ns = monotonic_ns
                self._status["paired_frame_count"] = self._count
            self._queue.task_done()

    def _fail_and_stop(self, message):
        LOGGER.error(message)
        with self._lock:
            self._status["error"] = message
            recording = self._status["recording"]
        if recording:
            self.stop_async()

    def _finish(self, work_queue):
        work_queue.put(None)
        self._writer.join()
        self._csv_stream.close()
        for writer in self._video_writers.values():
            writer.release()
        self._video_writers = {}
        with self._lock:
            self._status.update({"state": "processing", "message": "Finalizing video metadata"})
        try:
            self._write_metadata(self._video_outputs)
        except Exception as error:
            LOGGER.exception("Recording finalization failed")
            self._write_metadata([])
            with self._lock:
                self._status.update({"state": "idle", "message": "Local processing failed; data retained", "error": str(error)})
            return
        if not self.uploader.configured:
            with self._lock:
                self._status.update({
                    "state": "idle", "message": "Saved locally; upload repository is not configured",
                    "upload_state": "failed", "upload_message": "Upload repository is not configured",
                    "upload_error": "Set --upload-repo to the private TGLgeneral clone",
                })
                self._last_upload_batch = self._batch
            return
        batch = self._batch
        with self._lock:
            self._status.update({"state": "idle", "message": "Saved locally; upload queued", "upload_state": "queued"})
            self._last_upload_batch = batch
        self._start_upload_thread(batch)

    def _open_video_writers(self):
        height, width = self._frame_shape[:2]
        attempts = (("mp4v", ".mp4"), ("MJPG", ".avi"))
        outputs = []
        for eye in ("left", "right"):
            writer = None
            path = None
            codec = None
            for candidate, suffix in attempts:
                candidate_path = self._batch / "videos" / f"{eye}{suffix}"
                candidate_writer = cv2.VideoWriter(
                    str(candidate_path), cv2.VideoWriter_fourcc(*candidate),
                    float(self._video_fps), (width, height),
                )
                if candidate_writer.isOpened():
                    writer, path, codec = candidate_writer, candidate_path, candidate
                    break
                candidate_writer.release()
            if writer is None:
                raise RuntimeError(f"No supported video encoder for {eye} eye")
            self._video_writers[eye] = writer
            outputs.append({"eye": eye, "path": path.relative_to(self._batch).as_posix(), "codec": codec})
        self._video_outputs = outputs

    def _write_metadata(self, videos):
        processed_at_ns = time.time_ns()
        stopped_ns = self._stopped_wall_ns or processed_at_ns
        capture_duration = 0.0
        if (self._first_frame_monotonic_ns is not None
                and self._last_frame_monotonic_ns is not None):
            capture_duration = max(
                0.0,
                (self._last_frame_monotonic_ns - self._first_frame_monotonic_ns) / 1e9,
            )
        effective_fps = (
            (self._count - 1) / capture_duration
            if self._count > 1 and capture_duration > 0 else 0.0
        )
        metadata = {
            "schema": "babybot.stereo-recording/v2",
            "batch": self._batch.name,
            "started_at_ns": self._started_wall_ns,
            "stopped_at_ns": stopped_ns,
            "processed_at_ns": processed_at_ns,
            "requested_recording_duration_seconds": round(
                (stopped_ns - self._started_wall_ns) / 1e9, 6
            ),
            "capture_duration_seconds": round(capture_duration, 6),
            "postprocessing_duration_seconds": round(
                max(0, processed_at_ns - stopped_ns) / 1e9, 6
            ),
            "paired_frame_count": self._count,
            "submitted_frame_count": self._submitted,
            "width": self._frame_shape[1], "height": self._frame_shape[0],
            "channels": self._frame_shape[2] if len(self._frame_shape) > 2 else 1,
            "configured_fps": self.config.camera_fps,
            "effective_capture_fps": round(effective_fps, 6),
            "video_fps": round(self._video_fps, 6),
            "frame_storage": "paired video streams",
            "pairing": "same grab/retrieve capture cycle",
            "sync_delta_kind": "software gap between completion of left and right grab calls",
            "sync_delta_max_ns": self._sync_delta_max_ns,
            "sync_delta_mean_ns": (
                round(self._sync_delta_total_ns / self._count, 3) if self._count else 0
            ),
            "videos": videos,
            "error": self._status.get("error"),
        }
        temporary = self._batch / "metadata.json.tmp"
        temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        temporary.replace(self._batch / "metadata.json")

    def _upload(self, batch):
        with self._upload_lock:
            with self._lock:
                self._status.update({"upload_state": "uploading", "upload_message": f"Uploading {batch.name}"})
            try:
                upload_path = self.uploader.upload(batch)
            except Exception as error:
                LOGGER.exception("Recording upload failed")
                with self._lock:
                    self._last_upload_batch = batch
                    self._status.update({
                        "upload_state": "failed", "upload_message": "Upload failed; local data retained",
                        "upload_error": str(error),
                    })
                return
            with self._lock:
                self._status.update({
                    "upload_state": "complete", "upload_message": f"Uploaded {batch.name}",
                    "upload_path": upload_path, "upload_error": None,
                })

    def _start_upload_thread(self, batch):
        worker = threading.Thread(
            target=self._upload, args=(batch,), name="stereo-uploader", daemon=True
        )
        with self._lock:
            self._upload_threads = [item for item in self._upload_threads if item.is_alive()]
            self._upload_threads.append(worker)
        worker.start()
