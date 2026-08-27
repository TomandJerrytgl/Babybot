"""Unlimited local object memory shared by Awake and Dreaming modes."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid

import cv2
import numpy as np

from conscious import feature_similarity


SOURCE_PRIORITY = {"dreaming": 10, "awake": 100}


class SharedMemory:
    def __init__(self, root="memory", similarity_threshold=0.86,
                 duplicate_threshold=0.985):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.objects_root = self.root / "objects"
        self.objects_root.mkdir(exist_ok=True)
        self.database_path = self.root / "memory.sqlite3"
        self.lock_path = self.root / ".memory-write.lock"
        self.similarity_threshold = float(similarity_threshold)
        self.duplicate_threshold = float(duplicate_threshold)
        self._initialize()
        self._migrate_legacy_manifest()

    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self):
        with self.connection() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS objects(
                    object_id TEXT PRIMARY KEY,
                    source_priority INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS samples(
                    sample_id TEXT PRIMARY KEY,
                    object_id TEXT NOT NULL REFERENCES objects(object_id),
                    source_mode TEXT NOT NULL,
                    eye_mode TEXT NOT NULL,
                    left_image TEXT,
                    right_image TEXT,
                    feature_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    signature TEXT NOT NULL UNIQUE,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS decisions(
                    decision_id TEXT PRIMARY KEY,
                    first_object_id TEXT,
                    second_object_id TEXT,
                    decision TEXT NOT NULL,
                    source_mode TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
            """)

    @contextmanager
    def write_lock(self, timeout=10.0):
        deadline = time.monotonic() + timeout
        descriptor = None
        while descriptor is None:
            try:
                descriptor = os.open(
                    self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
                )
                os.write(descriptor, f"{os.getpid()}\n".encode())
            except FileExistsError:
                try:
                    if time.time() - self.lock_path.stat().st_mtime > 300:
                        self.lock_path.unlink()
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError("Local Memory is busy in another process")
                time.sleep(0.05)
        try:
            yield
        finally:
            os.close(descriptor)
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass

    def counts(self):
        with self.connection() as connection:
            return {
                "objects": connection.execute("SELECT COUNT(*) FROM objects").fetchone()[0],
                "samples": connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0],
                "decisions": connection.execute("SELECT COUNT(*) FROM decisions").fetchone()[0],
            }

    def find_similar(self, feature):
        query = np.asarray(feature, dtype=np.float32)
        best = None
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT o.object_id,o.source_priority,s.feature_json,s.left_image,s.right_image "
                "FROM objects o JOIN samples s ON s.object_id=o.object_id"
            )
            for row in rows:
                score = feature_similarity(query, json.loads(row["feature_json"]))
                if best is None or score > best["similarity"]:
                    best = {"object_id": row["object_id"],
                            "source_priority": row["source_priority"],
                            "similarity": score,
                            "left_image": row["left_image"],
                            "right_image": row["right_image"]}
        return best

    def add_object(self, left, right, feature, metadata, source_mode="dreaming",
                   eye_mode="stereo"):
        identifier = f"object_{uuid.uuid4().hex}"
        now = time.time()
        with self.write_lock(), self.connection() as connection:
            connection.execute(
                "INSERT INTO objects VALUES(?,?,?,?)",
                (identifier, SOURCE_PRIORITY[source_mode], now, now),
            )
            self._insert_sample(connection, identifier, left, right, feature,
                                metadata, source_mode, eye_mode)
        return identifier

    def add_sample(self, object_id, left, right, feature, metadata,
                   source_mode="dreaming", eye_mode="stereo"):
        with self.write_lock(), self.connection() as connection:
            row = connection.execute(
                "SELECT source_priority FROM objects WHERE object_id=?", (object_id,)
            ).fetchone()
            if row is None:
                raise KeyError(object_id)
            # Lower-priority Dreaming evidence can enrich but never demote Awake identity.
            priority = max(int(row[0]), SOURCE_PRIORITY[source_mode])
            connection.execute(
                "UPDATE objects SET source_priority=?,updated_at=? WHERE object_id=?",
                (priority, time.time(), object_id),
            )
            return self._insert_sample(connection, object_id, left, right, feature,
                                       metadata, source_mode, eye_mode)

    def record_decision(self, first_object_id, second_object_id, decision,
                        metadata=None, source_mode="dreaming"):
        with self.write_lock(), self.connection() as connection:
            connection.execute(
                "INSERT INTO decisions VALUES(?,?,?,?,?,?,?)",
                (uuid.uuid4().hex, first_object_id, second_object_id, decision,
                 source_mode, json.dumps(metadata or {}), time.time()),
            )

    def _insert_sample(self, connection, object_id, left, right, feature,
                       metadata, source_mode, eye_mode):
        feature_array = np.asarray(feature, dtype=np.float32)
        signature = hashlib.sha256(
            object_id.encode() + np.round(feature_array, 3).tobytes()
            + json.dumps(metadata, sort_keys=True, default=str).encode()
        ).hexdigest()
        if connection.execute(
                "SELECT 1 FROM samples WHERE signature=?", (signature,)).fetchone():
            return None
        sample_id = f"sample_{uuid.uuid4().hex}"
        directory = self.objects_root / object_id
        directory.mkdir(parents=True, exist_ok=True)
        left_rel = right_rel = None
        if left is not None:
            left_rel = f"objects/{object_id}/{sample_id}_left.jpg"
            if not cv2.imwrite(str(self.root / left_rel), left):
                raise OSError("Failed to write left Memory sample")
        if right is not None:
            right_rel = f"objects/{object_id}/{sample_id}_right.jpg"
            if not cv2.imwrite(str(self.root / right_rel), right):
                raise OSError("Failed to write right Memory sample")
        connection.execute(
            "INSERT INTO samples VALUES(?,?,?,?,?,?,?,?,?,?)",
            (sample_id, object_id, source_mode, eye_mode, left_rel, right_rel,
             json.dumps(feature_array.astype(float).tolist()),
             json.dumps(metadata, default=str), signature, time.time()),
        )
        return sample_id

    def _migrate_legacy_manifest(self):
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            return
        with self.connection() as connection:
            if connection.execute("SELECT COUNT(*) FROM objects").fetchone()[0]:
                return
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        with self.write_lock(), self.connection() as connection:
            for item in data.get("objects", []):
                object_id = item.get("memory_id") or f"object_{uuid.uuid4().hex}"
                created = float(item.get("created_at", time.time()))
                connection.execute(
                    "INSERT OR IGNORE INTO objects VALUES(?,?,?,?)",
                    (object_id, SOURCE_PRIORITY["awake"], created,
                     float(item.get("updated_at", created))),
                )
                for index, sample in enumerate(item.get("samples", [])):
                    signature = hashlib.sha256(
                        f"legacy:{object_id}:{index}".encode()
                    ).hexdigest()
                    base = self.root / object_id
                    left = base / sample.get("left_image", "")
                    right = base / sample.get("right_image", "")
                    connection.execute(
                        "INSERT OR IGNORE INTO samples VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (f"legacy_{uuid.uuid4().hex}", object_id, "awake", "stereo",
                         left.relative_to(self.root).as_posix() if left.is_file() else None,
                         right.relative_to(self.root).as_posix() if right.is_file() else None,
                         json.dumps(sample.get("feature", [])),
                         json.dumps(sample.get("metadata", {})), signature, created),
                    )
