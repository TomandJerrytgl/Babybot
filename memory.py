"""Persistent, diversity-preserving visual memory for Babybot."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from conscious import feature_similarity


class MemoryStore:
    def __init__(self, root="memory", maximum_objects=20, samples_per_object=3, similarity_threshold=0.86):
        self.root = Path(root)
        self.maximum_objects = int(maximum_objects)
        self.samples_per_object = int(samples_per_object)
        self.similarity_threshold = float(similarity_threshold)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.manifest = self._load_manifest()

    @property
    def object_count(self):
        return len(self.manifest["objects"])

    @property
    def learning_stopped(self):
        return bool(self.manifest.get("learning_stopped", False))

    def _load_manifest(self):
        if not self.manifest_path.exists():
            return {"version": 1, "learning_stopped": False, "objects": []}
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        data.setdefault("version", 1)
        data.setdefault("learning_stopped", False)
        data.setdefault("objects", [])
        return data

    def _save_manifest(self):
        temporary = self.manifest_path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.manifest, handle, ensure_ascii=False, indent=2)
        temporary.replace(self.manifest_path)

    @staticmethod
    def _ratio_similarity(first, second):
        if first is None or second is None or first <= 0 or second <= 0:
            return 0.5
        return float(np.exp(-abs(np.log(float(first) / float(second)))))

    @classmethod
    def perceptual_similarity(cls, feature, spatial, sample):
        """Identity score: appearance and size dominate; position is negligible."""
        visual = feature_similarity(feature, sample["feature"])
        if not spatial:
            return visual
        previous = sample.get("metadata", {})
        # Memories written before spatial descriptors were introduced remain
        # usable and are upgraded naturally when a new sample is learned.
        if "width_fraction" not in previous:
            return visual
        size = np.mean([
            cls._ratio_similarity(spatial.get("width_fraction"), previous.get("width_fraction")),
            cls._ratio_similarity(spatial.get("height_fraction"), previous.get("height_fraction")),
            cls._ratio_similarity(spatial.get("area_fraction"), previous.get("area_fraction")),
        ])
        position = np.mean([
            cls._ratio_similarity(1.0 + spatial.get("relative_x", 0.0),
                                  1.0 + previous.get("relative_x", 0.0)),
            cls._ratio_similarity(1.0 + spatial.get("relative_y", 0.0),
                                  1.0 + previous.get("relative_y", 0.0)),
        ])
        geometry = float(np.clip(
            min(spatial.get("geometry_confidence", 0.5),
                previous.get("geometry_confidence", 0.5)), 0.0, 1.0
        ))
        return float(np.clip(
            0.60 * visual + 0.32 * size + 0.03 * position + 0.05 * geometry,
            0.0, 1.0,
        ))

    def find_similar(self, feature, spatial=None) -> tuple[Optional[dict], float]:
        best, best_score = None, 0.0
        for item in self.manifest["objects"]:
            for sample in item.get("samples", []):
                score = self.perceptual_similarity(feature, spatial, sample)
                if score > best_score:
                    best, best_score = item, score
        if best_score < self.similarity_threshold:
            return None, best_score
        return best, best_score

    def learn(self, left, right, feature, metadata):
        """Create/update memory and return (memory_id, created, similarity)."""
        similar, similarity = self.find_similar(feature, metadata)
        if similar is None:
            if self.object_count >= self.maximum_objects:
                self.manifest["learning_stopped"] = True
                self._save_manifest()
                return None, False, similarity
            identifier = f"object_{self.object_count + 1:03d}"
            similar = {
                "memory_id": identifier,
                "created_at": metadata["timestamp"],
                "updated_at": metadata["timestamp"],
                "samples": [],
            }
            self.manifest["objects"].append(similar)
            created = True
        else:
            identifier = similar["memory_id"]
            created = False

        candidates = self._existing_samples(similar)
        candidates.append({
            "feature": np.asarray(feature, dtype=np.float32),
            "left": left.copy(),
            "right": right.copy(),
            "metadata": dict(metadata),
        })
        chosen = self._most_diverse(candidates, self.samples_per_object)
        self._write_samples(similar, chosen)
        similar["updated_at"] = metadata["timestamp"]
        if self.object_count >= self.maximum_objects:
            self.manifest["learning_stopped"] = True
        self._save_manifest()
        return identifier, created, similarity

    def _existing_samples(self, item):
        output = []
        object_dir = self.root / item["memory_id"]
        for sample in item.get("samples", []):
            left = cv2.imread(str(object_dir / sample["left_image"]))
            right = cv2.imread(str(object_dir / sample["right_image"]))
            if left is None or right is None:
                continue
            output.append({
                "feature": np.asarray(sample["feature"], dtype=np.float32),
                "left": left,
                "right": right,
                "metadata": sample.get("metadata", {}),
            })
        return output

    @staticmethod
    def _most_diverse(samples, maximum):
        if len(samples) <= maximum:
            return samples
        best_group, best_diversity = None, -1.0
        for group in itertools.combinations(samples, maximum):
            diversity = sum(
                1.0 - feature_similarity(first["feature"], second["feature"])
                for first, second in itertools.combinations(group, 2)
            )
            if diversity > best_diversity:
                best_group, best_diversity = group, diversity
        return list(best_group)

    def _write_samples(self, item, samples):
        object_dir = self.root / item["memory_id"]
        object_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for index, sample in enumerate(samples, start=1):
            left_name = f"sample_{index:03d}_left.jpg"
            right_name = f"sample_{index:03d}_right.jpg"
            if not cv2.imwrite(str(object_dir / left_name), sample["left"]):
                raise RuntimeError("Failed to save left memory sample")
            if not cv2.imwrite(str(object_dir / right_name), sample["right"]):
                raise RuntimeError("Failed to save right memory sample")
            records.append({
                "left_image": left_name,
                "right_image": right_name,
                "feature": np.asarray(sample["feature"], dtype=float).tolist(),
                "metadata": sample["metadata"],
            })
        item["samples"] = records
        with (object_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(item, handle, ensure_ascii=False, indent=2)
