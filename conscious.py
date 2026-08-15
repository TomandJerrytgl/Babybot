"""Short-lived objects currently available to Babybot's attention."""

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


Window = tuple[int, int, int, int]


def crop_window(image, window: Window):
    x, y, width, height = (int(value) for value in window)
    return image[y:y + height, x:x + width].copy()


def visual_feature(image, window: Optional[Window] = None):
    """Compact appearance feature used for conscious and memory comparison."""
    patch = crop_window(image, window) if window is not None else image
    if patch.size == 0:
        raise ValueError("Cannot extract a feature from an empty image")
    resized = cv2.resize(patch, (32, 32), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1)
    structure = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    mean_lab = np.mean(lab.reshape(-1, 3), axis=0).astype(np.float32) / 255.0
    return np.concatenate((mean_lab, histogram.flatten(), structure.flatten())).astype(np.float32)


def feature_similarity(first, second):
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    if first.shape != second.shape or first.size == 0:
        return 0.0
    color_similarity = 1.0 - float(np.mean(np.abs(first[:3] - second[:3])) / 0.35)
    histogram_distance = cv2.compareHist(
        first[3:35], second[3:35], cv2.HISTCMP_BHATTACHARYYA
    )
    structure_similarity = 1.0 - float(np.mean(np.abs(first[35:] - second[35:])) / 0.45)
    return float(np.clip(
        0.25 * color_similarity
        + 0.40 * (1.0 - histogram_distance)
        + 0.35 * structure_similarity,
        0.0,
        1.0,
    ))


@dataclass
class ConsciousObject:
    object_id: str
    left_template: np.ndarray
    right_template: np.ndarray
    feature: np.ndarray
    left_window: Window
    right_window: Window
    created_at: float
    curiosity: float = 1.0
    consecutive_matches: int = 0
    memory_id: Optional[str] = None
    last_similarity: float = 0.0

    def worker_payload(self):
        return {
            "object_id": self.object_id,
            "left_template": self.left_template,
            "right_template": self.right_template,
            "left_window": self.left_window,
            "right_window": self.right_window,
        }

    def matched(self, left_window, right_window, similarity):
        self.left_window = tuple(int(value) for value in left_window)
        self.right_window = tuple(int(value) for value in right_window)
        self.last_similarity = float(similarity)
        self.consecutive_matches += 1

    def missed(self):
        self.last_similarity = 0.0
        self.consecutive_matches = 0

    def reduce_curiosity(self, amount=0.2):
        self.curiosity = max(0.0, self.curiosity - float(amount))


@dataclass
class Conscious:
    capacity: int = 3
    objects: list[ConsciousObject] = field(default_factory=list)

    def add(self, item: ConsciousObject):
        if len(self.objects) >= self.capacity:
            return False
        self.objects.append(item)
        return True

    def remove(self, object_id):
        before = len(self.objects)
        self.objects = [item for item in self.objects if item.object_id != object_id]
        return len(self.objects) != before

    def active(self):
        return self.objects[0] if self.objects else None

    def __bool__(self):
        return bool(self.objects)

    def __len__(self):
        return len(self.objects)
