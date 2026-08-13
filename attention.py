"""Low-latency motion-first visual attention for Babybot."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


Window = Tuple[int, int, int, int]


@dataclass(frozen=True)
class TrackingResult:
    window: Optional[Window]
    confidence: float
    elapsed_time: float
    search_scale: float


@dataclass(frozen=True)
class VisualSignature:
    brightness: float
    contrast: float
    color: float
    edge_density: float
    visual_score: float
    aspect_ratio: float
    area_fraction: float


@dataclass(frozen=True)
class AttentionReference:
    observation_id: int
    timestamp: float
    window: Window
    signature: VisualSignature


@dataclass(frozen=True)
class ValidationResult:
    window: Optional[Window]
    similarity: float
    elapsed_time: float


class AttentionValidator:
    """Confirm an attention reference using visual features near its last box."""

    def __init__(
        self,
        reference: AttentionReference,
        similarity_threshold=0.78,
        search_scale=3.0,
        offsets=(-0.5, 0.0, 0.5),
        size_scales=(0.9, 1.0, 1.1),
    ):
        self.reference = reference
        self.similarity_threshold = float(similarity_threshold)
        self.search_scale = float(search_scale)
        self.offsets = tuple(float(value) for value in offsets)
        self.size_scales = tuple(float(value) for value in size_scales)
        self.last_confirmed_window = reference.window

    def replace_reference(self, reference: AttentionReference) -> None:
        self.reference = reference
        self.last_confirmed_window = reference.window

    def validate(self, image) -> ValidationResult:
        start = time.perf_counter()
        Attention._validate_image(image)
        search = TemplateTracker.expanded_window(
            self.last_confirmed_window, image.shape, self.search_scale
        )
        sx, sy, sw, sh = search
        base_x, base_y, base_width, base_height = self.last_confirmed_window
        center_x = base_x + base_width / 2.0
        center_y = base_y + base_height / 2.0
        best_window = None
        best_similarity = -1.0

        for size_scale in self.size_scales:
            width = max(8, int(round(base_width * size_scale)))
            height = max(8, int(round(base_height * size_scale)))
            if width > sw or height > sh:
                continue
            for offset_y in self.offsets:
                for offset_x in self.offsets:
                    candidate_x = int(round(center_x + offset_x * base_width - width / 2.0))
                    candidate_y = int(round(center_y + offset_y * base_height - height / 2.0))
                    candidate_x = int(np.clip(candidate_x, sx, sx + sw - width))
                    candidate_y = int(np.clip(candidate_y, sy, sy + sh - height))
                    window = (candidate_x, candidate_y, width, height)
                    signature = visual_signature(image, window)
                    similarity = signature_similarity(self.reference.signature, signature)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_window = window

        confirmed = best_window if best_similarity >= self.similarity_threshold else None
        if confirmed is not None:
            self.last_confirmed_window = confirmed
        return ValidationResult(
            window=confirmed,
            similarity=max(0.0, float(best_similarity)),
            elapsed_time=time.perf_counter() - start,
        )


def visual_signature(image, window: Window) -> VisualSignature:
    x, y, width, height = TemplateTracker.clip_window(window, image.shape)
    patch = image[y:y + height, x:x + width]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(hsv[:, :, 2]) / 255.0)
    color = float(np.mean(hsv[:, :, 1]) / 255.0)
    contrast = float(np.clip(np.std(gray) / 128.0, 0.0, 1.0))
    edges = cv2.Canny(gray, 60, 160)
    edge_density = float(np.mean(edges > 0))
    visual_score = float(np.clip(
        0.20 * brightness + 0.30 * contrast + 0.25 * color
        + 0.25 * np.clip(edge_density / 0.15, 0.0, 1.0),
        0.0,
        1.0,
    ))
    image_area = image.shape[0] * image.shape[1]
    return VisualSignature(
        brightness=brightness,
        contrast=contrast,
        color=color,
        edge_density=edge_density,
        visual_score=visual_score,
        aspect_ratio=float(width / max(1, height)),
        area_fraction=float(width * height / max(1, image_area)),
    )


def signature_similarity(reference: VisualSignature, candidate: VisualSignature) -> float:
    def closeness(first, second, scale):
        return float(np.clip(1.0 - abs(first - second) / scale, 0.0, 1.0))

    values = (
        (0.15, closeness(reference.brightness, candidate.brightness, 0.35)),
        (0.20, closeness(reference.contrast, candidate.contrast, 0.35)),
        (0.20, closeness(reference.color, candidate.color, 0.40)),
        (0.20, closeness(reference.edge_density, candidate.edge_density, 0.18)),
        (0.20, closeness(reference.visual_score, candidate.visual_score, 0.35)),
        (0.025, closeness(reference.aspect_ratio, candidate.aspect_ratio, 0.50)),
        (0.025, closeness(reference.area_fraction, candidate.area_fraction, 0.08)),
    )
    return float(np.clip(sum(weight * score for weight, score in values), 0.0, 1.0))


def make_attention_reference(observation, eye, attention) -> Optional[AttentionReference]:
    if attention.focus is None:
        return None
    image = getattr(observation, eye)
    window = TemplateTracker.clip_window(attention.focus, image.shape)
    return AttentionReference(
        observation_id=observation.observation_id,
        timestamp=observation.timestamp,
        window=window,
        signature=visual_signature(image, window),
    )


class TemplateTracker:
    """Relocate a fixed attention template near its last confirmed position."""

    def __init__(
        self,
        source_image,
        window: Window,
        high_confidence=0.75,
        medium_confidence=0.60,
        local_search_scale=3.0,
        expanded_search_scale=5.0,
        template_scales=(0.9, 1.0, 1.1),
    ):
        Attention._validate_image(source_image)
        self.high_confidence = float(high_confidence)
        self.medium_confidence = float(medium_confidence)
        self.local_search_scale = float(local_search_scale)
        self.expanded_search_scale = float(expanded_search_scale)
        self.template_scales = tuple(float(scale) for scale in template_scales)
        self.window = self.clip_window(window, source_image.shape)
        x, y, width, height = self.window
        self.template = source_image[y:y + height, x:x + width].copy()
        if self.template.size == 0 or width < 4 or height < 4:
            raise ValueError("Attention window does not contain a usable template")

    def locate(self, image) -> TrackingResult:
        start = time.perf_counter()
        best = self._search(image, self.local_search_scale)
        search_scale = self.local_search_scale
        if best[1] < self.high_confidence and best[1] >= self.medium_confidence:
            best = self._search(image, self.expanded_search_scale)
            search_scale = self.expanded_search_scale
        window, confidence = best
        if confidence >= self.high_confidence and window is not None:
            self.window = window
        else:
            window = None
        return TrackingResult(
            window=window,
            confidence=float(confidence),
            elapsed_time=time.perf_counter() - start,
            search_scale=search_scale,
        )

    def _search(self, image, search_scale):
        Attention._validate_image(image)
        search_window = self.expanded_window(self.window, image.shape, search_scale)
        sx, sy, sw, sh = search_window
        search_image = image[sy:sy + sh, sx:sx + sw]
        best_window = None
        best_confidence = -1.0

        for scale in self.template_scales:
            width = max(4, int(round(self.template.shape[1] * scale)))
            height = max(4, int(round(self.template.shape[0] * scale)))
            if width > sw or height > sh:
                continue
            template = cv2.resize(self.template, (width, height), interpolation=cv2.INTER_AREA)
            # SQDIFF_NORMED remains meaningful for low-texture templates.
            response = cv2.matchTemplate(search_image, template, cv2.TM_SQDIFF_NORMED)
            minimum, _maximum, location, _maximum_location = cv2.minMaxLoc(response)
            confidence = float(np.clip(1.0 - minimum, 0.0, 1.0))
            if confidence > best_confidence:
                best_confidence = confidence
                best_window = self.clip_window(
                    (sx + location[0], sy + location[1], width, height),
                    image.shape,
                )
        return best_window, max(0.0, best_confidence)

    @staticmethod
    def expanded_window(window, image_shape, scale):
        image_height, image_width = image_shape[:2]
        x, y, width, height = window
        expanded_width = min(image_width, max(width, int(round(width * scale))))
        expanded_height = min(image_height, max(height, int(round(height * scale))))
        center_x = x + width / 2.0
        center_y = y + height / 2.0
        expanded_x = int(np.clip(round(center_x - expanded_width / 2.0), 0, image_width - expanded_width))
        expanded_y = int(np.clip(round(center_y - expanded_height / 2.0), 0, image_height - expanded_height))
        return expanded_x, expanded_y, expanded_width, expanded_height

    @staticmethod
    def clip_window(window, image_shape):
        image_height, image_width = image_shape[:2]
        x, y, width, height = (int(value) for value in window)
        x = int(np.clip(x, 0, max(0, image_width - 1)))
        y = int(np.clip(y, 0, max(0, image_height - 1)))
        width = int(np.clip(width, 1, image_width - x))
        height = int(np.clip(height, 1, image_height - y))
        return x, y, width, height


class Attention:
    """Prefer novel motion, briefly retain focus, then use static saliency."""

    def __init__(
        self,
        observation,
        object=None,
        eye="left",
        previous_focus: Optional[Window] = None,
        previous_image=None,
        previous_focus_age=0,
        verbose=True,
        analysis_width=320,
        hold_observations=10,
    ):
        del object  # Kept in the signature for compatibility with older callers.
        if eye not in ("left", "right"):
            raise ValueError("eye must be 'left' or 'right'")
        self.observation = observation
        self.eye = eye
        self.previous_focus = previous_focus
        self.previous_image = previous_image
        self.previous_focus_age = int(previous_focus_age)
        self.verbose = verbose
        self.analysis_width = int(analysis_width)
        self.hold_observations = int(hold_observations)
        self.candidates = []
        self.elapsed_time = 0.0
        self.focus_source = "none"

        image = getattr(observation, eye)
        self.focus = self.find_attention_window(image)

    def find_attention_window(self, image):
        start = time.perf_counter()
        self._validate_image(image)
        candidates = []

        if self.previous_image is not None:
            candidates = self.motion_candidates(image, self.previous_image)

        if candidates:
            self.focus_source = "motion"
        elif (
            self.previous_focus is not None
            and self.previous_focus_age < self.hold_observations
        ):
            candidates = [self.retained_candidate(image, self.previous_focus)]
            self.focus_source = "retained"
        else:
            candidates = self.static_candidates(image)
            self.focus_source = "static" if candidates else "none"

        self.candidates = candidates[:5]
        self.elapsed_time = time.perf_counter() - start
        if self.verbose:
            self.print_results()
        return self.candidates[0]["window"] if self.candidates else None

    def motion_candidates(self, current, previous):
        if previous.shape[:2] != current.shape[:2]:
            return []
        current_small, scale_x, scale_y = self.analysis_image(current)
        previous_small = cv2.resize(
            previous,
            (current_small.shape[1], current_small.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        current_gray = cv2.cvtColor(current_small, cv2.COLOR_BGR2GRAY)
        previous_gray = cv2.cvtColor(previous_small, cv2.COLOR_BGR2GRAY)

        # Removing the median signed difference suppresses global exposure shifts.
        signed = current_gray.astype(np.int16) - previous_gray.astype(np.int16)
        signed -= int(np.median(signed))
        difference = np.abs(signed).astype(np.uint8)
        difference = cv2.GaussianBlur(difference, (5, 5), 0)
        noise_level = float(np.median(difference))
        threshold = int(np.clip(noise_level + 18.0, 18.0, 45.0))
        mask = np.where(difference >= threshold, 255, 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        changed_fraction = float(np.mean(mask > 0))
        # A near-global change is normally exposure adjustment or camera movement.
        if changed_fraction < 0.002 or changed_fraction > 0.55:
            return []

        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask)
        analysis_area = mask.shape[0] * mask.shape[1]
        minimum_area = max(40, int(analysis_area * 0.0025))
        candidates = []
        for label in range(1, count):
            x, y, width, height, area = stats[label]
            if area < minimum_area:
                continue
            window = self.map_and_pad_window(
                (int(x), int(y), int(width), int(height)),
                scale_x,
                scale_y,
                current.shape,
            )
            region_mask = labels[y:y + height, x:x + width] == label
            region_difference = difference[y:y + height, x:x + width]
            intensity = float(np.mean(region_difference[region_mask]) / 255.0)
            area_score = float(np.clip(area / (analysis_area * 0.12), 0.0, 1.0))
            center = self.center_preference(window, current.shape)
            score = float(np.clip(0.50 * intensity + 0.40 * area_score + 0.10 * center, 0.0, 1.0))
            candidates.append(
                self.make_candidate(current, window, score, motion=intensity, source="motion")
            )

        return self.suppress_overlaps(candidates, maximum_candidates=5)

    def static_candidates(self, image):
        """Cheap low-resolution fallback; motion candidates always take priority."""
        small, scale_x, scale_y = self.analysis_image(image)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.float32) / 255.0
        local_mean = cv2.GaussianBlur(gray, (31, 31), 0)
        local_contrast = cv2.absdiff(gray, local_mean)
        edges = cv2.Canny((gray * 255).astype(np.uint8), 60, 160).astype(np.float32) / 255.0
        saliency = 0.45 * local_contrast + 0.30 * saturation + 0.25 * edges
        saliency = cv2.GaussianBlur(saliency, (21, 21), 0)

        height, width = gray.shape
        window_size = max(24, int(min(height, width) * 0.20))
        step = max(12, window_size // 2)
        candidates = []
        for y in range(0, max(1, height - window_size + 1), step):
            for x in range(0, max(1, width - window_size + 1), step):
                patch = saliency[y:y + window_size, x:x + window_size]
                if patch.size == 0:
                    continue
                window = self.map_window(
                    (x, y, window_size, window_size),
                    scale_x,
                    scale_y,
                    image.shape,
                )
                visual = float(np.clip(np.mean(patch) * 4.0, 0.0, 1.0))
                center = self.center_preference(window, image.shape)
                score = 0.90 * visual + 0.10 * center
                candidates.append(
                    self.make_candidate(image, window, score, source="static")
                )
        return self.suppress_overlaps(candidates, maximum_candidates=5)

    def retained_candidate(self, image, window):
        window = self.clip_window(window, image.shape)
        # A small decay exposes stale focus without causing immediate flicker.
        decay = max(0.0, 1.0 - self.previous_focus_age / max(1, self.hold_observations))
        score = 0.35 + 0.35 * decay
        return self.make_candidate(image, window, score, source="retained")

    def analysis_image(self, image):
        height, width = image.shape[:2]
        target_width = min(width, self.analysis_width)
        target_height = max(1, int(round(height * target_width / width)))
        if target_width == width:
            return image, 1.0, 1.0
        small = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        return small, width / target_width, height / target_height

    def map_window(self, window, scale_x, scale_y, image_shape):
        x, y, width, height = window
        mapped = (
            int(round(x * scale_x)),
            int(round(y * scale_y)),
            max(1, int(round(width * scale_x))),
            max(1, int(round(height * scale_y))),
        )
        return self.clip_window(mapped, image_shape)

    def map_and_pad_window(self, window, scale_x, scale_y, image_shape):
        x, y, width, height = self.map_window(window, scale_x, scale_y, image_shape)
        padding_x = max(8, int(width * 0.20))
        padding_y = max(8, int(height * 0.20))
        return self.clip_window(
            (x - padding_x, y - padding_y, width + 2 * padding_x, height + 2 * padding_y),
            image_shape,
        )

    def clip_window(self, window, image_shape):
        image_height, image_width = image_shape[:2]
        x, y, width, height = (int(value) for value in window)
        x = int(np.clip(x, 0, max(0, image_width - 1)))
        y = int(np.clip(y, 0, max(0, image_height - 1)))
        width = int(np.clip(width, 1, image_width - x))
        height = int(np.clip(height, 1, image_height - y))
        return x, y, width, height

    def center_preference(self, window, image_shape):
        image_height, image_width = image_shape[:2]
        x, y, width, height = window
        dx = (x + width / 2.0 - image_width / 2.0) / max(1.0, image_width / 2.0)
        dy = (y + height / 2.0 - image_height / 2.0) / max(1.0, image_height / 2.0)
        return float(np.clip(1.0 - np.hypot(dx, dy) / np.sqrt(2.0), 0.0, 1.0))

    def make_candidate(self, image, window, score, motion=0.0, source="static"):
        x, y, width, height = self.clip_window(window, image.shape)
        patch = image[y:y + height, x:x + width]
        if patch.size:
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            brightness = float(np.mean(hsv[:, :, 2]) / 255.0)
            color = float(np.mean(hsv[:, :, 1]) / 255.0)
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            contrast = float(np.std(gray) / 128.0)
        else:
            brightness = color = contrast = 0.0
        center = self.center_preference((x, y, width, height), image.shape)
        return {
            "window": (x, y, width, height),
            "level": 0,
            "brightness": float(np.clip(brightness, 0.0, 1.0)),
            "contrast": float(np.clip(contrast, 0.0, 1.0)),
            "color": float(np.clip(color, 0.0, 1.0)),
            "visual": float(np.clip(score, 0.0, 1.0)),
            "center": center,
            "motion": float(np.clip(motion, 0.0, 1.0)),
            "source": source,
            "score": float(np.clip(score, 0.0, 1.0)),
        }

    def suppress_overlaps(self, candidates, overlap_threshold=0.65, maximum_candidates=5):
        selected = []
        for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
            if all(
                self.overlap_ratio(candidate["window"], chosen["window"]) < overlap_threshold
                for chosen in selected
            ):
                selected.append(candidate)
                if len(selected) >= maximum_candidates:
                    break
        return selected

    @staticmethod
    def overlap_ratio(first, second):
        ax, ay, aw, ah = first
        bx, by, bw, bh = second
        left, top = max(ax, bx), max(ay, by)
        right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        intersection = max(0, right - left) * max(0, bottom - top)
        smaller = min(aw * ah, bw * bh)
        return float(intersection / smaller) if smaller else 0.0

    @staticmethod
    def _validate_image(image):
        if image is None:
            raise ValueError("Input image is None")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a BGR color image")

    def print_results(self):
        print("---------------------------------------")
        print(f"Attention Source: {self.focus_source}")
        print(f"Attention Time  : {self.elapsed_time * 1000:.2f} ms")
        for rank, candidate in enumerate(self.candidates, start=1):
            print(
                f"#{rank} {candidate['window']} score={candidate['score']:.3f} "
                f"source={candidate['source']}"
            )
        print("---------------------------------------")
