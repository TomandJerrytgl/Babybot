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


def make_attention_references(observation, eye, attention):
    image = getattr(observation, eye)
    references = []
    for candidate in attention.candidates[:5]:
        window = TemplateTracker.clip_window(candidate["window"], image.shape)
        references.append(AttentionReference(
            observation_id=observation.observation_id,
            timestamp=observation.timestamp,
            window=window,
            signature=visual_signature(image, window),
        ))
    return references


def make_attention_reference(observation, eye, attention) -> Optional[AttentionReference]:
    references = make_attention_references(observation, eye, attention)
    return references[0] if references else None


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
    """Objectness-first attention with size-aware motion and center ranking."""

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
        objectness_threshold=0.18,
    ):
        del object, previous_focus, previous_focus_age, hold_observations
        if eye not in ("left", "right"):
            raise ValueError("eye must be 'left' or 'right'")
        self.observation = observation
        self.eye = eye
        self.previous_image = previous_image
        self.verbose = verbose
        self.analysis_width = int(analysis_width)
        self.objectness_threshold = float(objectness_threshold)
        self.candidates = []
        self.elapsed_time = 0.0
        self.focus_source = "none"
        image = getattr(observation, eye)
        self.focus = self.find_attention_window(image)

    def find_attention_window(self, image):
        start = time.perf_counter()
        self._validate_image(image)
        small, scale_x, scale_y = self.analysis_image(image)
        motion_mask, motion_strength, motion_windows = self.motion_evidence(
            small, self.previous_image
        )
        windows = self.static_windows(small.shape)
        windows.extend(motion_windows)
        windows = list(dict.fromkeys(windows))
        candidates = []

        for window in windows:
            candidate = self.evaluate_candidate(
                small, window, motion_mask, motion_strength
            )
            if candidate is None:
                continue
            mapped = self.map_and_pad_window(
                window, scale_x, scale_y, image.shape, padding_fraction=0.10
            )
            candidate["window"] = mapped
            candidates.append(candidate)

        self.candidates = self.suppress_overlaps(
            candidates, overlap_threshold=0.65, maximum_candidates=5
        )
        self.elapsed_time = time.perf_counter() - start
        if self.candidates:
            self.focus_source = self.candidates[0]["source"]
        if self.verbose:
            self.print_results()
        return self.candidates[0]["window"] if self.candidates else None

    def static_windows(self, image_shape):
        height, width = image_shape[:2]
        minimum_side = min(height, width)
        windows = []
        for fraction in (0.15, 0.25, 0.40):
            base = max(24, int(round(minimum_side * fraction)))
            for aspect_ratio in (1.0, 4/3, 3/2, 2.0, 3/4, 2/3, 0.5):
                window_width = min(width, max(12, int(round(base * np.sqrt(aspect_ratio)))))
                window_height = min(height, max(12, int(round(base / np.sqrt(aspect_ratio)))))
                step_x = max(12, window_width // 2)
                step_y = max(12, window_height // 2)
                for y in range(0, max(1, height - window_height + 1), step_y):
                    for x in range(0, max(1, width - window_width + 1), step_x):
                        windows.append((x, y, window_width, window_height))
        return windows

    def motion_evidence(self, current_small, previous_image):
        empty = np.zeros(current_small.shape[:2], dtype=np.uint8)
        if previous_image is None or previous_image.shape[:2] == (0, 0):
            return empty, np.zeros_like(empty, dtype=np.float32), []
        previous_small = cv2.resize(
            previous_image,
            (current_small.shape[1], current_small.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        current_gray = cv2.cvtColor(current_small, cv2.COLOR_BGR2GRAY)
        previous_gray = cv2.cvtColor(previous_small, cv2.COLOR_BGR2GRAY)
        signed = current_gray.astype(np.int16) - previous_gray.astype(np.int16)
        signed -= int(np.median(signed))
        difference = cv2.GaussianBlur(np.abs(signed).astype(np.uint8), (5, 5), 0)
        threshold = int(np.clip(float(np.median(difference)) + 18.0, 18.0, 45.0))
        raw_mask = np.where(difference >= threshold, 255, 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        changed_fraction = float(np.mean(raw_mask > 0))
        if changed_fraction > 0.55:
            return empty, np.zeros_like(empty, dtype=np.float32), []

        count, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask)
        image_area = raw_mask.size
        accepted = np.zeros_like(raw_mask)
        windows = []
        for label in range(1, count):
            x, y, width, height, area = stats[label]
            area_fraction = area / image_area
            if area_fraction < 0.005:
                continue
            accepted[labels == label] = 255
            windows.append((int(x), int(y), int(width), int(height)))
        strength = difference.astype(np.float32) / 255.0
        return accepted, strength, windows

    def evaluate_candidate(self, image, window, motion_mask, motion_strength):
        x, y, width, height = self.clip_window(window, image.shape)
        if width < 8 or height < 8:
            return None
        context_margin = max(4, int(round(max(width, height) * 0.25)))
        cx1, cy1 = max(0, x - context_margin), max(0, y - context_margin)
        cx2 = min(image.shape[1], x + width + context_margin)
        cy2 = min(image.shape[0], y + height + context_margin)
        context = image[cy1:cy2, cx1:cx2]
        patch = image[y:y + height, x:x + width]
        background_mask = np.ones(context.shape[:2], dtype=bool)
        background_mask[y - cy1:y - cy1 + height, x - cx1:x - cx1 + width] = False
        if not np.any(background_mask):
            return None

        patch_lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
        context_lab = cv2.cvtColor(context, cv2.COLOR_BGR2LAB).astype(np.float32)
        color_distance = np.linalg.norm(
            np.mean(patch_lab.reshape(-1, 3), axis=0)
            - np.mean(context_lab[background_mask], axis=0)
        )
        color_separation = float(np.clip(color_distance / 90.0, 0.0, 1.0))

        patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        context_gray = cv2.cvtColor(context, cv2.COLOR_BGR2GRAY)
        brightness_difference = float(np.clip(
            abs(float(np.mean(patch_gray)) - float(np.mean(context_gray[background_mask]))) / 128.0,
            0.0, 1.0,
        ))
        edges = cv2.Canny(context_gray, 60, 160)
        band = max(2, int(round(min(width, height) * 0.08)))
        local_x, local_y = x - cx1, y - cy1
        boundary = np.zeros(context.shape[:2], dtype=bool)
        boundary[max(0, local_y - band):min(context.shape[0], local_y + band), local_x:local_x + width] = True
        boundary[max(0, local_y + height - band):min(context.shape[0], local_y + height + band), local_x:local_x + width] = True
        boundary[local_y:local_y + height, max(0, local_x - band):min(context.shape[1], local_x + band)] = True
        boundary[local_y:local_y + height, max(0, local_x + width - band):min(context.shape[1], local_x + width + band)] = True
        boundary_edges = float(np.mean(edges[boundary] > 0)) if np.any(boundary) else 0.0
        edge_continuity = float(np.clip(boundary_edges / 0.18, 0.0, 1.0))
        internal_consistency = float(np.clip(1.0 - np.std(patch_gray) / 90.0, 0.0, 1.0))
        objectness = (
            0.45 * color_separation + 0.30 * edge_continuity
            + 0.15 * brightness_difference + 0.10 * internal_consistency
        )
        if objectness < self.objectness_threshold:
            return None

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        saturation = float(np.mean(hsv[:, :, 1]) / 255.0)
        color_appeal = float(np.clip(saturation / 0.65, 0.0, 1.0))
        brightness = float(np.mean(hsv[:, :, 2]) / 255.0)
        brightness_appeal = float(np.exp(-((brightness - 0.67) ** 2) / (2 * 0.24 ** 2)))
        appearance = 0.65 * color_appeal + 0.35 * brightness_appeal
        base_score = 0.65 * objectness + 0.35 * appearance

        region_motion = motion_mask[y:y + height, x:x + width] > 0
        motion_area_fraction = float(np.mean(region_motion))
        full_motion_fraction = float(np.sum(region_motion) / motion_mask.size)
        if full_motion_fraction < 0.005:
            size_response = 0.0
        elif full_motion_fraction < 0.02:
            size_response = 0.20 * (full_motion_fraction - 0.005) / 0.015
        elif full_motion_fraction < 0.10:
            size_response = 0.20 + 0.80 * (full_motion_fraction - 0.02) / 0.08
        else:
            size_response = 1.0
        if np.any(region_motion):
            strength = float(np.mean(motion_strength[y:y + height, x:x + width][region_motion]))
        else:
            strength = 0.0
        motion_value = float(np.clip(size_response * (0.5 + 0.5 * strength), 0.0, 1.0))
        center = self.center_preference((x, y, width, height), image.shape)
        motion_bonus = 0.35 * motion_value
        center_bonus = 0.20 * center * (1.0 - size_response)
        score = float(np.clip(0.65 * base_score + motion_bonus + center_bonus, 0.0, 1.0))
        source = "mixed" if motion_value > 0 else "static"
        return self.make_candidate(
            image, (x, y, width, height), score, motion=motion_value,
            source=source, objectness=objectness, appearance=appearance,
            motion_area=motion_area_fraction,
        )

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

    def map_and_pad_window(
            self, window, scale_x, scale_y, image_shape, padding_fraction=0.10):
        x, y, width, height = self.map_window(window, scale_x, scale_y, image_shape)
        padding_x = max(2, int(round(width * padding_fraction)))
        padding_y = max(2, int(round(height * padding_fraction)))
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

    def make_candidate(
            self, image, window, score, motion=0.0, source="static",
            objectness=0.0, appearance=0.0, motion_area=0.0):
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
            "motion_area": float(np.clip(motion_area, 0.0, 1.0)),
            "objectness": float(np.clip(objectness, 0.0, 1.0)),
            "appearance": float(np.clip(appearance, 0.0, 1.0)),
            "source": source,
            "score": float(np.clip(score, 0.0, 1.0)),
        }

    def suppress_overlaps(self, candidates, overlap_threshold=0.65, maximum_candidates=5):
        selected = []
        for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
            if all(self.windows_compatible(
                candidate["window"], chosen["window"], overlap_threshold
            ) for chosen in selected):
                selected.append(candidate)
                if len(selected) >= maximum_candidates:
                    break
        return selected

    @classmethod
    def windows_compatible(cls, first, second, overlap_threshold=0.65):
        if cls.nearly_identical(first, second):
            return False
        if cls.contains(first, second) or cls.contains(second, first):
            return True
        return cls.intersection_area(first, second) == 0

    @staticmethod
    def intersection_area(first, second):
        ax, ay, aw, ah = first
        bx, by, bw, bh = second
        return max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
            0, min(ay + ah, by + bh) - max(ay, by)
        )

    @classmethod
    def contains(cls, outer, inner, tolerance=2):
        ox, oy, ow, oh = outer
        ix, iy, iw, ih = inner
        return (
            ix >= ox - tolerance and iy >= oy - tolerance
            and ix + iw <= ox + ow + tolerance
            and iy + ih <= oy + oh + tolerance
        )

    @classmethod
    def nearly_identical(cls, first, second):
        intersection = cls.intersection_area(first, second)
        union = first[2] * first[3] + second[2] * second[3] - intersection
        return union > 0 and intersection / union >= 0.90

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
