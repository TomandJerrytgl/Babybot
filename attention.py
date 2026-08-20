"""Explainable single-frame visual attention for Babybot."""

from __future__ import annotations

import time
import heapq
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
    hsv_histogram: np.ndarray
    gray_structure: np.ndarray
    edge_structure: np.ndarray


@dataclass(frozen=True)
class AttentionReference:
    observation_id: int
    timestamp: float
    window: Window
    signature: VisualSignature
    attention_score: float = 0.0


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
        offsets=(-0.5, -0.25, 0.0, 0.25, 0.5),
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
        coarse_candidates = []

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
                    coarse = quick_signature_similarity(
                        self.reference.signature, image, window
                    )
                    coarse_candidates.append((coarse, window))

        best_window = None
        best_similarity = -1.0
        for _coarse, window in sorted(coarse_candidates, reverse=True)[:3]:
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
    histogram = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1)
    gray_structure = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    edge_structure = cv2.resize(edges, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return VisualSignature(
        brightness=brightness,
        contrast=contrast,
        color=color,
        edge_density=edge_density,
        visual_score=visual_score,
        aspect_ratio=float(width / max(1, height)),
        area_fraction=float(width * height / max(1, image_area)),
        hsv_histogram=histogram.flatten(),
        gray_structure=gray_structure,
        edge_structure=edge_structure,
    )


def signature_similarity(reference: VisualSignature, candidate: VisualSignature) -> float:
    def closeness(first, second, scale):
        return float(np.clip(1.0 - abs(first - second) / scale, 0.0, 1.0))

    histogram_distance = cv2.compareHist(
        reference.hsv_histogram.astype(np.float32),
        candidate.hsv_histogram.astype(np.float32),
        cv2.HISTCMP_BHATTACHARYYA,
    )
    histogram_similarity = float(np.clip(1.0 - histogram_distance, 0.0, 1.0))
    gray_difference = float(np.mean(np.abs(reference.gray_structure - candidate.gray_structure)))
    gray_similarity = float(np.clip(1.0 - gray_difference / 0.35, 0.0, 1.0))
    edge_difference = float(np.mean(np.abs(reference.edge_structure - candidate.edge_structure)))
    edge_similarity = float(np.clip(1.0 - edge_difference / 0.35, 0.0, 1.0))
    values = (
        (0.35, histogram_similarity),
        (0.18, gray_similarity),
        (0.12, edge_similarity),
        (0.07, closeness(reference.brightness, candidate.brightness, 0.30)),
        (0.08, closeness(reference.contrast, candidate.contrast, 0.30)),
        (0.07, closeness(reference.color, candidate.color, 0.35)),
        (0.07, closeness(reference.edge_density, candidate.edge_density, 0.15)),
        (0.04, closeness(reference.aspect_ratio, candidate.aspect_ratio, 0.45)),
        (0.02, closeness(reference.area_fraction, candidate.area_fraction, 0.06)),
    )
    return float(np.clip(sum(weight * score for weight, score in values), 0.0, 1.0))


def quick_signature_similarity(reference, image, window):
    x, y, width, height = TemplateTracker.clip_window(window, image.shape)
    patch = image[y:y + height, x:x + width]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(hsv[:, :, 2]) / 255.0)
    color = float(np.mean(hsv[:, :, 1]) / 255.0)
    contrast = float(np.clip(np.std(gray) / 128.0, 0.0, 1.0))
    score = (
        0.35 * max(0.0, 1.0 - abs(reference.brightness - brightness) / 0.30)
        + 0.35 * max(0.0, 1.0 - abs(reference.color - color) / 0.35)
        + 0.30 * max(0.0, 1.0 - abs(reference.contrast - contrast) / 0.30)
    )
    return float(np.clip(score, 0.0, 1.0))


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
            attention_score=float(candidate["score"]),
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
    """Single-frame visual attention using coarse-to-fine adaptive windows."""

    def __init__(
        self,
        observation,
        object=None,
        eye="left",
        previous_focus: Optional[Window] = None,
        previous_focus_age=0,
        verbose=True,
        analysis_width=320,
        hold_observations=10,
        objectness_threshold=0.45,
        maximum_candidates=10,
        partial_overlap_iou=0.30,
        proposal_windows=None,
    ):
        del object, previous_focus, previous_focus_age, hold_observations
        if eye not in ("left", "right"):
            raise ValueError("eye must be 'left' or 'right'")
        self.observation = observation
        self.eye = eye
        self.verbose = verbose
        self.analysis_width = int(analysis_width)
        self.objectness_threshold = float(objectness_threshold)
        self.maximum_candidates = int(maximum_candidates)
        self.partial_overlap_iou = float(partial_overlap_iou)
        self.proposal_windows = tuple(proposal_windows or ())
        self.candidates = []
        self.elapsed_time = 0.0
        self.focus_source = "none"
        image = getattr(observation, eye)
        self.focus = self.find_attention_window(image)

    def find_attention_window(self, image):
        start = time.perf_counter()
        self._validate_image(image)
        small, scale_x, scale_y = self.analysis_image(image)
        statistics = self.prepare_fixed_window_statistics(small)
        candidates_by_window = {}
        coarse_candidates = []
        for window in self.proposal_windows:
            window = self.clip_window(window, small.shape)
            candidate = self.evaluate_fixed_window(
                small, window, statistics, source="lab-region"
            )
            if candidate is not None:
                candidates_by_window[window] = candidate
                coarse_candidates.append(candidate)
        for window in self.coarse_adaptive_windows(small.shape):
            candidate = self.evaluate_fixed_window(small, window, statistics)
            if candidate is None:
                continue
            candidates_by_window[window] = candidate
            coarse_candidates.append(candidate)

        # Refine the strongest hypotheses in several successively smaller steps.
        # Every side can move independently, so the result is not restricted to
        # one of the coarse sizes or aspect ratios.
        refinement_seeds = sorted(
            coarse_candidates, key=lambda item: item["objectness"], reverse=True
        )[:12]
        for seed in refinement_seeds:
            current = seed
            for adjustment in (12, 6, 3):
                for _ in range(4):
                    improved = current
                    for window in self.refined_windows(
                            current["window"], small.shape, adjustment=adjustment):
                        candidate = candidates_by_window.get(window)
                        if candidate is None:
                            candidate = self.evaluate_fixed_window(
                                small, window, statistics,
                                source=current.get("source", "default"),
                            )
                            if candidate is not None:
                                candidates_by_window[window] = candidate
                        if (candidate is not None
                                and candidate["objectness"] > improved["objectness"]):
                            improved = candidate
                    if improved is current:
                        break
                    current = improved

        candidates = []
        for candidate in candidates_by_window.values():
            if candidate["objectness"] < self.objectness_threshold:
                continue
            window = candidate["window"]
            mapped = self.map_window(window, scale_x, scale_y, image.shape)
            candidate["window"] = mapped
            candidate["area_fraction"] = float(
                mapped[2] * mapped[3] / (image.shape[0] * image.shape[1])
            )
            candidates.append(candidate)

        self.candidates = self.suppress_overlaps(
            candidates,
            overlap_threshold=self.partial_overlap_iou,
            maximum_candidates=self.maximum_candidates,
        )
        for rank, candidate in enumerate(self.candidates, start=1):
            candidate["rank"] = rank
        self.elapsed_time = time.perf_counter() - start
        if self.candidates:
            self.focus_source = self.candidates[0]["source"]
        if self.verbose:
            self.print_results()
        return self.candidates[0]["window"] if self.candidates else None

    @staticmethod
    def coarse_adaptive_windows(
            image_shape,
            scales=(48, 64, 80, 112, 144),
            aspect_ratios=(1.0, 4/3, 3/4, 3/2, 2/3, 2.0, 0.5)):
        """Cover the image with several area scales and aspect ratios."""
        height, width = image_shape[:2]
        windows = []
        seen_shapes = set()
        for scale in scales:
            for ratio in aspect_ratios:
                window_width = max(16, int(round(scale * np.sqrt(ratio) / 8.0)) * 8)
                window_height = max(16, int(round(scale / np.sqrt(ratio) / 8.0)) * 8)
                window_width = min(width, window_width)
                window_height = min(height, window_height)
                shape = (window_width, window_height)
                if shape in seen_shapes:
                    continue
                seen_shapes.add(shape)
                step = max(12, min(window_width, window_height) // 3)
                xs = list(range(0, width - window_width + 1, step))
                ys = list(range(0, height - window_height + 1, step))
                if xs[-1] != width - window_width:
                    xs.append(width - window_width)
                if ys[-1] != height - window_height:
                    ys.append(height - window_height)
                windows.extend(
                    (x, y, window_width, window_height) for y in ys for x in xs
                )
        return windows

    @staticmethod
    def refined_windows(window, image_shape, adjustment=8):
        """Locally tune position, width and height around one coarse box."""
        image_height, image_width = image_shape[:2]
        x, y, width, height = window
        output = set()
        for delta_width in (-adjustment, 0, adjustment):
            for delta_height in (-adjustment, 0, adjustment):
                refined_width = int(np.clip(width + delta_width, 16, image_width))
                refined_height = int(np.clip(height + delta_height, 16, image_height))
                for delta_x in (-adjustment, 0, adjustment):
                    for delta_y in (-adjustment, 0, adjustment):
                        refined_x = int(np.clip(x + delta_x, 0, image_width - refined_width))
                        refined_y = int(np.clip(y + delta_y, 0, image_height - refined_height))
                        output.add((refined_x, refined_y, refined_width, refined_height))
        return output

    @staticmethod
    def prepare_fixed_window_statistics(image):
        image_float = image.astype(np.float32)
        vividness = np.max(image_float, axis=2) - np.min(image_float, axis=2)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        return {
            "vividness": cv2.integral(vividness, sdepth=cv2.CV_64F),
            "brightness": cv2.integral(hsv[:, :, 2], sdepth=cv2.CV_64F),
            "edges": cv2.integral(edges, sdepth=cv2.CV_64F),
            "lab": tuple(
                cv2.integral(lab[:, :, channel], sdepth=cv2.CV_64F)
                for channel in range(3)
            ),
            "lab_squared": tuple(
                cv2.integral(
                    lab[:, :, channel].astype(np.float32) ** 2,
                    sdepth=cv2.CV_64F,
                )
                for channel in range(3)
            ),
        }

    @staticmethod
    def _rectangle_sum(integral, x1, y1, x2, y2):
        return float(
            integral[y2, x2] - integral[y1, x2]
            - integral[y2, x1] + integral[y1, x1]
        )

    def evaluate_fixed_window(self, image, window, statistics=None, source="default"):
        """Score one box by boundary fit, color contrast, vividness and texture."""
        x, y, width, height = self.clip_window(window, image.shape)
        statistics = statistics or self.prepare_fixed_window_statistics(image)

        patch_area = width * height

        vividness = self._rectangle_sum(
            statistics["vividness"], x, y, x + width, y + height
        ) / (patch_area * 255.0)
        brightness = self._rectangle_sum(
            statistics["brightness"], x, y, x + width, y + height
        ) / (patch_area * 255.0)
        brightness_appeal = float(np.exp(
            -((brightness - 0.67) ** 2) / (2 * 0.24 ** 2)
        ))
        edge_density = self._rectangle_sum(
            statistics["edges"], x, y, x + width, y + height
        ) / (patch_area * 255.0)
        edge_score = float(np.clip(edge_density / 0.12, 0.0, 1.0))
        band = max(2, min(6, min(width, height) // 8))
        edge_integral = statistics["edges"]
        boundary_sum = (
            self._rectangle_sum(edge_integral, x, y, x + width, y + band)
            + self._rectangle_sum(edge_integral, x, y + height - band, x + width, y + height)
            + self._rectangle_sum(edge_integral, x, y + band, x + band, y + height - band)
            + self._rectangle_sum(edge_integral, x + width - band, y + band, x + width, y + height - band)
        )
        boundary_area = max(1, 2 * band * (width + height - 2 * band))
        boundary_fit = float(np.clip(
            boundary_sum / (boundary_area * 255.0 * 0.70), 0.0, 1.0
        ))

        foreground_sum = np.array([
            self._rectangle_sum(channel, x, y, x + width, y + height)
            for channel in statistics["lab"]
        ])
        foreground_mean = foreground_sum / patch_area
        foreground_squared_mean = np.array([
            self._rectangle_sum(channel, x, y, x + width, y + height)
            for channel in statistics["lab_squared"]
        ]) / patch_area
        foreground_std = np.sqrt(np.maximum(
            0.0, foreground_squared_mean - foreground_mean ** 2
        ))
        # Coherence is deliberately soft: patterned objects may have substantial
        # internal variation, but a box mixing several unrelated background areas
        # should receive less objectness.
        coherence = float(np.exp(-np.mean(foreground_std) / 55.0))
        contrast_band = max(3, min(12, int(round(min(width, height) * 0.10))))

        def lab_mean(x1, y1, x2, y2):
            area = (x2 - x1) * (y2 - y1)
            if area <= 0:
                return None
            return np.array([
                self._rectangle_sum(channel, x1, y1, x2, y2)
                for channel in statistics["lab"]
            ]) / area

        def side_contrast(inner_coordinates, outer_coordinates):
            inner_mean = lab_mean(*inner_coordinates)
            outer_mean = lab_mean(*outer_coordinates)
            if inner_mean is None or outer_mean is None:
                return None
            return float(np.clip(
                np.linalg.norm(inner_mean - outer_mean) / 85.0, 0.0, 1.0
            ))

        contrast_top = None
        if y > 0:
            contrast_top = side_contrast(
                (x, y, x + width, min(y + contrast_band, y + height)),
                (x, max(0, y - contrast_band), x + width, y),
            )
        contrast_bottom = None
        if y + height < image.shape[0]:
            contrast_bottom = side_contrast(
                (x, max(y, y + height - contrast_band), x + width, y + height),
                (x, y + height, x + width,
                 min(image.shape[0], y + height + contrast_band)),
            )

        # Top and bottom own the corner pixels. Vertical sides exclude those
        # short corner sections so the same boundary is not counted twice.
        corner_cut = min(contrast_band, max(0, height // 3))
        vertical_y1 = y + corner_cut
        vertical_y2 = y + height - corner_cut
        vertical_length = max(0, vertical_y2 - vertical_y1)
        contrast_left = None
        if x > 0 and vertical_length > 0:
            contrast_left = side_contrast(
                (x, vertical_y1, min(x + contrast_band, x + width), vertical_y2),
                (max(0, x - contrast_band), vertical_y1, x, vertical_y2),
            )
        contrast_right = None
        if x + width < image.shape[1] and vertical_length > 0:
            contrast_right = side_contrast(
                (max(x, x + width - contrast_band), vertical_y1,
                 x + width, vertical_y2),
                (x + width, vertical_y1,
                 min(image.shape[1], x + width + contrast_band), vertical_y2),
            )

        weighted_contrasts = [
            (contrast_top, width),
            (contrast_bottom, width),
            (contrast_left, vertical_length),
            (contrast_right, vertical_length),
        ]
        valid_contrasts = [
            (value, length) for value, length in weighted_contrasts
            if value is not None and length > 0
        ]
        if not valid_contrasts:
            return None
        surround_contrast = float(
            sum(value * length for value, length in valid_contrasts)
            / sum(length for _value, length in valid_contrasts)
        )
        center = self.center_preference(window, image.shape)
        # A bright achromatic patch can otherwise win solely through its Lab
        # lightness contrast. Suppress that specific white-background bias while
        # leaving dark achromatic objects and vivid bright objects unaffected.
        white_bias = (
            (1.0 - vividness)
            * float(np.clip((brightness - 0.80) / 0.20, 0.0, 1.0))
        )
        objectness = float(np.clip(
            0.34 * boundary_fit
            + 0.30 * surround_contrast
            + 0.16 * coherence
            + 0.12 * edge_score
            + 0.08 * vividness
            - 0.12 * white_bias,
            0.0, 1.0,
        ))
        ranking_score = float(np.clip(
            0.95 * objectness + 0.05 * center, 0.0, 1.0
        ))
        if objectness < 0.18:
            return None
        appearance = 0.60 * vividness + 0.40 * brightness_appeal
        return {
            "window": (x, y, width, height),
            "level": 0,
            "brightness": float(np.clip(brightness, 0.0, 1.0)),
            "contrast": surround_contrast,
            "contrast_top": contrast_top,
            "contrast_bottom": contrast_bottom,
            "contrast_left": contrast_left,
            "contrast_right": contrast_right,
            "color": float(np.clip(vividness, 0.0, 1.0)),
            "edge": edge_score,
            "boundary": boundary_fit,
            "white_bias": float(white_bias),
            "visual": ranking_score,
            "center": center,
            "objectness": objectness,
            "coherence": coherence,
            "appearance": float(np.clip(appearance, 0.0, 1.0)),
            "source": source,
            "ranking_score": ranking_score,
            "score": ranking_score,
        }

    def color_regions(self, image):
        """Create regions from continuous color/texture statistics at several scales."""
        deadline = time.perf_counter() + 0.35
        smoothed = cv2.GaussianBlur(image, (5, 5), 0)
        lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB).astype(np.float32)
        gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160) > 0
        image_area = image.shape[0] * image.shape[1]
        minimum_area = max(40, int(image_area * 0.003))
        maximum_area = int(image_area * 0.35)
        regions = []
        # Start at the middle scale so a useful object-sized representation is
        # always available even when the processing budget is exhausted.
        for cell_size in (16, 32, 8):
            if time.perf_counter() >= deadline:
                break
            regions.extend(self._statistical_scale_regions(
                lab, edges, cell_size, minimum_area, maximum_area
            ))

        # Prefer regions with a useful amount of image support. Duplicate scales
        # are intentionally retained here; the existing overlap suppression ranks
        # them after objectness has been measured against their surroundings.
        regions.sort(key=lambda item: int(np.sum(item[1])), reverse=True)
        return regions[:80]

    def _statistical_scale_regions(
            self, lab, edges, cell_size, minimum_area, maximum_area):
        """Join neighboring cells whose color or texture statistics agree."""
        height, width = lab.shape[:2]
        rows = (height + cell_size - 1) // cell_size
        columns = (width + cell_size - 1) // cell_size
        cells = []
        for row in range(rows):
            y1, y2 = row * cell_size, min(height, (row + 1) * cell_size)
            for column in range(columns):
                x1, x2 = column * cell_size, min(width, (column + 1) * cell_size)
                pixels = lab[y1:y2, x1:x2].reshape(-1, 3)
                cells.append({
                    "window": (x1, y1, x2 - x1, y2 - y1),
                    "mean": np.mean(pixels, axis=0),
                    "std": np.std(pixels, axis=0),
                    "edge_density": float(np.mean(edges[y1:y2, x1:x2])),
                    "area": int((x2 - x1) * (y2 - y1)),
                })

        parent = list(range(len(cells)))
        aggregate = [{
            "area": cell["area"],
            "sum": cell["mean"] * cell["area"],
            "sum_std": cell["std"] * cell["area"],
            "edge_sum": cell["edge_density"] * cell["area"],
        } for cell in cells]

        def find(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(first, second):
            first, second = find(first), find(second)
            if first == second:
                return
            if aggregate[first]["area"] < aggregate[second]["area"]:
                first, second = second, first
            if aggregate[first]["area"] + aggregate[second]["area"] > maximum_area:
                return
            parent[second] = first
            for key in ("area", "sum", "sum_std", "edge_sum"):
                aggregate[first][key] += aggregate[second][key]

        links = []
        for row in range(rows):
            for column in range(columns):
                first = row * columns + column
                if column + 1 < columns:
                    second = first + 1
                    cost = self._cell_similarity_cost(cells[first], cells[second])
                    if cost is not None:
                        links.append((cost, first, second))
                if row + 1 < rows:
                    second = first + columns
                    cost = self._cell_similarity_cost(cells[first], cells[second])
                    if cost is not None:
                        links.append((cost, first, second))

        # Strong local agreements are joined first. Before every join, compare
        # each new cell with the accumulated region to prevent color-chain leaks.
        for local_cost, first, second in sorted(links):
            first_root, second_root = find(first), find(second)
            if first_root == second_root:
                continue
            first_region = self._aggregate_descriptor(aggregate[first_root])
            second_region = self._aggregate_descriptor(aggregate[second_root])
            whole_region_match = self._cell_similarity_cost(
                first_region, second_region
            ) is not None
            # A shallow local difference repeated across cells is normally an
            # illumination gradient. Permit it even when the two accumulated
            # means have drifted apart; larger stepwise changes still require
            # whole-region agreement and therefore cannot form a color chain.
            smooth_gradient = (
                local_cost <= 0.55
                and float(np.mean(first_region["std"])) <= 12.0
                and float(np.mean(second_region["std"])) <= 12.0
            )
            if whole_region_match or smooth_gradient:
                union(first_root, second_root)

        groups = {}
        for index, cell in enumerate(cells):
            groups.setdefault(find(index), []).append(cell["window"])

        output = []
        for windows in groups.values():
            area = sum(item[2] * item[3] for item in windows)
            if area < minimum_area or area > maximum_area:
                continue
            mask = np.zeros((height, width), dtype=np.uint8)
            for x, y, cell_width, cell_height in windows:
                mask[y:y + cell_height, x:x + cell_width] = 255
            ys, xs = np.where(mask > 0)
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            region_width, region_height = x2 - x1, y2 - y1
            if region_width < 6 or region_height < 6:
                continue
            fill_ratio = area / max(1, region_width * region_height)
            if fill_ratio < 0.45:
                continue
            touches = sum((x1 == 0, y1 == 0, x2 >= width, y2 >= height))
            if touches >= 3:
                continue
            output.append(((x1, y1, region_width, region_height), mask[y1:y2, x1:x2] > 0))
        return output

    @staticmethod
    def _aggregate_descriptor(aggregate):
        area = max(1, aggregate["area"])
        return {
            "mean": aggregate["sum"] / area,
            "std": aggregate["sum_std"] / area,
            "edge_density": aggregate["edge_sum"] / area,
            "area": area,
        }

    @staticmethod
    def _cell_similarity_cost(first, second):
        mean_delta = first["mean"] - second["mean"]
        perceptual_distance = float(np.linalg.norm(
            np.array((0.55 * mean_delta[0], mean_delta[1], mean_delta[2]))
        ))
        std_distance = float(np.mean(np.abs(first["std"] - second["std"])))
        edge_distance = abs(first["edge_density"] - second["edge_density"])
        texture_level = float(np.mean(first["std"] + second["std"]) * 0.5)

        smooth_match = perceptual_distance <= 18.0
        textured_match = (
            texture_level >= 8.0
            and perceptual_distance <= 38.0
            and std_distance <= 11.0
            and edge_distance <= 0.12
        )
        if not smooth_match and not textured_match:
            return None
        return (
            perceptual_distance / (38.0 if textured_match else 18.0)
            + 0.35 * std_distance / 11.0
            + 0.25 * edge_distance / 0.12
        )

    def merge_adjacent_regions(
            self, lab, masks, maximum_area, time_budget=0.30, maximum_merges=40):
        """Merge a sparse region-adjacency graph with incremental statistics."""
        if len(masks) < 2:
            return [mask.copy() for mask in masks]
        started = time.perf_counter()
        owner = np.full(lab.shape[:2], -1, dtype=np.int16)
        for region_id, mask in enumerate(masks):
            owner[(mask > 0) & (owner < 0)] = region_id

        lab_float = lab.astype(np.float64)
        nodes = {}
        for region_id in range(len(masks)):
            region_mask = owner == region_id
            pixels = lab_float[region_mask]
            if pixels.size == 0:
                continue
            nodes[region_id] = {
                "mask": np.where(region_mask, 255, 0).astype(np.uint8),
                "area": int(pixels.shape[0]),
                "sum": np.sum(pixels, axis=0),
                "sum_sq": np.sum(pixels * pixels, axis=0),
                "neighbors": {},
                "version": 0,
            }

        edges = cv2.Canny(lab[:, :, 0], 50, 140) > 0
        self._build_region_adjacency(owner, edges, nodes)
        heap = []
        for first, node in nodes.items():
            for second in node["neighbors"]:
                if first < second:
                    self._push_merge_candidate(heap, nodes, first, second, maximum_area)

        merges = 0
        while heap and merges < maximum_merges:
            if time.perf_counter() - started >= time_budget:
                break
            _cost, first, second, first_version, second_version = heapq.heappop(heap)
            if first not in nodes or second not in nodes:
                continue
            if nodes[first]["version"] != first_version or nodes[second]["version"] != second_version:
                continue
            if second not in nodes[first]["neighbors"]:
                continue
            cost = self._merge_cost(nodes, first, second, maximum_area)
            if cost is None:
                continue
            self._merge_region_nodes(nodes, first, second)
            merges += 1
            for neighbor in list(nodes[first]["neighbors"]):
                self._push_merge_candidate(heap, nodes, first, neighbor, maximum_area)

        return [node["mask"] for node in nodes.values()]

    @staticmethod
    def _build_region_adjacency(owner, edges, nodes):
        edge_stats = {}
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            y1 = slice(max(0, dy), owner.shape[0] + min(0, dy))
            x1 = slice(max(0, dx), owner.shape[1] + min(0, dx))
            y2 = slice(max(0, -dy), owner.shape[0] + min(0, -dy))
            x2 = slice(max(0, -dx), owner.shape[1] + min(0, -dx))
            first_labels = owner[y1, x1]
            second_labels = owner[y2, x2]
            valid = (
                (first_labels >= 0) & (second_labels >= 0)
                & (first_labels != second_labels)
            )
            ys, xs = np.where(valid)
            for y, x in zip(ys, xs):
                first = int(first_labels[y, x])
                second = int(second_labels[y, x])
                if first not in nodes or second not in nodes:
                    continue
                pair = (min(first, second), max(first, second))
                strength = float(edges[y1, x1][y, x] or edges[y2, x2][y, x])
                total, count = edge_stats.get(pair, (0.0, 0))
                edge_stats[pair] = (total + strength, count + 1)
        for (first, second), statistics in edge_stats.items():
            nodes[first]["neighbors"][second] = statistics
            nodes[second]["neighbors"][first] = statistics

    def _push_merge_candidate(self, heap, nodes, first, second, maximum_area):
        if first not in nodes or second not in nodes:
            return
        cost = self._merge_cost(nodes, first, second, maximum_area)
        if cost is not None:
            heapq.heappush(heap, (
                cost, first, second,
                nodes[first]["version"], nodes[second]["version"],
            ))

    @staticmethod
    def _node_mean(node):
        return node["sum"] / node["area"]

    def _merge_cost(self, nodes, first, second, maximum_area):
        first_node, second_node = nodes[first], nodes[second]
        combined_area = first_node["area"] + second_node["area"]
        if combined_area > maximum_area:
            return None
        first_mean = self._node_mean(first_node)
        second_mean = self._node_mean(second_node)
        lightness_distance = abs(float(first_mean[0] - second_mean[0]))
        chroma_distance = float(np.linalg.norm(first_mean[1:] - second_mean[1:]))
        boundary_sum, boundary_count = first_node["neighbors"][second]
        boundary_strength = boundary_sum / max(1, boundary_count)
        if chroma_distance > 24.0 or lightness_distance > 58.0 or boundary_strength > 0.38:
            return None
        combined_sum = first_node["sum"] + second_node["sum"]
        combined_sum_sq = first_node["sum_sq"] + second_node["sum_sq"]
        mean = combined_sum / combined_area
        variance = np.maximum(0.0, combined_sum_sq / combined_area - mean * mean)
        combined_spread = float(np.mean(np.sqrt(variance)))
        if combined_spread > 48.0:
            return None
        smaller = min(first_node["area"], second_node["area"])
        larger = max(first_node["area"], second_node["area"])
        if larger / max(1, smaller) > 12.0 and (
                chroma_distance > 12.0 or boundary_strength > 0.15):
            return None
        return (
            chroma_distance / 24.0
            + 0.55 * lightness_distance / 58.0
            + 0.75 * boundary_strength / 0.38
            + 0.35 * combined_spread / 48.0
        )

    @staticmethod
    def _merge_region_nodes(nodes, first, second):
        first_node, second_node = nodes[first], nodes[second]
        first_node["mask"] = cv2.bitwise_or(first_node["mask"], second_node["mask"])
        first_node["area"] += second_node["area"]
        first_node["sum"] += second_node["sum"]
        first_node["sum_sq"] += second_node["sum_sq"]
        first_node["version"] += 1
        first_node["neighbors"].pop(second, None)
        for neighbor, statistics in list(second_node["neighbors"].items()):
            if neighbor == first or neighbor not in nodes:
                continue
            existing = first_node["neighbors"].get(neighbor, (0.0, 0))
            combined = (existing[0] + statistics[0], existing[1] + statistics[1])
            first_node["neighbors"][neighbor] = combined
            nodes[neighbor]["neighbors"].pop(second, None)
            nodes[neighbor]["neighbors"][first] = combined
        del nodes[second]

    def evaluate_region(self, image, window, region_mask):
        x, y, width, height = self.clip_window(window, image.shape)
        patch = image[y:y + height, x:x + width]
        if patch.size == 0 or not np.any(region_mask):
            return None
        margin = max(4, int(round(max(width, height) * 0.20)))
        cx1, cy1 = max(0, x - margin), max(0, y - margin)
        cx2 = min(image.shape[1], x + width + margin)
        cy2 = min(image.shape[0], y + height + margin)
        context = image[cy1:cy2, cx1:cx2]
        object_mask = np.zeros(context.shape[:2], dtype=np.uint8)
        object_mask[y - cy1:y - cy1 + height, x - cx1:x - cx1 + width][region_mask] = 255
        ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        ring = cv2.dilate(object_mask, ring_kernel, iterations=1)
        ring = (ring > 0) & (object_mask == 0)
        if np.sum(ring) < 20:
            return None

        context_lab = cv2.cvtColor(context, cv2.COLOR_BGR2LAB).astype(np.float32)
        foreground = context_lab[object_mask > 0]
        background = context_lab[ring]
        color_separation = float(np.clip(
            np.linalg.norm(np.mean(foreground, axis=0) - np.mean(background, axis=0)) / 85.0,
            0.0, 1.0,
        ))
        brightness_difference = float(np.clip(
            abs(float(np.mean(foreground[:, 0])) - float(np.mean(background[:, 0]))) / 110.0,
            0.0, 1.0,
        ))
        gray = cv2.cvtColor(context, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        boundary = cv2.morphologyEx(object_mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        boundary_score = float(np.clip(np.mean(edges[boundary] > 0) / 0.22, 0.0, 1.0))
        foreground_std = float(np.mean(np.std(foreground, axis=0)))
        cohesion = float(np.clip(1.0 - foreground_std / 65.0, 0.0, 1.0))
        objectness = (
            0.48 * color_separation + 0.17 * brightness_difference
            + 0.30 * boundary_score + 0.05 * cohesion
        )
        if objectness < 0.28:
            return None

        patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        foreground_hsv = patch_hsv[region_mask]
        saturation = float(np.mean(foreground_hsv[:, 1]) / 255.0)
        brightness = float(np.mean(foreground_hsv[:, 2]) / 255.0)
        color_appeal = float(np.clip(saturation / 0.65, 0.0, 1.0))
        brightness_appeal = float(np.exp(-((brightness - 0.67) ** 2) / (2 * 0.24 ** 2)))
        appearance = 0.65 * color_appeal + 0.35 * brightness_appeal
        center = self.center_preference((x, y, width, height), image.shape)
        score = float(np.clip(0.72 * objectness + 0.18 * appearance + 0.10 * center, 0.0, 1.0))
        if score < 0.40:
            return None
        return self.make_candidate(
            image, (x, y, width, height), score, source="default",
            objectness=objectness, appearance=appearance,
        )

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

    def fallback_windows(self, image_shape):
        """Small fixed fallback set; never perform an exhaustive grid scan."""
        height, width = image_shape[:2]
        windows = []
        for ratio in (1.0, 4/3, 3/4):
            base = min(height, width) * 0.25
            window_width = int(round(base * np.sqrt(ratio)))
            window_height = int(round(base / np.sqrt(ratio)))
            for center_x, center_y in (
                (width * 0.5, height * 0.5),
                (width * 0.25, height * 0.5),
                (width * 0.75, height * 0.5),
            ):
                x = int(np.clip(center_x - window_width / 2, 0, width - window_width))
                y = int(np.clip(center_y - window_height / 2, 0, height - window_height))
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
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        join_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        raw_mask = cv2.dilate(raw_mask, join_kernel, iterations=1)
        changed_fraction = float(np.mean(raw_mask > 0))
        if changed_fraction > 0.55:
            return empty, np.zeros_like(empty, dtype=np.float32), []

        image_area = raw_mask.size
        accepted = np.zeros_like(raw_mask)
        windows = []
        contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
        for contour in contours:
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            area_fraction = hull_area / image_area
            if area_fraction < 0.005:
                continue
            cv2.drawContours(accepted, [hull], -1, 255, thickness=-1)
            x, y, width, height = cv2.boundingRect(hull)
            windows.append((int(x), int(y), int(width), int(height)))
        strength = difference.astype(np.float32) / 255.0
        return accepted, strength, windows

    def static_contour_windows(self, image):
        """Generate object-shaped boxes from color boundaries and closed edges."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_lab = cv2.GaussianBlur(lab, (21, 21), 0)
        color_difference = np.linalg.norm(
            lab.astype(np.float32) - blurred_lab.astype(np.float32), axis=2
        )
        color_edges = np.where(color_difference >= 12.0, 255, 0).astype(np.uint8)
        canny = cv2.Canny(gray, 50, 140)
        boundary = cv2.bitwise_or(color_edges, canny)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        boundary = cv2.morphologyEx(boundary, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        contours, _ = cv2.findContours(boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
        image_area = image.shape[0] * image.shape[1]
        windows = []
        for contour in contours:
            hull = cv2.convexHull(contour)
            area_fraction = cv2.contourArea(hull) / image_area
            if area_fraction < 0.005 or area_fraction > 0.40:
                continue
            x, y, width, height = cv2.boundingRect(hull)
            if width < 8 or height < 8:
                continue
            windows.append((int(x), int(y), int(width), int(height)))
        return windows

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
        region_motion = motion_mask[y:y + height, x:x + width] > 0
        use_foreground = float(np.mean(region_motion)) >= 0.05
        foreground_lab = patch_lab[region_motion] if use_foreground else patch_lab.reshape(-1, 3)
        color_distance = np.linalg.norm(
            np.mean(foreground_lab, axis=0)
            - np.mean(context_lab[background_mask], axis=0)
        )
        color_separation = float(np.clip(color_distance / 90.0, 0.0, 1.0))

        patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        context_gray = cv2.cvtColor(context, cv2.COLOR_BGR2GRAY)
        foreground_gray = patch_gray[region_motion] if use_foreground else patch_gray.reshape(-1)
        brightness_difference = float(np.clip(
            abs(float(np.mean(foreground_gray)) - float(np.mean(context_gray[background_mask]))) / 128.0,
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
        internal_consistency = float(np.clip(1.0 - np.std(foreground_gray) / 90.0, 0.0, 1.0))
        objectness = (
            0.45 * color_separation + 0.30 * edge_continuity
            + 0.15 * brightness_difference + 0.10 * internal_consistency
        )
        if objectness < self.objectness_threshold:
            return None

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        foreground_hsv = hsv[region_motion] if use_foreground else hsv.reshape(-1, 3)
        saturation = float(np.mean(foreground_hsv[:, 1]) / 255.0)
        color_appeal = float(np.clip(saturation / 0.65, 0.0, 1.0))
        brightness = float(np.mean(foreground_hsv[:, 2]) / 255.0)
        brightness_appeal = float(np.exp(-((brightness - 0.67) ** 2) / (2 * 0.24 ** 2)))
        appearance = 0.65 * color_appeal + 0.35 * brightness_appeal
        base_score = 0.65 * objectness + 0.35 * appearance

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
            "objectness": float(np.clip(objectness, 0.0, 1.0)),
            "appearance": float(np.clip(appearance, 0.0, 1.0)),
            "source": source,
            "score": float(np.clip(score, 0.0, 1.0)),
        }

    def suppress_overlaps(
            self, candidates, overlap_threshold=0.30,
            maximum_candidates=10, maximum_partial_group=2):
        """Keep all containment alternatives and the best two per partial-overlap group."""
        ordered = sorted(
            candidates,
            key=lambda item: (
                item["score"], item["window"][2] * item["window"][3]
            ),
            reverse=True,
        )
        adjacency = [set() for _ in ordered]
        for first_index, first in enumerate(ordered):
            for second_index in range(first_index + 1, len(ordered)):
                second = ordered[second_index]
                first_window, second_window = first["window"], second["window"]
                if (self.contains(first_window, second_window)
                        or self.contains(second_window, first_window)):
                    continue
                if self.intersection_over_union(first_window, second_window) >= overlap_threshold:
                    adjacency[first_index].add(second_index)
                    adjacency[second_index].add(first_index)

        kept_indices = set()
        unseen = set(range(len(ordered)))
        while unseen:
            root = unseen.pop()
            component = {root}
            pending = [root]
            while pending:
                current = pending.pop()
                neighbors = adjacency[current] & unseen
                unseen.difference_update(neighbors)
                component.update(neighbors)
                pending.extend(neighbors)
            ranked_component = sorted(
                component,
                key=lambda index: ordered[index]["score"],
                reverse=True,
            )
            kept_indices.update(ranked_component[:maximum_partial_group])

        selected = [ordered[index] for index in kept_indices]
        selected.sort(
            key=lambda item: (
                item["score"], item["window"][2] * item["window"][3]
            ),
            reverse=True,
        )
        return selected[:maximum_candidates]

    @classmethod
    def candidates_compatible(cls, first, second):
        first_window, second_window = first["window"], second["window"]
        if cls.intersection_area(first_window, second_window) == 0:
            return True
        if cls.contains(first_window, second_window):
            outer, inner = first, second
        elif cls.contains(second_window, first_window):
            outer, inner = second, first
        else:
            return False
        return inner["score"] > outer["score"]

    @classmethod
    def windows_compatible(cls, first, second, overlap_threshold=0.65):
        del overlap_threshold
        return cls.intersection_area(first, second) == 0

    @staticmethod
    def intersection_area(first, second):
        ax, ay, aw, ah = first
        bx, by, bw, bh = second
        return max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
            0, min(ay + ah, by + bh) - max(ay, by)
        )

    @classmethod
    def intersection_over_union(cls, first, second):
        intersection = cls.intersection_area(first, second)
        if intersection <= 0:
            return 0.0
        first_area = first[2] * first[3]
        second_area = second[2] * second[3]
        return float(intersection / max(1, first_area + second_area - intersection))

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
