"""Experimental Lab-region proposals and uncalibrated stereo evidence."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class RegionProposalConfig:
    minimum_region_fraction: float = 0.03
    growth_threshold: float = 14.0
    edge_stop_threshold: float = 110.0
    minimum_merge_score: float = 0.65
    stereo_merge_weight: float = 0.12
    vertical_search_tolerance: int = 12
    interface_band: int = 5


class _DisjointSet:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first, second):
        first, second = self.find(first), self.find(second)
        if first == second:
            return first
        self.parent[second] = first
        return first


class StereoRegionProposer:
    """Create object-size suggestions before the regular attention scan."""

    def __init__(self, config=None):
        self.config = config or RegionProposalConfig()

    @staticmethod
    def _lab_distance(first, second):
        delta = np.abs(np.asarray(first, np.float32) - np.asarray(second, np.float32))
        return float(0.35 * delta[0] + 0.65 * math.hypot(delta[1], delta[2]))

    def grow_regions(self, image):
        """Grow four-connected regions against each region's evolving Lab mean."""
        lab = cv2.cvtColor(cv2.GaussianBlur(image, (3, 3), 0), cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        height, width = lab.shape[:2]
        labels = np.full((height, width), -1, np.int32)
        regions = []
        for seed_y in range(height):
            for seed_x in range(width):
                if labels[seed_y, seed_x] >= 0:
                    continue
                identifier = len(regions)
                queue = deque([(seed_x, seed_y)])
                labels[seed_y, seed_x] = identifier
                total = lab[seed_y, seed_x].astype(np.float64)
                squared = total * total
                count = 1
                x1 = x2 = seed_x
                y1 = y2 = seed_y
                while queue:
                    x, y = queue.popleft()
                    mean = total / count
                    for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                        if nx < 0 or ny < 0 or nx >= width or ny >= height:
                            continue
                        if labels[ny, nx] >= 0:
                            continue
                        value = lab[ny, nx]
                        threshold = self.config.growth_threshold
                        if max(edges[y, x], edges[ny, nx]) >= self.config.edge_stop_threshold:
                            threshold *= .60
                        if self._lab_distance(value, mean) > threshold:
                            continue
                        labels[ny, nx] = identifier
                        queue.append((nx, ny))
                        value_float = value.astype(np.float64)
                        total += value_float
                        squared += value_float * value_float
                        count += 1
                        x1, x2 = min(x1, nx), max(x2, nx)
                        y1, y2 = min(y1, ny), max(y2, ny)
                variance = np.maximum(0.0, squared / count - (total / count) ** 2)
                regions.append({
                    "id": identifier,
                    "area": count,
                    "lab_sum": total,
                    "lab_squared_sum": squared,
                    "lab_mean": total / count,
                    "lab_std": np.sqrt(variance),
                    "bbox": (x1, y1, x2 - x1 + 1, y2 - y1 + 1),
                })
        return labels, regions, edges

    @staticmethod
    def adjacency(labels, edge_image):
        """Measure shared boundaries between initial labels."""
        graph = {}
        for axis, first, second in (
            ("vertical", labels[:, :-1], labels[:, 1:]),
            ("horizontal", labels[:-1, :], labels[1:, :]),
        ):
            different = first != second
            ys, xs = np.nonzero(different)
            for y, x in zip(ys.tolist(), xs.tolist()):
                a, b = int(first[y, x]), int(second[y, x])
                if a > b:
                    a, b = b, a
                item = graph.setdefault((a, b), {
                    "shared": 0, "horizontal": 0, "vertical": 0,
                    "edge_sum": 0.0,
                })
                item["shared"] += 1
                item[axis] += 1
                item["edge_sum"] += float(edge_image[y, x]) / 255.0
        return graph

    @staticmethod
    def _aggregate(regions, members):
        chosen = [regions[index] for index in members]
        area = sum(item["area"] for item in chosen)
        lab_sum = sum((item["lab_sum"] for item in chosen), np.zeros(3))
        squared = sum((item["lab_squared_sum"] for item in chosen), np.zeros(3))
        x1 = min(item["bbox"][0] for item in chosen)
        y1 = min(item["bbox"][1] for item in chosen)
        x2 = max(item["bbox"][0] + item["bbox"][2] for item in chosen)
        y2 = max(item["bbox"][1] + item["bbox"][3] for item in chosen)
        mean = lab_sum / area
        std = np.sqrt(np.maximum(0.0, squared / area - mean * mean))
        return {"area": area, "lab_mean": mean, "lab_std": std,
                "bbox": (x1, y1, x2 - x1, y2 - y1)}

    def merge_features(self, first, second, boundary, image_area, depth_similarity=None):
        color_similarity = math.exp(-self._lab_distance(
            first["lab_mean"], second["lab_mean"]
        ) / 18.0)
        shared = boundary["shared"]
        edge_density = boundary["edge_sum"] / max(1, shared)
        weak_boundary = float(np.clip(1.0 - edge_density, 0.0, 1.0))
        horizontal_interface = boundary["horizontal"] >= boundary["vertical"]
        if horizontal_interface:
            first_span, second_span = first["bbox"][2], second["bbox"][2]
        else:
            first_span, second_span = first["bbox"][3], second["bbox"][3]
        span_similarity = min(first_span, second_span) / max(1, max(first_span, second_span))
        interface_coverage = min(1.0, shared / max(1, min(first_span, second_span)))
        area_similarity = min(first["area"], second["area"]) / max(
            first["area"], second["area"]
        )
        scale_compatibility = (
            .50 * span_similarity + .30 * interface_coverage + .20 * area_similarity
        )
        x1 = min(first["bbox"][0], second["bbox"][0])
        y1 = min(first["bbox"][1], second["bbox"][1])
        x2 = max(first["bbox"][0] + first["bbox"][2],
                 second["bbox"][0] + second["bbox"][2])
        y2 = max(first["bbox"][1] + first["bbox"][3],
                 second["bbox"][1] + second["bbox"][3])
        merged_fill = min(1.0, (first["area"] + second["area"]) / max(1, (x2-x1)*(y2-y1)))
        shape_continuity = .65 * span_similarity + .35 * merged_fill
        threshold_area = image_area * self.config.minimum_region_fraction
        small_preference = 1.0 if first["area"] < threshold_area and second["area"] < threshold_area else (
            .5 if min(first["area"], second["area"]) < threshold_area else 0.0
        )
        visual_score = (
            .35 * color_similarity + .20 * weak_boundary
            + .15 * scale_compatibility + .15 * merged_fill
            + .10 * shape_continuity + .05 * small_preference
        )
        if depth_similarity is None:
            score = visual_score
        else:
            depth_weight = self.config.stereo_merge_weight
            score = (1.0 - depth_weight) * visual_score + depth_weight * depth_similarity
        return {
            "color_similarity": color_similarity,
            "weak_boundary": weak_boundary,
            "scale_compatibility": scale_compatibility,
            "merged_fill": merged_fill,
            "shape_continuity": shape_continuity,
            "small_region_preference": small_preference,
            "depth_similarity": depth_similarity,
            "merge_score": score,
        }

    def merge_pass(self, regions, graph, image_shape, depth_by_pair=None,
                   stage="visual"):
        image_area = image_shape[0] * image_shape[1]
        sets = _DisjointSet(len(regions))
        members = {index: {index} for index in range(len(regions))}
        diagnostics = []
        # Strongest interfaces are considered first; region statistics are
        # recalculated after every accepted union.
        edges = sorted(graph.items(), key=lambda item: item[1]["shared"], reverse=True)
        changed = True
        while changed:
            changed = False
            for (raw_a, raw_b), boundary in edges:
                a, b = sets.find(raw_a), sets.find(raw_b)
                if a == b:
                    continue
                first = self._aggregate(regions, members[a])
                second = self._aggregate(regions, members[b])
                threshold_area = image_area * self.config.minimum_region_fraction
                first_variation = .35 * first["lab_std"][0] + .65 * math.hypot(
                    first["lab_std"][1], first["lab_std"][2]
                )
                second_variation = .35 * second["lab_std"][0] + .65 * math.hypot(
                    second["lab_std"][1], second["lab_std"][2]
                )
                first_protected = first["area"] >= threshold_area and first_variation <= 12.0
                second_protected = second["area"] >= threshold_area and second_variation <= 12.0
                depth = None if not depth_by_pair else depth_by_pair.get(
                    tuple(sorted((raw_a, raw_b)))
                )
                scores = self.merge_features(first, second, boundary, image_area, depth)
                accepted = scores["merge_score"] >= self.config.minimum_merge_score
                if first_protected or second_protected:
                    accepted = accepted and scores["merge_score"] >= .78
                diagnostics.append({
                    "stage": stage,
                    "order": len(diagnostics) + 1,
                    "regions": [int(raw_a), int(raw_b)], "accepted": bool(accepted),
                    "component_a": sorted(int(value) for value in members[a]),
                    "component_b": sorted(int(value) for value in members[b]),
                    "protected": [bool(first_protected), bool(second_protected)],
                    **{key: None if value is None else round(float(value), 4)
                       for key, value in scores.items()},
                })
                if not accepted:
                    continue
                root = sets.union(a, b)
                other = b if root == a else a
                members[root] = members[a] | members[b]
                members.pop(other, None)
                changed = True
        groups = {}
        for index in range(len(regions)):
            groups.setdefault(sets.find(index), set()).add(index)
        return groups, diagnostics

    @staticmethod
    def approximate_disparity(left, right):
        """Return uncalibrated left-view disparity; invalid values are NaN."""
        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        matcher = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=128, blockSize=7,
            P1=8 * 7 * 7, P2=32 * 7 * 7,
            uniquenessRatio=8, speckleWindowSize=40, speckleRange=2,
            disp12MaxDiff=2,
        )
        disparity = matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
        disparity[disparity <= 0.0] = np.nan
        return disparity

    @staticmethod
    def region_disparities(labels, groups, disparity):
        output = {}
        for root, members in groups.items():
            mask = np.isin(labels, tuple(members))
            values = disparity[mask]
            values = values[np.isfinite(values)]
            if values.size >= 12:
                output[root] = float(np.median(values))
        return output

    def interface_depth_scores(self, labels, graph, disparity):
        """Compare disparity in narrow strips immediately inside each interface."""
        if disparity is None:
            return {}
        kernel_size = 2 * self.config.interface_band + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        scores = {}
        initial_medians = {}
        for identifier in np.unique(labels):
            values = disparity[labels == identifier]
            values = values[np.isfinite(values)]
            if values.size >= 12:
                initial_medians[int(identifier)] = float(np.median(values))
        for pair in graph:
            first_id, second_id = pair
            first_mask = (labels == first_id).astype(np.uint8)
            second_mask = (labels == second_id).astype(np.uint8)
            first_contact = first_mask & cv2.dilate(second_mask, np.ones((3, 3), np.uint8))
            second_contact = second_mask & cv2.dilate(first_mask, np.ones((3, 3), np.uint8))
            first_band = cv2.dilate(first_contact, kernel) & first_mask
            second_band = cv2.dilate(second_contact, kernel) & second_mask
            first_values = disparity[first_band.astype(bool)]
            second_values = disparity[second_band.astype(bool)]
            first_values = first_values[np.isfinite(first_values)]
            second_values = second_values[np.isfinite(second_values)]
            interface_score = None
            if first_values.size >= 6 and second_values.size >= 6:
                first_median, second_median = np.median(first_values), np.median(second_values)
                tolerance = max(1.5, .25 * max(abs(first_median), abs(second_median)))
                continuity = math.exp(-abs(first_median-second_median) / tolerance)
                spread = np.median(np.abs(first_values-first_median)) + np.median(
                    np.abs(second_values-second_median)
                )
                confidence = math.exp(-spread / max(1.5, tolerance))
                coverage = min(1.0, min(first_values.size, second_values.size) / 20.0)
                interface_score = continuity * (.7 * confidence + .3 * coverage)
            regional_score = None
            if first_id in initial_medians and second_id in initial_medians:
                first_median = initial_medians[first_id]
                second_median = initial_medians[second_id]
                tolerance = max(1.5, .25 * max(abs(first_median), abs(second_median)))
                regional_score = math.exp(-abs(first_median-second_median) / tolerance)
            if interface_score is not None:
                scores[pair] = .75 * interface_score + .25 * (
                    regional_score if regional_score is not None else interface_score
                )
        return scores

    def visualize_labels(self, labels, regions, groups=None, base_image=None,
                         depth_by_root=None, fill_regions=False):
        """Draw clear colored boundaries over the original perception image."""
        roots = {}
        if groups:
            for root, members in groups.items():
                for member in members:
                    roots[member] = root
        else:
            groups = {int(identifier): {int(identifier)} for identifier in np.unique(labels)}
        output = (base_image.copy() if base_image is not None
                  else np.full((*labels.shape, 3), 32, np.uint8))
        image_area = labels.shape[0] * labels.shape[1]
        minimum_area = image_area * self.config.minimum_region_fraction
        for root, members in groups.items():
            color = ((root * 67 + 47) % 220 + 25,
                     (root * 113 + 31) % 220 + 25,
                     (root * 173 + 19) % 220 + 25)
            mask = np.isin(labels, tuple(members)).astype(np.uint8)
            aggregate = self._aggregate(regions, members)
            area = aggregate["area"]
            if fill_regions:
                output[mask.astype(bool)] = color
            contours, _hierarchy = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            thickness = 2 if area >= minimum_area else 1
            outline_color = (12, 12, 12) if fill_regions else color
            cv2.drawContours(output, contours, -1, outline_color, thickness, cv2.LINE_8)
            x, y, width, height = aggregate["bbox"]
            if area >= 120 or (width >= 18 and height >= 12):
                protected_variation = .35 * aggregate["lab_std"][0] + .65 * math.hypot(
                    aggregate["lab_std"][1], aggregate["lab_std"][2]
                )
                state = "P" if area >= minimum_area and protected_variation <= 12.0 else (
                    "C" if area >= minimum_area else "S"
                )
                disparity = None if not depth_by_root else depth_by_root.get(root)
                label = f"R{root} {state} {area/image_area:.1%}"
                if disparity is not None:
                    label += f" d{disparity:.1f}"
                cv2.putText(
                    output, label, (x + 2, max(12, y + 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, .32,
                    (255, 255, 255) if fill_regions else color,
                    1, cv2.LINE_AA,
                )
            if area >= minimum_area:
                cv2.rectangle(output, (x, y), (x + width, y + height), color, 1)
        return output

    def propose_eye(self, labels, regions, edges, disparity=None, base_image=None):
        graph = self.adjacency(labels, edges)
        groups, first_diagnostics = self.merge_pass(
            regions, graph, labels.shape, stage="visual"
        )
        visual_groups = {root: set(members) for root, members in groups.items()}
        depth_by_pair = self.interface_depth_scores(labels, graph, disparity)
        groups, second_diagnostics = self.merge_pass(
            regions, graph, labels.shape, depth_by_pair=depth_by_pair,
            stage="stereo",
        )
        depth_by_root = self.region_disparities(labels, groups, disparity) if disparity is not None else {}
        image_area = labels.shape[0] * labels.shape[1]
        minimum_area = image_area * self.config.minimum_region_fraction
        proposals = []
        region_summaries = []
        for root, members in groups.items():
            aggregate = self._aggregate(regions, members)
            variation = .35 * aggregate["lab_std"][0] + .65 * math.hypot(
                aggregate["lab_std"][1], aggregate["lab_std"][2]
            )
            protected = aggregate["area"] >= minimum_area and variation <= 12.0
            summary = {
                "members": sorted(int(value) for value in members),
                "area": int(aggregate["area"]),
                "area_fraction": round(aggregate["area"] / image_area, 5),
                "bbox": [int(value) for value in aggregate["bbox"]],
                "color_variation": round(float(variation), 4),
                "protected": bool(protected),
                "approximate_disparity": depth_by_root.get(root),
            }
            region_summaries.append(summary)
            if aggregate["area"] >= minimum_area:
                proposals.append(aggregate["bbox"])
        return proposals, {
            "initial_region_count": len(regions),
            "final_region_count": len(groups),
            "regions": region_summaries,
            "visual_merges": first_diagnostics,
            "stereo_merges": second_diagnostics,
        }, {
            "initial_mask": self.visualize_labels(
                labels, regions, fill_regions=True
            ),
            "initial_overlay": self.visualize_labels(
                labels, regions, base_image=base_image
            ),
            "visual_merged_mask": self.visualize_labels(
                labels, regions, visual_groups, depth_by_root=depth_by_root,
                fill_regions=True,
            ),
            "stereo_merged_mask": self.visualize_labels(
                labels, regions, groups, depth_by_root=depth_by_root,
                fill_regions=True,
            ),
            "stereo_merged_overlay": self.visualize_labels(
                labels, regions, groups, base_image, depth_by_root
            ),
        }

    def propose(self, left, right):
        left_labels, left_regions, left_edges = self.grow_regions(left)
        right_labels, right_regions, right_edges = self.grow_regions(right)
        try:
            left_disparity = self.approximate_disparity(left, right)
            right_disparity = np.ascontiguousarray(np.fliplr(
                self.approximate_disparity(
                    np.ascontiguousarray(np.fliplr(right)),
                    np.ascontiguousarray(np.fliplr(left)),
                )
            ))
            disparity_error = None
        except cv2.error as error:
            left_disparity = None
            right_disparity = None
            disparity_error = str(error)
        left_windows, left_diagnostics, left_views = self.propose_eye(
            left_labels, left_regions, left_edges, left_disparity, left
        )
        right_windows, right_diagnostics, right_views = self.propose_eye(
            right_labels, right_regions, right_edges, right_disparity, right
        )
        return {
            "left": left_windows,
            "right": right_windows,
            "diagnostics": {
                "mode": "uncalibrated_relative_disparity",
                "disparity_error": disparity_error,
                "left": left_diagnostics,
                "right": right_diagnostics,
            },
            "visualizations": {
                f"{eye}_{name}": image
                for eye, views in (("left", left_views), ("right", right_views))
                for name, image in views.items()
            },
        }
