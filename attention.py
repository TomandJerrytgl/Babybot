import time

import cv2
import numpy as np


class Attention:
    """Pyramid-based visual attention with coarse-to-fine refinement."""

    def __init__(
            self,
            observation,
            object=None,
            eye="left",
            previous_focus=None,
            verbose=True):
        self.observation = observation
        self.object = object
        if eye not in ("left", "right"):
            raise ValueError("eye must be 'left' or 'right'")
        self.eye = eye
        self.previous_focus = previous_focus
        self.verbose = verbose
        self.candidates = []
        self.elapsed_time = 0.0

        if object is None:
            self.focus = self.default_focus()
        else:
            self.focus = None

    def default_focus(self):
        image = getattr(self.observation, self.eye)
        return self.find_attention_window(
            image,
            previous_focus=self.previous_focus,
        )

    def brightness_preference(self, brightness):
        ideal = 170.0
        sigma = 60.0
        score = np.exp(
            -((brightness - ideal) ** 2) / (2.0 * sigma ** 2)
        )
        return float(np.clip(score, 0.0, 1.0))

    def center_preference(self, x, y, width, height):
        center_x = width / 2.0
        center_y = height / 2.0
        distance = np.sqrt(
            (x - center_x) ** 2 + (y - center_y) ** 2
        )
        max_distance = np.sqrt(center_x ** 2 + center_y ** 2)

        if max_distance == 0:
            return 1.0

        score = 1.0 - distance / max_distance
        return float(np.clip(score, 0.0, 1.0))

    def contrast_score(self, image, x, y, window_size):
        image_height, image_width = image.shape[:2]
        window = image[y:y + window_size, x:x + window_size]

        if window.size == 0:
            return 0.0

        margin = window_size // 2
        context_x1 = max(0, x - margin)
        context_y1 = max(0, y - margin)
        context_x2 = min(image_width, x + window_size + margin)
        context_y2 = min(image_height, y + window_size + margin)
        context = image[context_y1:context_y2, context_x1:context_x2]

        if context.size == 0:
            return 0.0

        window_lab = cv2.cvtColor(
            window, cv2.COLOR_BGR2LAB
        ).astype(np.float32)
        context_lab = cv2.cvtColor(
            context, cv2.COLOR_BGR2LAB
        ).astype(np.float32)

        background_mask = np.ones(context.shape[:2], dtype=bool)
        local_x = x - context_x1
        local_y = y - context_y1
        background_mask[
            local_y:local_y + window_size,
            local_x:local_x + window_size
        ] = False
        background_pixels = context_lab[background_mask]

        if background_pixels.size == 0:
            background_contrast = 0.0
        else:
            window_mean = np.mean(window_lab.reshape(-1, 3), axis=0)
            background_mean = np.mean(background_pixels, axis=0)
            color_distance = np.linalg.norm(window_mean - background_mean)
            background_contrast = np.clip(
                color_distance / 100.0, 0.0, 1.0
            )

        gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        edge_density = np.mean(edges > 0)
        edge_score = np.clip(edge_density / 0.15, 0.0, 1.0)

        score = 0.55 * background_contrast + 0.45 * edge_score
        return float(np.clip(score, 0.0, 1.0))

    def color_preference(self, window):
        if window.size == 0:
            return 0.0

        hsv = cv2.cvtColor(window, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.float32) / 255.0
        threshold = np.percentile(saturation, 70)
        colorful_pixels = saturation[saturation >= threshold]

        if colorful_pixels.size == 0:
            return 0.0

        score = np.mean(colorful_pixels)
        return float(np.clip(score, 0.0, 1.0))

    def visual_saliency(self, image, x, y, window_size):
        image_height, image_width = image.shape[:2]
        x = int(x)
        y = int(y)
        window_size = int(window_size)

        if window_size <= 0 or x < 0 or y < 0:
            return None
        if x + window_size > image_width:
            return None
        if y + window_size > image_height:
            return None

        window = image[y:y + window_size, x:x + window_size]
        if window.size == 0:
            return None

        window_float = window.astype(np.float32)
        b = window_float[:, :, 0]
        g = window_float[:, :, 1]
        r = window_float[:, :, 2]
        brightness = np.mean(0.114 * b + 0.587 * g + 0.299 * r)
        brightness_score = self.brightness_preference(brightness)
        contrast_score = self.contrast_score(image, x, y, window_size)
        color_score = self.color_preference(window)

        visual_score = (
            0.15 * brightness_score
            + 0.50 * contrast_score
            + 0.35 * color_score
        )
        visual_score = float(np.clip(visual_score, 0.0, 1.0))

        return (
            brightness_score,
            contrast_score,
            color_score,
            visual_score,
        )

    def evaluate_window(self, image, x, y, window_size, level):
        image_height, image_width = image.shape[:2]
        x = int(x)
        y = int(y)
        window_size = int(window_size)

        if window_size < 8:
            return None
        if window_size > image_width or window_size > image_height:
            return None

        x = int(np.clip(x, 0, image_width - window_size))
        y = int(np.clip(y, 0, image_height - window_size))
        result = self.visual_saliency(image, x, y, window_size)

        if result is None:
            return None

        brightness_score, contrast_score, color_score, visual_score = result
        window_center_x = x + window_size / 2.0
        window_center_y = y + window_size / 2.0
        center_score = self.center_preference(
            window_center_x,
            window_center_y,
            image_width,
            image_height,
        )
        final_score = 0.95 * visual_score + 0.05 * center_score
        final_score = float(np.clip(final_score, 0.0, 1.0))

        return {
            "window": (x, y, window_size, window_size),
            "level": level,
            "brightness": brightness_score,
            "contrast": contrast_score,
            "color": color_score,
            "visual": visual_score,
            "center": center_score,
            "score": final_score,
        }

    def build_pyramid(self, image, pyramid_scale=0.5, coarse_min_side=240):
        if image is None:
            raise ValueError("Input image is None.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a BGR color image.")
        if not 0.0 < pyramid_scale < 1.0:
            raise ValueError("pyramid_scale must be between 0 and 1.")
        if coarse_min_side < 8:
            raise ValueError("coarse_min_side must be at least 8.")

        pyramid = [image]
        current_image = image

        while min(current_image.shape[:2]) > coarse_min_side:
            current_height, current_width = current_image.shape[:2]
            new_width = max(1, int(current_width * pyramid_scale))
            new_height = max(1, int(current_height * pyramid_scale))

            if new_width == current_width and new_height == current_height:
                break

            current_image = cv2.resize(
                current_image,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA,
            )
            pyramid.append(current_image)

        pyramid.reverse()
        return pyramid

    def generate_positions(self, image_length, window_size, step):
        maximum_start = image_length - window_size
        if maximum_start < 0:
            return []

        positions = list(range(0, maximum_start + 1, step))
        if not positions:
            positions = [0]
        elif positions[-1] != maximum_start:
            positions.append(maximum_start)
        return positions

    def coarse_search(self, image, level, max_candidates=30):
        image_height, image_width = image.shape[:2]
        minimum_side = min(image_height, image_width)
        coarse_window_size = int(np.clip(minimum_side // 4, 32, 96))
        coarse_window_size = min(
            coarse_window_size, image_width, image_height
        )
        coarse_step = max(8, coarse_window_size // 2)

        x_positions = self.generate_positions(
            image_width, coarse_window_size, coarse_step
        )
        y_positions = self.generate_positions(
            image_height, coarse_window_size, coarse_step
        )
        candidates = []

        for y in y_positions:
            for x in x_positions:
                candidate = self.evaluate_window(
                    image, x, y, coarse_window_size, level
                )
                if candidate is not None:
                    candidates.append(candidate)

        # NMS is applied before candidates enter the next level.
        return self.suppress_overlaps(
            candidates,
            overlap_threshold=0.65,
            maximum_candidates=max_candidates,
        )

    def window_sizes_for_level(self, image_shape):
        """Create level-local sizes; no box size is inherited."""
        image_height, image_width = image_shape[:2]
        minimum_side = min(image_height, image_width)

        # Test several absolute fractions independently at every level.
        raw_sizes = [
            minimum_side / 8.0,
            minimum_side / 4.0,
            minimum_side / 2.0,
        ]
        sizes = []

        for raw_size in raw_sizes:
            candidate_size = int(round(raw_size))
            candidate_size = int(
                np.clip(candidate_size, 16, minimum_side)
            )
            if candidate_size not in sizes:
                sizes.append(candidate_size)

        return sizes

    def refine_candidates(
            self,
            previous_candidates,
            previous_shape,
            current_image,
            level,
            max_candidates=30):
        previous_height, previous_width = previous_shape[:2]
        current_height, current_width = current_image.shape[:2]
        scale_x = current_width / previous_width
        scale_y = current_height / previous_height
        candidate_sizes = self.window_sizes_for_level(current_image.shape)
        refined_candidates = []
        evaluated_windows = set()

        for previous_candidate in previous_candidates:
            previous_x, previous_y, previous_w, previous_h = (
                previous_candidate["window"]
            )

            # Only the center is inherited from the previous level.
            current_center_x = (
                previous_x + previous_w / 2.0
            ) * scale_x
            current_center_y = (
                previous_y + previous_h / 2.0
            ) * scale_y

            for candidate_size in candidate_sizes:
                # A 7 x 7 grid centered on the mapped center. Adjacent grid
                # points are separated by one eighth of the window size.
                grid_step = max(2, int(round(candidate_size / 8.0)))

                for grid_y in range(-3, 4):
                    for grid_x in range(-3, 4):
                        candidate_center_x = current_center_x + grid_x * grid_step
                        candidate_center_y = current_center_y + grid_y * grid_step
                        candidate_x = int(round(
                            candidate_center_x - candidate_size / 2.0
                        ))
                        candidate_y = int(round(
                            candidate_center_y - candidate_size / 2.0
                        ))
                        candidate_x = int(np.clip(
                            candidate_x, 0, current_width - candidate_size
                        ))
                        candidate_y = int(np.clip(
                            candidate_y, 0, current_height - candidate_size
                        ))
                        window_key = (
                            candidate_x, candidate_y, candidate_size
                        )

                        if window_key in evaluated_windows:
                            continue
                        evaluated_windows.add(window_key)

                        candidate = self.evaluate_window(
                            current_image,
                            candidate_x,
                            candidate_y,
                            candidate_size,
                            level,
                        )
                        if candidate is not None:
                            refined_candidates.append(candidate)

        # Every refinement level performs NMS before propagation.
        return self.suppress_overlaps(
            refined_candidates,
            overlap_threshold=0.65,
            maximum_candidates=max_candidates,
        )

    def overlap_ratio(self, first_window, second_window):
        first_x, first_y, first_w, first_h = first_window
        second_x, second_y, second_w, second_h = second_window
        intersection_x1 = max(first_x, second_x)
        intersection_y1 = max(first_y, second_y)
        intersection_x2 = min(first_x + first_w, second_x + second_w)
        intersection_y2 = min(first_y + first_h, second_y + second_h)
        intersection_width = max(0, intersection_x2 - intersection_x1)
        intersection_height = max(0, intersection_y2 - intersection_y1)
        intersection_area = intersection_width * intersection_height
        smaller_area = min(first_w * first_h, second_w * second_h)

        if smaller_area <= 0:
            return 0.0
        return float(intersection_area / smaller_area)

    def suppress_overlaps(
            self,
            candidates,
            overlap_threshold=0.70,
            maximum_candidates=None):
        sorted_candidates = sorted(
            candidates, key=lambda item: item["score"], reverse=True
        )
        selected_candidates = []

        for candidate in sorted_candidates:
            if all(
                self.overlap_ratio(
                    candidate["window"], selected["window"]
                ) < overlap_threshold
                for selected in selected_candidates
            ):
                selected_candidates.append(candidate)

                if (
                    maximum_candidates is not None
                    and len(selected_candidates) >= maximum_candidates
                ):
                    break

        return selected_candidates

    def candidate_limit_for_level(
            self, level, number_of_levels, coarse_candidates, top_k):
        """Reduce the computation budget as attention moves upward."""
        if level <= 0:
            return coarse_candidates

        focus_limits = [15, 8]
        refinement_index = level - 1

        if level == number_of_levels - 1:
            return max(top_k, 5)
        if refinement_index < len(focus_limits):
            return min(coarse_candidates, focus_limits[refinement_index])
        return max(top_k, 5)

    def find_attention_window(
            self,
            image,
            previous_focus=None,
            local_score_threshold=0.20,
            top_k=5,
            coarse_candidates=30,
            pyramid_scale=0.5,
            coarse_min_side=240):
        """Search the previous focus neighborhood before scanning globally."""
        start_time = time.perf_counter()
        attempts = []

        if previous_focus is not None:
            for expansion in (1.5, 3.0):
                region = self.expanded_region(
                    previous_focus, image.shape, expansion
                )
                if region is not None and region not in attempts:
                    attempts.append(region)

        for region in attempts:
            x, y, width, height = region
            crop = image[y:y + height, x:x + width]
            candidates, levels = self._search_pyramid(
                crop,
                top_k,
                coarse_candidates,
                pyramid_scale,
                min(coarse_min_side, min(crop.shape[:2])),
            )
            candidates = self.offset_candidates(candidates, x, y)
            if candidates and candidates[0]["score"] >= local_score_threshold:
                self.candidates = candidates
                self.elapsed_time = time.perf_counter() - start_time
                if self.verbose:
                    self.print_results(candidates, levels)
                return candidates[0]["window"]

        candidates, levels = self._search_pyramid(
            image,
            top_k,
            coarse_candidates,
            pyramid_scale,
            coarse_min_side,
        )
        self.candidates = candidates
        self.elapsed_time = time.perf_counter() - start_time
        if self.verbose:
            self.print_results(candidates, levels)

        if not candidates:
            return None
        return candidates[0]["window"]

    def _search_pyramid(
            self,
            image,
            top_k,
            coarse_candidates,
            pyramid_scale,
            coarse_min_side):
        pyramid = self.build_pyramid(
            image, pyramid_scale, coarse_min_side
        )
        candidates = self.coarse_search(
            pyramid[0], level=0, max_candidates=coarse_candidates
        )

        for level in range(1, len(pyramid)):
            if not candidates:
                break

            maximum_candidates = self.candidate_limit_for_level(
                level,
                len(pyramid),
                coarse_candidates,
                top_k,
            )
            candidates = self.refine_candidates(
                previous_candidates=candidates,
                previous_shape=pyramid[level - 1].shape,
                current_image=pyramid[level],
                level=level,
                max_candidates=maximum_candidates,
            )

        candidates = self.suppress_overlaps(
            candidates,
            overlap_threshold=0.70,
            maximum_candidates=top_k,
        )
        return candidates, len(pyramid)

    def expanded_region(self, window, image_shape, expansion):
        image_height, image_width = image_shape[:2]
        x, y, width, height = window
        region_width = min(image_width, max(32, int(round(width * expansion))))
        region_height = min(image_height, max(32, int(round(height * expansion))))
        center_x = x + width / 2.0
        center_y = y + height / 2.0
        region_x = int(np.clip(round(center_x - region_width / 2.0), 0, image_width - region_width))
        region_y = int(np.clip(round(center_y - region_height / 2.0), 0, image_height - region_height))
        if region_width < 8 or region_height < 8:
            return None
        return region_x, region_y, region_width, region_height

    def offset_candidates(self, candidates, offset_x, offset_y):
        translated = []
        for candidate in candidates:
            item = candidate.copy()
            x, y, width, height = item["window"]
            item["window"] = (
                x + offset_x, y + offset_y, width, height
            )
            translated.append(item)
        return translated

    def print_results(self, candidates, pyramid_levels):
        print("---------------------------------------")
        print(f"Pyramid Levels : {pyramid_levels}")
        print(f"Attention Time : {self.elapsed_time * 1000:.2f} ms")

        if self.elapsed_time > 0:
            print(f"Estimated FPS  : {1.0 / self.elapsed_time:.2f}")

        print(f"Candidates     : {len(candidates)}")

        for rank, candidate in enumerate(candidates, start=1):
            print("---------------------------------------")
            print(f"Rank         : {rank}")
            print(f"Window       : {candidate['window']}")
            print(f"Brightness   : {candidate['brightness']:.3f}")
            print(f"Contrast     : {candidate['contrast']:.3f}")
            print(f"Color        : {candidate['color']:.3f}")
            print(f"Visual Score : {candidate['visual']:.3f}")
            print(f"Center Score : {candidate['center']:.3f}")
            print(f"Final Score  : {candidate['score']:.3f}")

        print("---------------------------------------")
