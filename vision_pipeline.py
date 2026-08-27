"""Hardware-independent visual processing shared by Awake and Dreaming."""

from __future__ import annotations

import time
import traceback

import cv2

from attention import Attention
from region_proposal import RegionProposalConfig, StereoRegionProposer


def default_attention_settings():
    return {
        "minimum_objectness": 0.45,
        "maximum_candidates": 10,
        "partial_overlap_iou": 0.30,
        "minimum_region_fraction": 0.03,
        "region_growth_threshold": 14.0,
        "minimum_merge_score": 0.65,
        "stereo_merge_weight": 0.12,
    }


def calculate_attention_pair(perception, settings):
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
            "diagnostics": {"mode": "multiscale_fallback",
                            "proposal_error": traceback.format_exc()},
        }
    arguments = {
        "verbose": False,
        "objectness_threshold": settings["minimum_objectness"],
        "maximum_candidates": settings["maximum_candidates"],
        "partial_overlap_iou": settings["partial_overlap_iou"],
    }
    result = {"left": [], "right": [], "left_elapsed": 0.0,
              "right_elapsed": 0.0, "left_error": None, "right_error": None}
    for eye in ("left", "right"):
        eye_started = time.perf_counter()
        try:
            attention = Attention(perception, eye=eye,
                                  proposal_windows=region_result[eye], **arguments)
            result[eye] = [candidate.copy() for candidate in attention.candidates]
            result[f"{eye}_elapsed"] = attention.elapsed_time
        except Exception:
            result[f"{eye}_elapsed"] = time.perf_counter() - eye_started
            result[f"{eye}_error"] = traceback.format_exc()
    result["elapsed_time"] = time.perf_counter() - started
    result["region_diagnostics"] = region_result["diagnostics"]
    result["region_visualizations"] = region_result["visualizations"]
    return result


def encode_jpeg(image, quality):
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode preview image")
    return encoded.tobytes()


def encode_preview(image, candidates, quality):
    annotated = image.copy()
    colors = [(0, 0, 255), (0, 165, 255), (0, 255, 255),
              (0, 255, 0), (255, 0, 0)]
    for rank, candidate in enumerate(candidates[:10], start=1):
        x, y, width, height = candidate["window"]
        color = colors[(rank - 1) % len(colors)]
        cv2.rectangle(annotated, (x, y), (x + width, y + height),
                      color, 3 if rank == 1 else 2)
        cv2.putText(
            annotated,
            f"#{candidate.get('rank', rank)} R:{candidate.get('ranking_score', candidate.get('score', 0.0)):.2f} O:{candidate.get('objectness', 0.0):.2f}",
            (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            color, 1, cv2.LINE_AA,
        )
    return encode_jpeg(annotated, quality)
