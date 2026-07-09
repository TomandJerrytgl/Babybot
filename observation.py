from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Observation:
    """
    All raw sensory information received by Babybot
    at one instant.
    """

    # ---------- Basic ----------
    observation_id: int
    timestamp: float

    # ---------- Vision ----------
    left: np.ndarray
    right: np.ndarray

    def center_focus(self):
    left_center = get_center_patch_color(self.left)
    right_center = get_center_patch_color(self.right)

    if color_distance(left_center, right_center) > center_threshold:
        return {"status": "too_close"}

    mask = grow_region_from_center(self.left, left_center)

    return {
        "status": "ok",
        "mask": mask,
        "bbox": get_bbox(mask),
        "area": int(mask.sum() / 255)
    }



        
