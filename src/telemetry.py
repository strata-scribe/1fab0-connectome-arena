"""
Trajectory & Kinematic Telemetry Engine (Grant 1fab0)
Computes continuous unclipped distance metrics, trajectory displacement,
path divergence, and distinguishes active wandering from quiescent paralysis.
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

def calculate_unclipped_displacement(
    start_pos: Union[List[float], np.ndarray],
    final_pos: Union[List[float], np.ndarray],
    target_pos: Union[List[float], np.ndarray]
) -> Tuple[float, float, float, float]:
    """
    Computes initial distance, final distance, raw displacement toward target,
    and unclipped chemotaxis index.
    
    Returns
    -------
    (d0, dfinal, displacement, ci_unclipped)
      - d0: Euclidean distance from start_pos to target_pos
      - dfinal: Euclidean distance from final_pos to target_pos
      - displacement: d0 - dfinal (positive = progress toward target, negative = retreat)
      - ci_unclipped: (d0 - dfinal) / d0 (unbounded, unclipped by max(0, ...))
    """
    p0 = np.array(start_pos, dtype=np.float64)
    pf = np.array(final_pos, dtype=np.float64)
    pt = np.array(target_pos, dtype=np.float64)

    d0 = float(np.linalg.norm(p0 - pt))
    dfinal = float(np.linalg.norm(pf - pt))
    displacement = d0 - dfinal

    if d0 < 1e-6:
        ci_unclipped = 0.0
    else:
        ci_unclipped = displacement / d0

    return d0, dfinal, displacement, ci_unclipped


def calculate_path_length(trajectory: Union[List[List[float]], np.ndarray]) -> float:
    """
    Calculates total Euclidean arc length along a recorded 2D trajectory.
    """
    if len(trajectory) < 2:
        return 0.0
    pts = np.array(trajectory, dtype=np.float64)
    diffs = np.diff(pts, axis=0)
    step_lens = np.sqrt(np.sum(diffs ** 2, axis=1))
    return float(np.sum(step_lens))


def classify_behavior(
    path_length: float,
    displacement: float,
    paralysis_threshold: float = 5.0
) -> str:
    """
    Distinguishes wandering from paralysis based on path length and displacement.
    
    - 'paralysis': Agent did not move appreciably (path_length < paralysis_threshold)
    - 'wandering': Agent actively moved but failed to progress toward target (displacement <= 0)
    - 'directed': Agent moved and made net progress toward target (displacement > 0)
    """
    if path_length < paralysis_threshold:
        return "paralysis"
    elif displacement <= 0.0:
        return "wandering"
    else:
        return "directed"


def extract_trajectory_metrics(
    start_pos: Union[List[float], np.ndarray],
    final_pos: Union[List[float], np.ndarray],
    target_pos: Union[List[float], np.ndarray],
    trajectory: Optional[Union[List[List[float]], np.ndarray]] = None,
    paralysis_threshold: float = 5.0
) -> Dict[str, Union[float, str]]:
    """
    Comprehensive kinematic and chemotactic telemetry extraction.
    """
    d0, dfinal, displacement, ci_unclipped = calculate_unclipped_displacement(
        start_pos, final_pos, target_pos
    )
    ci_clipped = max(0.0, ci_unclipped)

    p0 = np.array(start_pos, dtype=np.float64)
    pf = np.array(final_pos, dtype=np.float64)
    straight_line_dist = float(np.linalg.norm(pf - p0))

    if trajectory is not None and len(trajectory) >= 2:
        path_length = calculate_path_length(trajectory)
    else:
        path_length = straight_line_dist

    tortuosity = path_length / max(1e-5, straight_line_dist)
    directed_efficiency = displacement / max(1e-5, path_length)
    behavior = classify_behavior(path_length, displacement, paralysis_threshold=paralysis_threshold)

    return {
        "d0": d0,
        "dfinal": dfinal,
        "displacement": displacement,
        "ci_unclipped": ci_unclipped,
        "ci_clipped": ci_clipped,
        "path_length": path_length,
        "straight_line_dist": straight_line_dist,
        "tortuosity": tortuosity,
        "directed_efficiency": directed_efficiency,
        "behavior": behavior
    }
