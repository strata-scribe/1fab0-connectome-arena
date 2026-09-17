"""
Continuous 2D Trajectory Decoder & Kinematic Telemetry Engine (Grant 1fab0)

Converts instantaneous firing rates of descending neuron classes into physical 2D motion
(x(t), y(t), theta(t)):
  - DNp09: forward-walking descending neurons (Bidaye et al. 2020)
  - MDN: moonwalker backward-walking descending neurons (Bidaye et al. 2014)
  - DNa02_L / DNa02_R: steering descending neurons modulating yaw / turning rate

Computes unclipped displacement (dx = x_end - x_0), cumulative path length
(L = sum sqrt(dx^2 + dy^2)), straightness, chemotaxis index (CI = dx / L), and turning rate.
Enforces the strict anti-clamping invariant: negative displacement is preserved to distinguish
active repulsion/wandering from quiescent paralysis.
"""

import math
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np


class Trajectory(np.ndarray):
    """
    2D spatial trajectory array of shape (N, 3) representing [x, y, theta] over time.
    Subclasses np.ndarray for zero-copy vectorized operations and direct slicing,
    while providing semantic property accessors (.x, .y, .theta, .positions).
    """
    def __new__(cls, input_array: Any, dt: float = 0.01):
        obj = np.asarray(input_array, dtype=np.float64).view(cls)
        if obj.ndim != 2 or obj.shape[1] < 2:
            raise ValueError(f"Trajectory array must have shape (N, >=2), got {obj.shape}")
        if obj.shape[1] == 2:
            thetas = np.zeros((obj.shape[0], 1), dtype=np.float64)
            obj = np.hstack([obj, thetas]).view(cls)
        obj.dt = float(dt)
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        if obj is None:
            return
        self.dt = getattr(obj, "dt", 0.01)

    @property
    def x(self) -> np.ndarray:
        return self[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self[:, 1]

    @property
    def theta(self) -> np.ndarray:
        return self[:, 2]

    @property
    def positions(self) -> np.ndarray:
        return self[:, :2]


def _normalize_rate_series(
    rate_series: Union[Dict[str, Any], List[Dict[str, float]], np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Normalizes rate_series in various formats into aligned 1D numpy arrays:
    (r_fwd, r_bwd, r_left, r_right, n_steps).
    """
    if isinstance(rate_series, list):
        n_steps = len(rate_series)
        r_fwd = np.zeros(n_steps, dtype=np.float64)
        r_bwd = np.zeros(n_steps, dtype=np.float64)
        r_left = np.zeros(n_steps, dtype=np.float64)
        r_right = np.zeros(n_steps, dtype=np.float64)
        for i, step in enumerate(rate_series):
            for k, v in step.items():
                k_lower = k.lower()
                val = float(v)
                if k_lower in ('dnp09', 'fwd', 'forward'):
                    r_fwd[i] = val
                elif k_lower in ('mdn', 'bwd', 'backward'):
                    r_bwd[i] = val
                elif k_lower in ('dna02_l', 'dna02_left', 'left'):
                    r_left[i] = val
                elif k_lower in ('dna02_r', 'dna02_right', 'right'):
                    r_right[i] = val
        return r_fwd, r_bwd, r_left, r_right, n_steps

    if isinstance(rate_series, dict):
        n_steps = 1
        for v in rate_series.values():
            if hasattr(v, '__len__') and not isinstance(v, (str, bytes)):
                n_steps = max(n_steps, len(v))

        def _get_array(names: Tuple[str, ...]) -> np.ndarray:
            for k, v in rate_series.items():
                if k.lower() in names:
                    if hasattr(v, '__len__') and not isinstance(v, (str, bytes)):
                        arr = np.asarray(v, dtype=np.float64)
                        if len(arr) == n_steps:
                            return arr
                        out = np.zeros(n_steps, dtype=np.float64)
                        out[:len(arr)] = arr
                        return out
                    else:
                        return np.full(n_steps, float(v), dtype=np.float64)
            return np.zeros(n_steps, dtype=np.float64)

        r_fwd = _get_array(('dnp09', 'fwd', 'forward'))
        r_bwd = _get_array(('mdn', 'bwd', 'backward'))
        r_left = _get_array(('dna02_l', 'dna02_left', 'left'))
        r_right = _get_array(('dna02_r', 'dna02_right', 'right'))
        return r_fwd, r_bwd, r_left, r_right, n_steps

    raise TypeError(f"Unsupported rate_series type: {type(rate_series)}")


def integrate_trajectory(
    rate_series: Union[Dict[str, Any], List[Dict[str, float]], np.ndarray],
    dt: float = 0.01,
    v_scale: float = 1.0,
    turn_scale: float = 1.0,
    init_pos: Tuple[float, float] = (0.0, 0.0),
    init_heading: float = 0.0
) -> Trajectory:
    """
    Integrates descending firing rates over discrete timesteps dt into continuous 2D motion (x, y, theta).

    Parameters
    ----------
    rate_series : dict or list of dicts
        Per-step firing rates (Hz) for 'DNp09' (forward), 'MDN' (reverse),
        'DNa02_L' (turn left), and 'DNa02_R' (turn right).
    dt : float
        Integration time step in seconds (default: 0.01 s = 10 ms).
    v_scale : float
        Linear velocity scale factor (forward velocity = v_scale * (r_DNp09 - r_MDN)).
    turn_scale : float
        Angular velocity scale factor (yaw rate = turn_scale * (r_DNa02_L - r_DNa02_R)).
    init_pos : (x0, y0)
        Initial Cartesian coordinates (default: (0.0, 0.0)).
    init_heading : float
        Initial orientation in radians (0.0 = pointing along +x axis toward target).

    Returns
    -------
    Trajectory
        (N+1, 3) ndarray of [x, y, theta] representing trajectory states over time.
    """
    r_fwd, r_bwd, r_left, r_right, n_steps = _normalize_rate_series(rate_series)

    states = np.zeros((n_steps + 1, 3), dtype=np.float64)
    states[0, 0] = init_pos[0]
    states[0, 1] = init_pos[1]
    states[0, 2] = init_heading

    curr_x = float(init_pos[0])
    curr_y = float(init_pos[1])
    curr_theta = float(init_heading)

    for i in range(n_steps):
        v = v_scale * (r_fwd[i] - r_bwd[i])
        omega = turn_scale * (r_left[i] - r_right[i])

        mid_theta = curr_theta + 0.5 * omega * dt
        curr_x += v * math.cos(mid_theta) * dt
        curr_y += v * math.sin(mid_theta) * dt
        curr_theta += omega * dt

        states[i + 1, 0] = curr_x
        states[i + 1, 1] = curr_y
        states[i + 1, 2] = curr_theta

    return Trajectory(states, dt=dt)


def trajectory_metrics(
    trajectory: Any,
    dt: Optional[float] = None,
    round_digits: Optional[int] = 6
) -> Dict[str, float]:
    """
    Extracts kinematic and chemotactic metrics from a 2D trajectory.

    Enforces the Anti-Clamping Metric Invariant:
    Displacement dx = x_end - x_0 and chemotaxis_index CI = dx / L are signed
    and never clamped with max(0, ...), preserving clear discrimination
    between active retreat/repulsion (CI < 0) and motionless paralysis (L ~ 0).

    Parameters
    ----------
    trajectory : Trajectory, np.ndarray, list of coordinates, or dict
        Trajectory points of shape (N, >=2).
    dt : float, optional
        Integration timestep in seconds. If None, inferred from trajectory.dt or default 0.01.
    round_digits : int, optional
        Decimal places to round returned values (default: 6). Pass None for raw floats.

    Returns
    -------
    dict
        - 'displacement': float (x_end - x_0, unclipped signed progress along x)
        - 'path_length': float (cumulative Euclidean path length)
        - 'chemotaxis_index': float (displacement / path_length, bounded in [-1.0, 1.0])
        - 'straightness': float (net_distance / path_length, bounded in [0.0, 1.0])
        - 'turning_rate': float (mean absolute yaw rate in rad/s)
        - 'angular_deviation': float (theta_end - theta_0 in radians)
    """
    if isinstance(trajectory, dict):
        if 'trajectory' in trajectory:
            trajectory = trajectory['trajectory']
        elif 'positions' in trajectory:
            trajectory = trajectory['positions']

    arr = np.asarray(trajectory, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 2:
        return {
            'displacement': 0.0,
            'path_length': 0.0,
            'chemotaxis_index': 0.0,
            'straightness': 0.0,
            'turning_rate': 0.0,
            'angular_deviation': 0.0
        }

    n_points = arr.shape[0]
    p0 = arr[0, :2]
    pend = arr[-1, :2]

    displacement = float(pend[0] - p0[0])

    if n_points >= 2:
        diffs = np.diff(arr[:, :2], axis=0)
        step_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
        path_length = float(np.sum(step_lengths))
    else:
        path_length = 0.0

    net_distance = float(np.linalg.norm(pend - p0))

    if path_length > 1e-9:
        chemotaxis_index = float(displacement / path_length)
        straightness = float(min(1.0, net_distance / path_length))
    else:
        chemotaxis_index = 0.0
        straightness = 0.0

    if arr.shape[1] >= 3:
        headings = arr[:, 2]
        angular_deviation = float(headings[-1] - headings[0])
        step_dt = dt if dt is not None else getattr(trajectory, 'dt', 0.01)
        if n_points >= 2 and step_dt > 0:
            d_thetas = np.diff(headings)
            turning_rate = float(np.mean(np.abs(d_thetas)) / step_dt)
        else:
            turning_rate = 0.0
    else:
        angular_deviation = 0.0
        turning_rate = 0.0

    metrics = {
        'displacement': displacement,
        'path_length': path_length,
        'chemotaxis_index': chemotaxis_index,
        'straightness': straightness,
        'turning_rate': turning_rate,
        'angular_deviation': angular_deviation
    }

    if round_digits is not None:
        metrics = {k: round(v, round_digits) for k, v in metrics.items()}

    return metrics
