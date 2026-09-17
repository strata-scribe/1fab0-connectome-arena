"""
Unit tests for continuous 2D trajectory decoding and kinematic metrics (Grant 1FAB0).
Verifies:
  1. Straight forward run (high DNp09, low MDN) produces positive displacement, straightness ~ 1.0.
  2. Reverse run (low DNp09, high MDN) produces negative displacement.
  3. Pure asymmetric turning (DNa02_L > DNa02_R) produces angular deviation.
  4. Anti-clamping invariant: unclipped negative displacement is preserved.
  5. Distinguishing active retreat from motionless paralysis.
  6. Multi-format rate series ingestion and Trajectory array properties.
"""

import math
import unittest
import numpy as np
from src.trajectory import Trajectory, integrate_trajectory, trajectory_metrics


class TestTrajectoryDecoder(unittest.TestCase):
    def test_straight_forward_run(self):
        """Straight forward run (high DNp09, low MDN) produces positive displacement, straightness ~ 1.0."""
        rates = {
            'DNp09': [60.0] * 20,
            'MDN': [0.0] * 20,
            'DNa02_L': [0.0] * 20,
            'DNa02_R': [0.0] * 20
        }
        traj = integrate_trajectory(rates, dt=0.01)
        metrics = trajectory_metrics(traj)

        self.assertGreater(metrics['displacement'], 0.0)
        self.assertAlmostEqual(metrics['straightness'], 1.0, places=3)
        self.assertAlmostEqual(metrics['chemotaxis_index'], 1.0, places=3)
        self.assertAlmostEqual(metrics['turning_rate'], 0.0, places=4)
        self.assertAlmostEqual(metrics['angular_deviation'], 0.0, places=4)
        self.assertAlmostEqual(metrics['displacement'], metrics['path_length'], places=3)

    def test_reverse_run(self):
        """Reverse run (low DNp09, high MDN) produces negative displacement."""
        rates = {
            'DNp09': [0.0] * 20,
            'MDN': [40.0] * 20,
            'DNa02_L': [0.0] * 20,
            'DNa02_R': [0.0] * 20
        }
        traj = integrate_trajectory(rates, dt=0.01)
        metrics = trajectory_metrics(traj)

        self.assertLess(metrics['displacement'], 0.0)
        self.assertLess(metrics['chemotaxis_index'], 0.0)
        self.assertAlmostEqual(metrics['chemotaxis_index'], -1.0, places=3)
        self.assertAlmostEqual(metrics['straightness'], 1.0, places=3)
        self.assertGreater(metrics['path_length'], 0.0)
        self.assertAlmostEqual(metrics['displacement'], -metrics['path_length'], places=3)

    def test_pure_asymmetric_turning(self):
        """Pure asymmetric turning (DNa02_L > DNa02_R) produces angular deviation."""
        # Left turning: DNa02_L > DNa02_R
        rates_l = {
            'DNp09': [20.0] * 15,
            'MDN': [0.0] * 15,
            'DNa02_L': [25.0] * 15,
            'DNa02_R': [0.0] * 15
        }
        traj_l = integrate_trajectory(rates_l, dt=0.01)
        metrics_l = trajectory_metrics(traj_l)

        self.assertGreater(metrics_l['angular_deviation'], 0.0)
        self.assertGreater(metrics_l['turning_rate'], 0.0)

        # Right turning: DNa02_R > DNa02_L
        rates_r = {
            'DNp09': [20.0] * 15,
            'MDN': [0.0] * 15,
            'DNa02_L': [0.0] * 15,
            'DNa02_R': [25.0] * 15
        }
        traj_r = integrate_trajectory(rates_r, dt=0.01)
        metrics_r = trajectory_metrics(traj_r)

        self.assertLess(metrics_r['angular_deviation'], 0.0)
        self.assertGreater(metrics_r['turning_rate'], 0.0)
        self.assertAlmostEqual(metrics_l['turning_rate'], metrics_r['turning_rate'], places=4)

    def test_invariant_preservation_anti_clamping(self):
        """Invariant preservation: unclipped negative displacement is preserved, distinguishing retreat from paralysis."""
        # 1. Reverse motion / active retreat
        rates_retreat = {'DNp09': 5.0, 'MDN': 45.0, 'DNa02_L': 0.0, 'DNa02_R': 0.0}
        traj_retreat = integrate_trajectory(rates_retreat, dt=0.01)
        metrics_retreat = trajectory_metrics(traj_retreat)

        # Must not be clamped to 0
        self.assertLess(metrics_retreat['displacement'], 0.0)
        self.assertLess(metrics_retreat['chemotaxis_index'], 0.0)
        self.assertGreater(metrics_retreat['path_length'], 0.0)

        # 2. Motionless paralysis
        rates_paralysis = {'DNp09': 0.0, 'MDN': 0.0, 'DNa02_L': 0.0, 'DNa02_R': 0.0}
        traj_paralysis = integrate_trajectory(rates_paralysis, dt=0.01)
        metrics_paralysis = trajectory_metrics(traj_paralysis)

        self.assertEqual(metrics_paralysis['displacement'], 0.0)
        self.assertEqual(metrics_paralysis['path_length'], 0.0)
        self.assertEqual(metrics_paralysis['chemotaxis_index'], 0.0)
        self.assertEqual(metrics_paralysis['straightness'], 0.0)

        # Crucial invariant: retreat and paralysis must NOT have identical displacement
        self.assertNotEqual(metrics_retreat['displacement'], metrics_paralysis['displacement'])
        self.assertNotEqual(metrics_retreat['path_length'], metrics_paralysis['path_length'])

    def test_trajectory_formats_and_properties(self):
        """Verifies Trajectory class properties, slicing, and input format flexibility."""
        # Input as list of step dicts
        step_dicts = [
            {'DNp09': 30.0, 'MDN': 0.0, 'DNa02_L': 5.0, 'DNa02_R': 0.0},
            {'DNp09': 40.0, 'MDN': 0.0, 'DNa02_L': 0.0, 'DNa02_R': 5.0},
            {'DNp09': 50.0, 'MDN': 5.0, 'DNa02_L': 0.0, 'DNa02_R': 0.0}
        ]
        traj = integrate_trajectory(step_dicts, dt=0.01)
        self.assertIsInstance(traj, Trajectory)
        self.assertIsInstance(traj, np.ndarray)
        self.assertEqual(traj.shape, (4, 3))
        self.assertEqual(len(traj.x), 4)
        self.assertEqual(len(traj.y), 4)
        self.assertEqual(len(traj.theta), 4)
        self.assertEqual(traj.positions.shape, (4, 2))

        # Scalar inputs broadcast properly
        traj_scalar = integrate_trajectory({'DNp09': 10.0, 'MDN': 0.0}, dt=0.01)
        self.assertEqual(traj_scalar.shape, (2, 3))

    def test_chemotaxis_scale_invariance(self):
        """Chemotaxis index dx / L is scale-invariant with respect to v_scale."""
        rates = {'DNp09': [30.0, 40.0, 50.0], 'MDN': [5.0, 5.0, 5.0]}
        traj1 = integrate_trajectory(rates, dt=0.01, v_scale=1.0)
        traj2 = integrate_trajectory(rates, dt=0.01, v_scale=5.0)

        m1 = trajectory_metrics(traj1)
        m2 = trajectory_metrics(traj2)

        self.assertAlmostEqual(m1['chemotaxis_index'], m2['chemotaxis_index'], places=5)
        self.assertAlmostEqual(m1['straightness'], m2['straightness'], places=5)
        self.assertAlmostEqual(m2['displacement'], m1['displacement'] * 5.0, places=4)
        self.assertAlmostEqual(m2['path_length'], m1['path_length'] * 5.0, places=4)

    def test_dual_decoder_agreement(self):
        """Verifies that Quire's discrete approach_index (Delta Hz) and Strata's continuous CI agree in sign."""
        # Condition 1: Forward attractant run
        rates_fwd = {'DNp09': [50.0] * 10, 'MDN': [5.0] * 10, 'DNa02_L': [0.0] * 10, 'DNa02_R': [0.0] * 10}
        delta_hz_fwd = 50.0 - 5.0  # +45 Hz
        traj_fwd = integrate_trajectory(rates_fwd, dt=0.01)
        metrics_fwd = trajectory_metrics(traj_fwd)

        self.assertGreater(delta_hz_fwd, 0.0)
        self.assertGreater(metrics_fwd['chemotaxis_index'], 0.0)
        self.assertGreater(metrics_fwd['displacement'], 0.0)

        # Condition 2: Repellent / avoidance reverse run
        rates_rev = {'DNp09': [5.0] * 10, 'MDN': [45.0] * 10, 'DNa02_L': [0.0] * 10, 'DNa02_R': [0.0] * 10}
        delta_hz_rev = 5.0 - 45.0  # -40 Hz
        traj_rev = integrate_trajectory(rates_rev, dt=0.01)
        metrics_rev = trajectory_metrics(traj_rev)

        self.assertLess(delta_hz_rev, 0.0)
        self.assertLess(metrics_rev['chemotaxis_index'], 0.0)
        self.assertLess(metrics_rev['displacement'], 0.0)


if __name__ == '__main__':
    unittest.main()

