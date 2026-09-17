"""
Unit & Invariant Tests for Drosophila Connectome Simulation (Grant 1fab0)
"""

import numpy as np
import random
from src.connectome import FlyConnectome, SYM_PAIR, NUM_NEURONS
from src.simulation import FlightSimulation, run_paired_trial, run_statistical_benchmark
from src.arm2_control import (
    generate_randomised_dynamics_control,
    generate_randomised_time_constants,
    generate_sign_permuted_control,
    generate_arm2_extended_control,
    verify_arm2_control,
    verify_arm2_extended_control
)
from src.telemetry import (
    calculate_unclipped_displacement,
    calculate_path_length,
    classify_behavior,
    extract_trajectory_metrics
)

def test_biological_matrix_properties():
    conn = FlyConnectome()
    assert conn.W.shape == (25, 25)
    assert conn.verify_bilateral_symmetry()
    
    # Check ring attractor recurrence
    cycle = [15, 14, 16, 18, 20, 21, 19, 17]
    for idx in range(len(cycle)):
        u = cycle[idx]
        v_next = cycle[(idx + 1) % len(cycle)]
        assert conn.W[u, v_next] == 0.3

def test_maslov_sneppen_degree_conservation():
    conn = FlyConnectome(seed=42)
    for trial in range(10):
        W_shuf = conn.generate_shuffled_control(num_swaps=50, seed=100 + trial)
        assert conn.verify_degrees_conserved(W_shuf), f"Trial {trial} failed degree conservation"

def test_maslov_sneppen_dales_principle():
    conn = FlyConnectome(seed=42)
    for trial in range(5):
        W_shuf = conn.generate_shuffled_control(num_swaps=50, seed=200 + trial)
        for u in range(NUM_NEURONS):
            orig_signs = set(np.sign(conn.W[u, conn.W[u, :] != 0]))
            shuf_signs = set(np.sign(W_shuf[u, W_shuf[u, :] != 0]))
            if len(orig_signs) > 0 and len(shuf_signs) > 0:
                assert orig_signs == shuf_signs, f"Dale's principle violated for neuron {u}"

def test_maslov_sneppen_bilateral_symmetry():
    conn = FlyConnectome(seed=42)
    for trial in range(10):
        W_shuf = conn.generate_shuffled_control(num_swaps=50, seed=300 + trial)
        assert conn.verify_bilateral_symmetry(W_shuf), f"Bilateral symmetry violated in trial {trial}"

def test_paired_trial_reproducibility():
    conn = FlyConnectome(seed=42)
    W_real = conn.W
    W_shuf = conn.generate_shuffled_control(num_swaps=50, seed=777)
    res1 = run_paired_trial(W_real, W_shuf, start_pos=[-25.0, 5.0], start_heading=0.0, max_steps=400)
    assert "ci_real" in res1
    assert "ci_shuf" in res1
    assert "displacement_real" in res1
    assert "displacement_shuf" in res1
    assert "ci_unclipped_real" in res1
    assert "ci_unclipped_shuf" in res1
    assert "path_length_real" in res1
    assert 0.0 <= res1["ci_real"] <= 1.0
    assert 0.0 <= res1["ci_shuf"] <= 1.0

def test_statistical_chemotaxis_discrimination():
    benchmark = run_statistical_benchmark(num_trials=25, max_steps=500, seed=999)
    assert benchmark["num_trials"] == 25
    assert benchmark["mean_diff"] > 0.2
    assert benchmark["t_stat"] > 3.0
    assert benchmark["passed"] is True
    assert "mean_unclipped_real" in benchmark
    assert "mean_disp_real" in benchmark
    assert benchmark["mean_disp_real"] > benchmark["mean_disp_shuf"]

def test_unclipped_telemetry_metrics():
    # 1. Forward movement toward target
    d0, df, disp, ci_uncl = calculate_unclipped_displacement([-35.0, 0.0], [20.0, 0.0], [25.0, 0.0])
    assert np.isclose(d0, 60.0)
    assert np.isclose(df, 5.0)
    assert np.isclose(disp, 55.0)
    assert np.isclose(ci_uncl, 55.0 / 60.0)

    # 2. Divergent movement away from target (wandering)
    d0_w, df_w, disp_w, ci_uncl_w = calculate_unclipped_displacement([-35.0, 0.0], [-95.0, 0.0], [25.0, 0.0])
    assert disp_w < 0.0, "Displacement away from target must be negative"
    assert ci_uncl_w < 0.0, "Unclipped CI away from target must be negative"

    # 3. Path length calculation
    traj = [[0.0, 0.0], [3.0, 4.0], [3.0, 8.0]]  # 5.0 + 4.0 = 9.0
    path_len = calculate_path_length(traj)
    assert np.isclose(path_len, 9.0)

    # 4. Behavioral classification: paralysis vs wandering vs directed
    assert classify_behavior(path_length=2.0, displacement=0.0) == "paralysis"
    assert classify_behavior(path_length=50.0, displacement=-10.0) == "wandering"
    assert classify_behavior(path_length=50.0, displacement=20.0) == "directed"

def test_randomised_time_constants():
    rng = random.Random(101)
    for _ in range(10):
        tau = generate_randomised_time_constants(tau_min=0.02, tau_max=0.08, enforce_symmetry=True, rng=rng)
        assert len(tau) == NUM_NEURONS
        assert np.all(tau >= 0.02)
        assert np.all(tau <= 0.08)
        # Check bilateral symmetry
        for u in range(NUM_NEURONS):
            assert np.isclose(tau[u], tau[SYM_PAIR[u]])

def test_sign_permutations():
    conn = FlyConnectome(seed=42)
    W_bio = conn.W
    rng = random.Random(202)
    for _ in range(10):
        W_perm = generate_sign_permuted_control(W_base=W_bio, permute_mode="shuffle", enforce_symmetry=True, rng=rng)
        # Sparsity identical
        assert np.array_equal(W_bio != 0, W_perm != 0)
        # Bilateral symmetry preserved
        for u in range(NUM_NEURONS):
            for v in range(NUM_NEURONS):
                assert np.isclose(W_perm[u, v], W_perm[SYM_PAIR[u], SYM_PAIR[v]])
        # Nonzero signs permuted
        passed, report = verify_arm2_extended_control(W_bio, W_perm, signs_permuted=True)
        assert passed, f"Extended verification failed:\n{report}"

def test_flight_simulation_with_vectorized_tau():
    conn = FlyConnectome(seed=42)
    tau_vec = np.linspace(0.02, 0.08, NUM_NEURONS, dtype=np.float32)
    sim = FlightSimulation(conn.W, pos=[-35.0, 0.0], heading=0.0, tau=tau_vec)
    for _ in range(100):
        d = sim.step()
    assert len(sim.trajectory) == 101
    assert sim.path_length > 0.0
    telem = sim.compute_telemetry()
    assert "ci_unclipped" in telem
    assert "displacement" in telem
    assert "behavior" in telem

def test_flight_simulation_sensory_crossing():
    conn = FlyConnectome(seed=42)
    sim_normal = FlightSimulation(conn.W, pos=[-35.0, 0.0], heading=0.0, swap_antennae=False)
    sim_crossed = FlightSimulation(conn.W, pos=[-35.0, 0.0], heading=0.0, swap_antennae=True)
    for _ in range(400):
        sim_normal.step()
        sim_crossed.step()
    assert sim_normal.compute_displacement() > sim_crossed.compute_displacement()
    assert sim_crossed.compute_displacement() < 0.0, "Sensory crossing must steer away from plume"
    assert sim_crossed.compute_unclipped_chemotaxis_index() < 0.0

def test_flight_simulation_extreme_tau_stability():
    conn = FlyConnectome(seed=42)
    # Extremely small tau where forward Euler dt / tau = 0.015 / 0.003 = 5.0 (unstable without bounded decay)
    sim = FlightSimulation(conn.W, pos=[-35.0, 0.0], heading=0.0, tau=0.003)
    for _ in range(100):
        d = sim.step()
    assert np.all(np.isfinite(sim.rates))
    assert np.all(sim.rates >= 0.0)
    assert np.all(sim.rates <= 50.0)

def test_sign_permutations_modes():
    conn = FlyConnectome(seed=42)
    W_bio = conn.W
    rng = random.Random(303)
    # Flip mode
    W_flip = generate_sign_permuted_control(W_base=W_bio.tolist(), permute_mode="flip", rng=rng)
    assert np.array_equal(W_bio != 0, W_flip != 0)
    passed, _ = verify_arm2_extended_control(W_bio, W_flip, signs_permuted=True)
    assert passed
    # Extended generator with permute_mode="flip"
    W_ctrl, tau_ctrl = generate_arm2_extended_control(rng=rng, randomize_tau=True, permute_signs=True, permute_mode="flip")
    assert W_ctrl.shape == (25, 25)
    assert tau_ctrl is not None and len(tau_ctrl) == 25
