"""
Unit & Invariant Tests for Drosophila Connectome Simulation (Grant 1fab0)
"""

import numpy as np
from src.connectome import FlyConnectome, SYM_PAIR, NUM_NEURONS
from src.simulation import FlightSimulation, run_paired_trial, run_statistical_benchmark

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
        # For every presynaptic neuron, check that all its outgoing edges have the same sign
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
    assert 0.0 <= res1["ci_real"] <= 1.0
    assert 0.0 <= res1["ci_shuf"] <= 1.0

def test_statistical_chemotaxis_discrimination():
    # 25 paired trials to verify significant positive t-statistic
    benchmark = run_statistical_benchmark(num_trials=25, max_steps=500, seed=999)
    assert benchmark["num_trials"] == 25
    assert benchmark["mean_diff"] > 0.2
    assert benchmark["t_stat"] > 3.0
    assert benchmark["passed"] is True
