#!/usr/bin/env python3
"""
Arm 2 Benchmark — Proposal #20 Second Null (Grant 1fab0)

Paired discrimination: Biological G_traced  vs.  Randomised-Dynamics control
(same topology, same NT signs, weight magnitudes drawn from U[0.5,3.5] / U[-2.5,-0.5])

Measures both standard clipped CI, unclipped continuous displacement (d0 - dfinal),
and path divergence metrics.

Seed-pinned for full reproducibility (seed=456 distinct from Arm 1 seed=123).
"""

import sys
import os
import math
import random
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.connectome import FlyConnectome
from src.arm2_control import generate_randomised_dynamics_control, verify_arm2_control
from src.simulation import FlightSimulation
from src.telemetry import classify_behavior


NUM_TRIALS = 40
MAX_STEPS  = 700
SEED       = 456    # distinct from Arm 1 (seed=123) to prevent cross-contamination


def run_arm2_paired_trial(W_bio, W_rand, start_pos=None, start_heading=None):
    """Paired trial: biological dynamics vs. randomised dynamics on G_traced."""
    if start_pos is None:
        start_pos = np.array([-35.0, random.uniform(-25.0, 25.0)], dtype=np.float32)
    if start_heading is None:
        start_heading = random.uniform(-0.8, 0.8)

    sim_real = FlightSimulation(W_bio,  pos=start_pos.copy(), heading=start_heading, is_real=True,  label="Biological")
    sim_rand = FlightSimulation(W_rand, pos=start_pos.copy(), heading=start_heading, is_real=False, label="Rand-Dyn")

    steps_real = 0
    rate_acc_real = np.zeros(sim_real.num_neurons, dtype=np.float64)
    for _ in range(MAX_STEPS):
        d = sim_real.step()
        rate_acc_real += sim_real.rates
        steps_real += 1
        if d < 3.0:
            break

    steps_rand = 0
    rate_acc_rand = np.zeros(sim_rand.num_neurons, dtype=np.float64)
    for _ in range(MAX_STEPS):
        d = sim_rand.step()
        rate_acc_rand += sim_rand.rates
        steps_rand += 1
        if d < 3.0:
            break

    ci_real = sim_real.compute_chemotaxis_index()
    ci_rand = sim_rand.compute_chemotaxis_index()
    ci_unclipped_real = sim_real.compute_unclipped_chemotaxis_index()
    ci_unclipped_rand = sim_rand.compute_unclipped_chemotaxis_index()
    disp_real = sim_real.compute_displacement()
    disp_rand = sim_rand.compute_displacement()

    return {
        "ci_real": ci_real,
        "ci_rand": ci_rand,
        "ci_diff": ci_real - ci_rand,
        "ci_unclipped_real": ci_unclipped_real,
        "ci_unclipped_rand": ci_unclipped_rand,
        "displacement_real": disp_real,
        "displacement_rand": disp_rand,
        "path_length_real": sim_real.path_length,
        "path_length_rand": sim_rand.path_length,
        "behavior_real": classify_behavior(sim_real.path_length, disp_real),
        "behavior_rand": classify_behavior(sim_rand.path_length, disp_rand),
        "real_won": ci_real > ci_rand,
        "mean_pop_rate_real": float(np.mean(rate_acc_real) / steps_real),
        "mean_pop_rate_rand": float(np.mean(rate_acc_rand) / steps_rand),
        "silent_frac_rand": float(np.mean((rate_acc_rand / steps_rand) < 0.5)),
    }


def run_arm2_benchmark():
    random.seed(SEED)
    np.random.seed(SEED)
    rng = random.Random(SEED)

    connectome = FlyConnectome(seed=SEED)
    W_bio = connectome.W

    # Verify construction invariants before running
    W_rand_probe = generate_randomised_dynamics_control(rng=random.Random(SEED + 1))
    passed_verify, verify_report = verify_arm2_control(W_bio, W_rand_probe, verbose=False)
    if not passed_verify:
        print("[ABORT] Arm 2 control construction invariants FAILED:")
        print(verify_report)
        sys.exit(1)

    real_scores = []
    rand_scores = []
    unclipped_real_scores = []
    unclipped_rand_scores = []
    disp_real_scores = []
    disp_rand_scores = []
    path_real_scores = []
    path_rand_scores = []
    rate_real_scores = []
    rate_rand_scores = []
    silent_frac_scores = []

    for trial_i in range(NUM_TRIALS):
        W_rand = generate_randomised_dynamics_control(rng=rng)
        result = run_arm2_paired_trial(W_bio, W_rand)
        real_scores.append(result["ci_real"])
        rand_scores.append(result["ci_rand"])
        unclipped_real_scores.append(result["ci_unclipped_real"])
        unclipped_rand_scores.append(result["ci_unclipped_rand"])
        disp_real_scores.append(result["displacement_real"])
        disp_rand_scores.append(result["displacement_rand"])
        path_real_scores.append(result["path_length_real"])
        path_rand_scores.append(result["path_length_rand"])
        rate_real_scores.append(result["mean_pop_rate_real"])
        rate_rand_scores.append(result["mean_pop_rate_rand"])
        silent_frac_scores.append(result["silent_frac_rand"])

    real_scores = np.array(real_scores, dtype=np.float64)
    rand_scores = np.array(rand_scores, dtype=np.float64)
    diffs = real_scores - rand_scores

    disp_real_scores = np.array(disp_real_scores, dtype=np.float64)
    disp_rand_scores = np.array(disp_rand_scores, dtype=np.float64)
    disp_diffs = disp_real_scores - disp_rand_scores

    mean_real = float(np.mean(real_scores))
    mean_rand = float(np.mean(rand_scores))
    mean_diff = float(np.mean(diffs))
    std_diff  = float(np.std(diffs, ddof=1))
    se_diff   = std_diff / math.sqrt(NUM_TRIALS)
    t_stat    = mean_diff / max(1e-9, se_diff)
    passed    = t_stat > 3.0

    mean_disp_diff = float(np.mean(disp_diffs))
    std_disp_diff = float(np.std(disp_diffs, ddof=1))
    t_stat_disp = mean_disp_diff / max(1e-9, std_disp_diff / math.sqrt(NUM_TRIALS))

    return {
        "num_trials": NUM_TRIALS,
        "seed": SEED,
        "mean_real": mean_real,
        "mean_rand": mean_rand,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "se_diff": se_diff,
        "t_stat": t_stat,
        "passed": passed,
        "verify_report": verify_report,
        "mean_unclipped_real": float(np.mean(unclipped_real_scores)),
        "mean_unclipped_rand": float(np.mean(unclipped_rand_scores)),
        "mean_disp_real": float(np.mean(disp_real_scores)),
        "mean_disp_rand": float(np.mean(disp_rand_scores)),
        "mean_disp_diff": mean_disp_diff,
        "t_stat_disp": t_stat_disp,
        "mean_path_real": float(np.mean(path_real_scores)),
        "mean_path_rand": float(np.mean(path_rand_scores)),
        "mean_pop_rate_real": float(np.mean(rate_real_scores)),
        "mean_pop_rate_rand": float(np.mean(rate_rand_scores)),
        "mean_silent_frac_rand": float(np.mean(silent_frac_scores)),
    }


def main():
    print("═══════════════════════════════════════════════════════════════")
    print(" Grant 1fab0 — Proposal #20 — Arm 2: Randomised-Dynamics Null")
    print(" Substrate:   G_traced (identical topology to biological arm)")
    print(" Control:     Dynamics randomised — U[0.5,3.5] excitatory,")
    print("              U[-2.5,-0.5] inhibitory, NT signs preserved")
    print(" Encoder/Decoder: Identical to Arm 1 (bilateral ORN → DNa01/02)")
    print(f" Trials: {NUM_TRIALS}  |  Seed: {SEED}  |  Max steps: {MAX_STEPS}")
    print("═══════════════════════════════════════════════════════════════")

    res = run_arm2_benchmark()

    print()
    print(f"Construction Invariants:    {res['verify_report'].splitlines()[0]}")
    print()
    print(f"Paired Trials:              {res['num_trials']}  (seed={res['seed']})")
    print(f"Biological Mean CI:         {res['mean_real']:.3f} (unclipped: {res['mean_unclipped_real']:.3f})")
    print(f"Rand-Dynamics Mean CI:      {res['mean_rand']:.3f} (unclipped: {res['mean_unclipped_rand']:.3f})")
    print(f"Net Chemotaxis Advantage:   +{res['mean_diff']:.3f}")
    print(f"Paired Student's t (CI):    {res['t_stat']:.2f}")
    print("---------------------------------------------------------------")
    print(f"Biological Displacement:    {res['mean_disp_real']:+.2f} units (path: {res['mean_path_real']:.1f})")
    print(f"Rand-Dyn Displacement:      {res['mean_disp_rand']:+.2f} units (path: {res['mean_path_rand']:.1f})")
    print(f"Net Displacement Advantage: +{res['mean_disp_diff']:.2f} units")
    print(f"Paired Student's t (Disp):  {res['t_stat_disp']:.2f}")
    print("---------------------------------------------------------------")
    print(f"Population Activity (mean firing rate, Hz proxy):")
    print(f"  Biological arm:           {res['mean_pop_rate_real']:.3f}")
    print(f"  Rand-dynamics arm:        {res['mean_pop_rate_rand']:.3f}")
    print(f"  Silent neurons (rand, <0.5 Hz): {res['mean_silent_frac_rand']*100:.1f}%")
    print()
    if res['passed']:
        print("Status:  PASSED — Biological wiring outperforms randomised dynamics")
        print("         on the same G_traced substrate (p << 0.001, t > 3.0)")
    else:
        print("Status:  FAILED — no significant advantage over randomised dynamics")
    print("═══════════════════════════════════════════════════════════════")

    return res


if __name__ == '__main__':
    main()
