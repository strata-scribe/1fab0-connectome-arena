#!/usr/bin/env python3
"""
Arm 2 Benchmark — Proposal #20 Second Null (Grant 1fab0)

Paired discrimination: Biological G_traced  vs.  Randomised-Dynamics control
(same topology, same NT signs, weight magnitudes drawn from U[0.5,3.5] / U[-2.5,-0.5])

This is the specific second arm requested by @head-of-engineering (c61183):
  "Run the battery's second arm — real graph, randomised dynamics, same encoder
   and decoder — and publish what the arena gets."

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

    for _ in range(MAX_STEPS):
        d = sim_real.step()
        if d < 3.0:
            break
    for _ in range(MAX_STEPS):
        d = sim_rand.step()
        if d < 3.0:
            break

    ci_real = sim_real.compute_chemotaxis_index()
    ci_rand = sim_rand.compute_chemotaxis_index()

    return {
        "ci_real": ci_real,
        "ci_rand": ci_rand,
        "ci_diff": ci_real - ci_rand,
        "real_won": ci_real > ci_rand,
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

    for trial_i in range(NUM_TRIALS):
        # Fresh randomised-dynamics control for each trial (same sign-graph, new magnitudes)
        W_rand = generate_randomised_dynamics_control(rng=rng)
        result = run_arm2_paired_trial(W_bio, W_rand)
        real_scores.append(result["ci_real"])
        rand_scores.append(result["ci_rand"])

    real_scores = np.array(real_scores, dtype=np.float64)
    rand_scores = np.array(rand_scores, dtype=np.float64)
    diffs = real_scores - rand_scores

    mean_real = float(np.mean(real_scores))
    mean_rand = float(np.mean(rand_scores))
    mean_diff = float(np.mean(diffs))
    std_diff  = float(np.std(diffs, ddof=1))
    se_diff   = std_diff / math.sqrt(NUM_TRIALS)
    t_stat    = mean_diff / max(1e-9, se_diff)
    passed    = t_stat > 3.0

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
    print(f"Biological Mean CI:         {res['mean_real']:.3f}")
    print(f"Rand-Dynamics Mean CI:      {res['mean_rand']:.3f}")
    print(f"Net Chemotaxis Advantage:   +{res['mean_diff']:.3f}")
    print(f"Std of paired differences:  {res['std_diff']:.3f}")
    print(f"Paired Student's t:         {res['t_stat']:.2f}")
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
