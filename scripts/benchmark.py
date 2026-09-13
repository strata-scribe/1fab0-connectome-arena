#!/usr/bin/env python3
"""
CLI Benchmark Runner for Drosophila Connectome Discrimination (Grant 1fab0)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulation import run_statistical_benchmark

def main():
    print("===============================================================")
    print(" Grant 1fab0: Janelia MaleCNS Chemotaxis vs. Shuffled Control")
    print(" Substrate Graph:   G_traced = (V_traced, E_traced ∩ (V_traced × V_traced))")
    print(" Truncation Arc:    ORN (0.669) -> ALPN (0.442) -> KC (0.843) -> DN (0.505)")
    print(" Degree Invariance: k_i = deg_G_traced(i) preserved under Maslov-Sneppen")
    print("===============================================================")
    res = run_statistical_benchmark(num_trials=40, max_steps=700, seed=123)
    print(f"Paired Trials:            {res['num_trials']}")
    print(f"Biological Mean CI:       {res['mean_real']:.3f}")
    print(f"Shuffled Control Mean CI: {res['mean_shuf']:.3f}")
    print(f"Net Chemotaxis Advantage: +{res['mean_diff']:.3f}")
    print(f"Paired Student's t:       {res['t_stat']:.2f}")
    print(f"Status:                   {'PASSED (Statistically Outperforms Scrambled Twin)' if res['passed'] else 'FAILED'}")
    print("===============================================================")

if __name__ == '__main__':
    main()
