"""
Arm 2 Control Generator (Grant 1fab0 — Second Null)

Generates the randomised-dynamics control for Proposal #20 Arm 2:
  - Topology: IDENTICAL to G_traced (same sparsity pattern, same NT signs)
  - Dynamics:  Weight MAGNITUDES uniformly randomised within each neurotransmitter class
  - Encoder:   IDENTICAL bilateral antenna sensory projection to ORN_L/ORN_R
  - Decoder:   IDENTICAL read-out from DNa01 (22), DNa02 (23), thrust DN (24)

This tests whether the biological weight *values* carry information beyond what any
nonlinear system of the same sign-graph topology would produce under identical
sensorimotor I/O.

Construction invariants preserved:
  - sign(W_rand[u,v]) == sign(W_bio[u,v])  for all (u,v)
  - Bilateral symmetry: W_rand[u,v] == W_rand[SYM(u), SYM(v)]
  - Excitatory range: U[0.5, 3.5]  (bio range: 0.6 – 2.2)
  - Inhibitory range: U[-2.5, -0.5] (bio range: -1.4)
  - All diagonal self-connections remain zero
"""

import random
import numpy as np
from src.connectome import FlyConnectome, SYM_PAIR

NUM_NEURONS = 25

# Weight sampling bounds (symmetric around biological magnitudes)
_EXC_LO = 0.5
_EXC_HI = 3.5
_INH_LO = -2.5
_INH_HI = -0.5


def generate_randomised_dynamics_control(rng=None):
    """
    Returns W_rand: a 25×25 numpy float32 matrix with the same topology as
    W_bio but all nonzero weight magnitudes independently randomised within
    NT-sign bounds, with bilateral symmetry enforced.

    Parameters
    ----------
    rng : random.Random or None
        Optional seeded RNG for reproducibility.
    """
    if rng is None:
        rng = random.Random()

    connectome = FlyConnectome()
    W_bio = connectome.W  # (25, 25) float32

    W_rand = np.zeros((NUM_NEURONS, NUM_NEURONS), dtype=np.float32)

    # Iterate over canonical (left-hemisphere) half of symmetric pairs only,
    # then mirror, to guarantee exact bilateral symmetry.
    assigned = set()
    for u in range(NUM_NEURONS):
        for v in range(NUM_NEURONS):
            if (u, v) in assigned:
                continue
            if W_bio[u, v] == 0.0:
                continue

            u_s = SYM_PAIR[u]
            v_s = SYM_PAIR[v]

            # Draw one random magnitude for this canonical edge pair
            if W_bio[u, v] > 0.0:
                mag = rng.uniform(_EXC_LO, _EXC_HI)
            else:
                mag = rng.uniform(_INH_LO, _INH_HI)

            W_rand[u, v] = mag
            W_rand[u_s, v_s] = mag  # mirror to contralateral

            assigned.add((u, v))
            assigned.add((u_s, v_s))

    return W_rand


def verify_arm2_control(W_bio, W_rand, verbose=False):
    """
    Verify all Arm 2 construction invariants hold.
    Returns (passed: bool, report: str).
    """
    checks = []

    # 1. Sign preservation
    sign_ok = True
    for u in range(NUM_NEURONS):
        for v in range(NUM_NEURONS):
            bio_sign = np.sign(W_bio[u, v])
            rand_sign = np.sign(W_rand[u, v])
            if bio_sign != rand_sign:
                sign_ok = False
                checks.append(f"  FAIL sign[{u},{v}]: bio={W_bio[u,v]:.3f} rand={W_rand[u,v]:.3f}")
    checks.insert(0, f"[{'PASS' if sign_ok else 'FAIL'}] NT-sign preservation across all edges")

    # 2. Sparsity (nonzero pattern identical)
    sparsity_ok = np.array_equal(W_bio != 0, W_rand != 0)
    checks.append(f"[{'PASS' if sparsity_ok else 'FAIL'}] Sparsity pattern (nonzero topology) identical to G_traced")

    # 3. Bilateral symmetry
    sym_ok = True
    for u in range(NUM_NEURONS):
        for v in range(NUM_NEURONS):
            if not np.isclose(W_rand[u, v], W_rand[SYM_PAIR[u], SYM_PAIR[v]], atol=1e-5):
                sym_ok = False
                break
    checks.append(f"[{'PASS' if sym_ok else 'FAIL'}] Bilateral symmetry W_rand[u,v] == W_rand[SYM(u),SYM(v)]")

    # 4. Weight magnitude ranges
    exc_vals = W_rand[W_rand > 0]
    inh_vals = W_rand[W_rand < 0]
    range_ok = True
    if len(exc_vals) > 0:
        if exc_vals.min() < _EXC_LO - 1e-5 or exc_vals.max() > _EXC_HI + 1e-5:
            range_ok = False
    if len(inh_vals) > 0:
        if inh_vals.max() > _INH_HI + 1e-5 or inh_vals.min() < _INH_LO - 1e-5:
            range_ok = False
    checks.append(f"[{'PASS' if range_ok else 'FAIL'}] Excitatory U[{_EXC_LO},{_EXC_HI}], Inhibitory U[{_INH_LO},{_INH_HI}]")

    # 5. Self-connections absent
    self_ok = all(W_rand[i, i] == 0.0 for i in range(NUM_NEURONS))
    checks.append(f"[{'PASS' if self_ok else 'FAIL'}] Zero self-connections")

    passed = sign_ok and sparsity_ok and sym_ok and range_ok and self_ok
    report = "\n".join(checks)

    if verbose:
        print(report)

    return passed, report
