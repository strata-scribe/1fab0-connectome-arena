"""
Arm 2 Control Generator (Grant 1fab0 — Second Null & Quire v4 Dynamics Battery)

Generates the randomised-dynamics controls for Proposal #20 Arm 2 and extended v4 benchmarks:
  - Topology: IDENTICAL to G_traced (same sparsity pattern)
  - Dynamics: Weight MAGNITUDES uniformly randomised within each neurotransmitter class
  - Encoder:  IDENTICAL bilateral antenna sensory projection to ORN_L/ORN_R
  - Decoder:  IDENTICAL read-out from DNa01 (22), DNa02 (23), thrust DN (24)
  - Time Constants (v4): Randomized per-neuron membrane time constants tau_i ~ U[tau_min, tau_max]
  - Sign Scrambling (v4): Randomized / permuted synaptic signs to test sensitivity to excitation/inhibition structure

This tests whether biological steering depends on specific synapse-count ratios,
time constants, or E/I sign layout beyond what any dynamical system on G_traced
produces under identical sensorimotor I/O.
"""

import math
import random
from typing import Optional, Tuple
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


def generate_randomised_time_constants(
    tau_min: float = 0.01,
    tau_max: float = 0.10,
    enforce_symmetry: bool = True,
    rng: Optional[random.Random] = None
) -> np.ndarray:
    """
    Generate randomized per-neuron membrane time constants tau_i ~ U[tau_min, tau_max].
    If enforce_symmetry is True, tau[u] == tau[SYM(u)] is enforced.
    
    Returns
    -------
    np.ndarray of shape (25,) float32
    """
    if rng is None:
        rng = random.Random()

    tau = np.zeros(NUM_NEURONS, dtype=np.float32)
    assigned = set()

    for u in range(NUM_NEURONS):
        if enforce_symmetry:
            if u in assigned:
                continue
            u_s = SYM_PAIR[u]
            val = rng.uniform(tau_min, tau_max)
            tau[u] = val
            tau[u_s] = val
            assigned.add(u)
            assigned.add(u_s)
        else:
            tau[u] = rng.uniform(tau_min, tau_max)

    return tau


def generate_sign_permuted_control(
    W_base: Optional[np.ndarray] = None,
    permute_mode: str = "shuffle",
    enforce_symmetry: bool = True,
    rng: Optional[random.Random] = None
) -> np.ndarray:
    """
    Generate a control matrix with identical nonzero topology and magnitude
    distribution, but with synaptic signs permuted or flipped.
    
    Parameters
    ----------
    W_base : np.ndarray or None
        Base matrix to scramble (defaults to W_bio).
    permute_mode : str
        'shuffle': Shuffles the existing signs among canonical edges, exactly preserving
                   the overall E/I count while scrambling their positions.
        'flip': Independently randomizes each canonical edge sign uniformly from {-1, +1}.
    enforce_symmetry : bool
        If True, contralateral partner edges share the same sign.
    rng : random.Random or None
        Seeded RNG.
        
    Returns
    -------
    np.ndarray of shape (25, 25) float32
    """
    if rng is None:
        rng = random.Random()

    if W_base is None:
        connectome = FlyConnectome()
        W_base = connectome.W
    else:
        W_base = np.array(W_base, dtype=np.float32)

    W_out = np.zeros((NUM_NEURONS, NUM_NEURONS), dtype=np.float32)

    canonical_edges = []
    assigned = set()
    for u in range(NUM_NEURONS):
        for v in range(NUM_NEURONS):
            if (u, v) in assigned:
                continue
            if W_base[u, v] == 0.0:
                continue
            u_s = SYM_PAIR[u]
            v_s = SYM_PAIR[v]
            canonical_edges.append((u, v, u_s, v_s))
            assigned.add((u, v))
            assigned.add((u_s, v_s))

    if permute_mode == "shuffle":
        orig_signs = [1.0 if W_base[u, v] > 0 else -1.0 for (u, v, _, _) in canonical_edges]
        signs = orig_signs.copy()
        if len(set(signs)) > 1:
            for _ in range(100):
                rng.shuffle(signs)
                if any(signs[k] != orig_signs[k] for k in range(len(canonical_edges))):
                    break
    else:  # 'flip', 'random', or 'scramble'
        signs = [1.0 if rng.random() < 0.5 else -1.0 for _ in canonical_edges]

    for idx, (u, v, u_s, v_s) in enumerate(canonical_edges):
        mag = abs(float(W_base[u, v]))
        val = mag * signs[idx]
        W_out[u, v] = val
        if enforce_symmetry:
            W_out[u_s, v_s] = val
        else:
            other_sign = 1.0 if rng.random() < 0.5 else -1.0
            W_out[u_s, v_s] = abs(float(W_base[u_s, v_s])) * other_sign

    return W_out


def generate_arm2_extended_control(
    rng: Optional[random.Random] = None,
    randomize_magnitudes: bool = True,
    randomize_tau: bool = False,
    permute_signs: bool = False,
    permute_mode: str = "shuffle",
    tau_min: float = 0.01,
    tau_max: float = 0.10,
    enforce_symmetry: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Unified generator for Arm 2 and v4 extended dynamics controls.
    Returns (W_ctrl, tau_ctrl).
    """
    if rng is None:
        rng = random.Random()

    if randomize_magnitudes:
        W_ctrl = generate_randomised_dynamics_control(rng=rng)
    else:
        connectome = FlyConnectome()
        W_ctrl = connectome.W.copy()

    if permute_signs:
        W_ctrl = generate_sign_permuted_control(
            W_base=W_ctrl,
            permute_mode=permute_mode,
            enforce_symmetry=enforce_symmetry,
            rng=rng
        )

    tau_ctrl = None
    if randomize_tau:
        tau_ctrl = generate_randomised_time_constants(
            tau_min=tau_min,
            tau_max=tau_max,
            enforce_symmetry=enforce_symmetry,
            rng=rng
        )

    return W_ctrl, tau_ctrl


def verify_arm2_control(W_bio, W_rand, verbose=False):
    """
    Verify all standard Arm 2 construction invariants hold.
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


def verify_arm2_extended_control(W_bio, W_ctrl, tau_ctrl=None, signs_permuted=False, verbose=False):
    """
    Verify invariants for extended dynamics controls (allowing sign permutations and custom tau).
    """
    checks = []

    # 1. Sparsity
    sparsity_ok = np.array_equal(W_bio != 0, W_ctrl != 0)
    checks.append(f"[{'PASS' if sparsity_ok else 'FAIL'}] Nonzero topology identical to G_traced")

    # 2. Bilateral symmetry of W
    sym_ok = True
    for u in range(NUM_NEURONS):
        for v in range(NUM_NEURONS):
            if not np.isclose(W_ctrl[u, v], W_ctrl[SYM_PAIR[u], SYM_PAIR[v]], atol=1e-5):
                sym_ok = False
                break
    checks.append(f"[{'PASS' if sym_ok else 'FAIL'}] Bilateral symmetry of weights")

    # 3. Sign check
    if signs_permuted:
        # Check that signs are not identically equal to bio (unless by rare chance), but nonzero pattern intact
        diff_count = sum(
            1 for u in range(NUM_NEURONS) for v in range(NUM_NEURONS)
            if W_bio[u, v] != 0 and np.sign(W_bio[u, v]) != np.sign(W_ctrl[u, v])
        )
        sign_check = diff_count > 0
        checks.append(f"[{'PASS' if sign_check else 'FAIL'}] Synaptic signs permuted ({diff_count} flipped edges)")
    else:
        sign_check = all(
            np.sign(W_bio[u, v]) == np.sign(W_ctrl[u, v])
            for u in range(NUM_NEURONS) for v in range(NUM_NEURONS)
        )
        checks.append(f"[{'PASS' if sign_check else 'FAIL'}] NT-signs preserved")

    # 4. Zero self-connections
    self_ok = all(W_ctrl[i, i] == 0.0 for i in range(NUM_NEURONS))
    checks.append(f"[{'PASS' if self_ok else 'FAIL'}] Zero self-connections")

    # 5. Tau checks
    tau_ok = True
    if tau_ctrl is not None:
        if len(tau_ctrl) != NUM_NEURONS:
            tau_ok = False
        elif np.any(tau_ctrl <= 0.0):
            tau_ok = False
        else:
            # Check bilateral symmetry of tau
            tau_sym_ok = all(
                np.isclose(tau_ctrl[u], tau_ctrl[SYM_PAIR[u]], atol=1e-5)
                for u in range(NUM_NEURONS)
            )
            if not tau_sym_ok:
                tau_ok = False
        checks.append(f"[{'PASS' if tau_ok else 'FAIL'}] Tau vector valid and bilateral-symmetric")

    passed = sparsity_ok and sym_ok and sign_check and self_ok and tau_ok
    report = "\n".join(checks)
    if verbose:
        print(report)

    return passed, report
