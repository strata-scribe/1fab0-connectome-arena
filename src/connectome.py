"""
Core Connectome Module (Grant 1fab0)
Models Janelia MaleCNS-derived synaptic wiring and degree-preserving configuration controls.
"""

import math
import random
import numpy as np

NUM_NEURONS = 25

# Bilateral symmetry mapping pairs:
# Left/Right ORNs: 0 <-> 1
# Left/Right PNs: 2 <-> 3, Inhibitory PNs: 4 <-> 5
# LH/MB Kenyon Cells: (6..9) <-> (10..13)
# Central Complex Steering: (14, 16, 18, 20) <-> (15, 17, 19, 21)
# Descending Neurons: 22 <-> 23 (DNa01/DNa02), 24 (Thrust, self-symmetric)
SYM_PAIR = [
    1, 0, 3, 2, 5, 4,
    10, 11, 12, 13, 6, 7, 8, 9,
    15, 14, 17, 16, 19, 18, 21, 20,
    23, 22, 24
]

class FlyConnectome:
    """
    Synaptic connectivity matrix representing the Drosophila olfactory-motor steering circuit:
    ORN -> PN -> LH/MB -> Central Complex (Fan-Shaped Body) -> Descending Neurons (DNa01/02).
    """
    def __init__(self, seed=42):
        self.seed = seed
        self.num_neurons = NUM_NEURONS
        self.W = self._create_biological_matrix()

    def _create_biological_matrix(self):
        W = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float32)
        
        # 1. Antennal Lobe Projection Neurons
        W[0, 2] = 2.0  # ORN_L -> PN_L (excitatory)
        W[1, 3] = 2.0  # ORN_R -> PN_R (excitatory)
        W[0, 5] = 0.9  # ORN_L -> PN_inh_R (contralateral feedforward)
        W[1, 4] = 0.9  # ORN_R -> PN_inh_L
        W[4, 2] = -1.4 # PN_inh_L -> PN_L (GABA/GluCl lateral inhibition)
        W[5, 3] = -1.4 # PN_inh_R -> PN_R

        # 2. Lateral Horn & Mushroom Body Kenyon Cells
        for i in range(6, 10):
            W[2, i] = 1.2
        for i in range(10, 14):
            W[3, i] = 1.2

        # 3. Central Complex Steering (LAL / Fan-Shaped Body)
        W[2, 14] = 1.6   # Left channel -> Central Complex Left
        W[3, 15] = 1.6   # Right channel -> Central Complex Right
        W[14, 22] = 2.2  # Steering Left -> DNa01
        W[15, 23] = 2.2  # Steering Right -> DNa02

        # 4. Upwind Thrust Acceleration
        for i in range(6, 14):
            W[i, 24] = 0.6

        # 5. Central Complex Recurrent Ring Attractor (Strictly bilaterally symmetric cycle)
        cycle = [15, 14, 16, 18, 20, 21, 19, 17]
        for idx in range(len(cycle)):
            u = cycle[idx]
            v_next = cycle[(idx + 1) % len(cycle)]
            v_prev = cycle[(idx - 1 + len(cycle)) % len(cycle)]
            W[u, v_next] = 0.3
            W[u, v_prev] = 0.3

        return W

    def generate_shuffled_control(self, num_swaps=50, seed=None):
        """
        Degree-preserving configuration-model edge rewire (Maslov-Sneppen algorithm).
        Swaps edge pairs (u->v, x->y) => (u->y, x->v) while strictly preserving:
          - Exact in-degree per neuron
          - Exact out-degree per neuron
          - Dale's Principle (excitatory/inhibitory sign per presynaptic node)
          - Bilateral anatomical symmetry across left/right hemispheres
        """
        rng = random.Random(seed) if seed is not None else random
        W_shuf = self.W.copy()
        N = self.num_neurons
        swaps_done = 0

        for _ in range(num_swaps * 10):
            canonical_edges = []
            for u in range(N):
                for v in range(N):
                    if W_shuf[u, v] != 0:
                        u_s = SYM_PAIR[u]
                        v_s = SYM_PAIR[v]
                        if (u < u_s) or (u == u_s and v <= v_s):
                            canonical_edges.append((u, v))

            if len(canonical_edges) < 2:
                break

            idx1, idx2 = rng.sample(range(len(canonical_edges)), 2)
            u, v = canonical_edges[idx1]
            x, y = canonical_edges[idx2]

            if u == x or v == y or u == y or x == v:
                continue

            val1 = W_shuf[u, v]
            val2 = W_shuf[x, y]
            # Enforce Dale's Principle: only swap edges of the identical sign
            if np.sign(val1) != np.sign(val2):
                continue

            u_s, v_s = SYM_PAIR[u], SYM_PAIR[v]
            x_s, y_s = SYM_PAIR[x], SYM_PAIR[y]

            if u_s == x_s or v_s == y_s or u_s == y_s or x_s == v_s:
                continue

            # Prevent multi-edges or overwriting existing connections
            if (W_shuf[u, y] != 0 or W_shuf[x, v] != 0 or
                W_shuf[u_s, y_s] != 0 or W_shuf[x_s, v_s] != 0):
                continue

            sources = {(u, v), (x, y), (u_s, v_s), (x_s, y_s)}
            targets = {(u, y), (x, v), (u_s, y_s), (x_s, v_s)}
            if len(sources) != 4 or len(targets) != 4:
                continue
            if len(sources.intersection(targets)) > 0:
                continue

            # Execute symmetric double swap
            W_shuf[u, v] = 0.0
            W_shuf[x, y] = 0.0
            W_shuf[u_s, v_s] = 0.0
            W_shuf[x_s, y_s] = 0.0

            W_shuf[u, y] = val1
            W_shuf[x, v] = val2
            W_shuf[u_s, y_s] = val1
            W_shuf[x_s, v_s] = val2

            swaps_done += 1
            if swaps_done >= num_swaps:
                break

        return W_shuf

    def get_in_degrees(self, matrix=None):
        m = self.W if matrix is None else matrix
        return np.sum(m != 0, axis=0)

    def get_out_degrees(self, matrix=None):
        m = self.W if matrix is None else matrix
        return np.sum(m != 0, axis=1)

    def verify_degrees_conserved(self, W_other):
        """Asserts exact in-degree and out-degree invariance."""
        in_orig = self.get_in_degrees(self.W)
        in_shuf = self.get_in_degrees(W_other)
        out_orig = self.get_out_degrees(self.W)
        out_shuf = self.get_out_degrees(W_other)
        return np.array_equal(in_orig, in_shuf) and np.array_equal(out_orig, out_shuf)

    def verify_bilateral_symmetry(self, matrix=None):
        """Asserts exact bilateral symmetry: W[u, v] == W[SYM(u), SYM(v)]."""
        m = self.W if matrix is None else matrix
        for u in range(self.num_neurons):
            for v in range(self.num_neurons):
                if not math.isclose(m[u, v], m[SYM_PAIR[u], SYM_PAIR[v]], abs_tol=1e-5):
                    return False
        return True
