"""
Flight Simulation & Statistical Chemotaxis Harness (Grant 1fab0)
Executes continuous neural ODE trajectory integration and paired A/B discrimination trials.
"""

import math
import random
import numpy as np
from src.connectome import FlyConnectome

class FlightSimulation:
    """
    Continuous agent physics and neural dynamics in an odor-plume environment.
    All flies share 100% identical kinematics, ODE parameters, and unbiased exploratory casting.
    """
    def __init__(self, W, pos=None, heading=None, target_pos=None, is_real=True, label="Fly"):
        self.W = np.array(W, dtype=np.float32)
        self.is_real = is_real
        self.label = label
        self.num_neurons = self.W.shape[0]
        
        # Neural state vector (firing rates in Hz)
        self.rates = np.zeros(self.num_neurons, dtype=np.float32)
        self.tau = 0.05   # 50 ms membrane time constant
        self.dt = 0.015   # 15 ms integration timestep
        
        # Environment geometry: Target odor at (25.0, 0.0)
        self.target_pos = np.array([25.0, 0.0], dtype=np.float32) if target_pos is None else np.array(target_pos, dtype=np.float32)
        self.pos = np.array([-35.0, random.uniform(-25.0, 25.0)], dtype=np.float32) if pos is None else np.array(pos, dtype=np.float32)
        self.start_pos = self.pos.copy()
        self.heading = random.uniform(-0.8, 0.8) if heading is None else float(heading)
        
        # Physical morphology
        self.antenna_dist = 1.2
        self.wander_phase = random.uniform(0, 10)
        self.min_dist = float(np.linalg.norm(self.pos - self.target_pos))
        self.history = []

    def step(self):
        """
        Execute one continuous timestep of sensory encoding, neural integration, and motor kinematics.
        """
        # 1. Bilateral Antennae Coordinate Sampling
        ant_l = self.pos + self.antenna_dist * np.array([-math.sin(self.heading), math.cos(self.heading)], dtype=np.float32)
        ant_r = self.pos + self.antenna_dist * np.array([math.sin(self.heading), -math.cos(self.heading)], dtype=np.float32)

        d_l = float(np.linalg.norm(ant_l - self.target_pos))
        d_r = float(np.linalg.norm(ant_r - self.target_pos))

        # 2. Odor Concentration Plume (Gaussian decay + mild stochastic intermittency)
        c_l = max(0.0, 20.0 / (1.0 + 0.05 * d_l) + random.gauss(0, 0.02))
        c_r = max(0.0, 20.0 / (1.0 + 0.05 * d_r) + random.gauss(0, 0.02))

        # 3. External Sensory Current Injection to ORNs
        I_ext = np.zeros(self.num_neurons, dtype=np.float32)
        I_ext[0] = c_l
        I_ext[1] = c_r

        # 4. Continuous Leaky Integrate Firing Rate ODE:
        # tau * dr/dt = -r + ReLU(W^T * r + I_ext)
        synaptic_input = np.dot(self.rates, self.W) + I_ext
        dr = (-self.rates + np.maximum(0.0, synaptic_input)) / self.tau * self.dt
        self.rates = np.clip(self.rates + dr, 0.0, 50.0)

        # 5. Motor Decoding from Descending Neurons
        turn_left = float(self.rates[22])
        turn_right = float(self.rates[23])

        # Unbiased exploratory casting baseline (strictly symmetric across all agents)
        self.wander_phase += 0.06
        casting_torque = math.sin(self.wander_phase) * 0.8 + random.gauss(0, 0.2)
        yaw_rate = (turn_left - turn_right) * 1.8 + casting_torque

        self.heading += yaw_rate * self.dt

        # 6. Forward Kinematics with Thrust Modulation (neuron 24)
        speed = 8.0 + 0.4 * float(self.rates[24])
        velocity = speed * np.array([math.cos(self.heading), math.sin(self.heading)], dtype=np.float32)
        self.pos += velocity * self.dt

        dist = float(np.linalg.norm(self.pos - self.target_pos))
        if dist < self.min_dist:
            self.min_dist = dist

        return dist

    def compute_chemotaxis_index(self):
        d_init = float(np.linalg.norm(self.start_pos - self.target_pos))
        d_final = float(np.linalg.norm(self.pos - self.target_pos))
        if d_init < 1e-5:
            return 0.0
        return max(0.0, (d_init - d_final) / d_init)


def run_paired_trial(W_real, W_shuf, start_pos=None, start_heading=None, max_steps=700):
    if start_pos is None:
        start_pos = np.array([-35.0, random.uniform(-25.0, 25.0)], dtype=np.float32)
    if start_heading is None:
        start_heading = random.uniform(-0.8, 0.8)

    sim_real = FlightSimulation(W_real, pos=start_pos.copy(), heading=start_heading, is_real=True, label="Biological")
    sim_shuf = FlightSimulation(W_shuf, pos=start_pos.copy(), heading=start_heading, is_real=False, label="Shuffled")

    for _ in range(max_steps):
        d_real = sim_real.step()
        if d_real < 3.0:
            break

    for _ in range(max_steps):
        d_shuf = sim_shuf.step()
        if d_shuf < 3.0:
            break

    ci_real = sim_real.compute_chemotaxis_index()
    ci_shuf = sim_shuf.compute_chemotaxis_index()

    return {
        "ci_real": ci_real,
        "ci_shuf": ci_shuf,
        "ci_diff": ci_real - ci_shuf,
        "real_won": ci_real > ci_shuf,
        "shuf_won": ci_shuf > ci_real,
        "min_dist_real": sim_real.min_dist,
        "min_dist_shuf": sim_shuf.min_dist
    }


def run_statistical_benchmark(num_trials=40, max_steps=700, seed=123):
    random.seed(seed)
    np.random.seed(seed)
    connectome = FlyConnectome(seed=seed)
    W_real = connectome.W

    real_scores = []
    shuf_scores = []

    for _ in range(num_trials):
        W_shuf = connectome.generate_shuffled_control(num_swaps=60)
        res = run_paired_trial(W_real, W_shuf, max_steps=max_steps)
        real_scores.append(res["ci_real"])
        shuf_scores.append(res["ci_shuf"])

    mean_real = float(np.mean(real_scores))
    mean_shuf = float(np.mean(shuf_scores))
    diffs = np.array(real_scores) - np.array(shuf_scores)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    se_diff = std_diff / math.sqrt(num_trials)
    t_stat = mean_diff / max(1e-6, se_diff)

    return {
        "num_trials": num_trials,
        "mean_real": mean_real,
        "mean_shuf": mean_shuf,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_stat": t_stat,
        "passed": t_stat > 3.0
    }
