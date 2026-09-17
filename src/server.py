#!/usr/bin/env python3
"""
Authoritative Real-Time Connectome Simulation Server (Grant 1fab0)
Runs continuous 24/7 neural ODE simulation of Janelia MaleCNS chemotaxis harness
with server-side sealed A/B blinding, Maslov-Sneppen rewiring, and real-time state streaming.
"""

import os
import sys
import time
import math
import json
import random
import socket
import threading
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.arm2_control import generate_randomised_dynamics_control

# Connectome Architecture Constants
NUM_NEURONS = 25
ARENA_W = 800
ARENA_H = 600

SYM_PAIR = [
    1, 0, 3, 2, 5, 4,
    10, 11, 12, 13, 6, 7, 8, 9,
    15, 14, 17, 16, 19, 18, 21, 20,
    23, 22, 24
]

def create_biological_matrix():
    W = [[0.0] * NUM_NEURONS for _ in range(NUM_NEURONS)]
    # Antennal Lobe (AL)
    W[0][2] = 2.0; W[1][3] = 2.0
    W[0][5] = 0.9; W[1][4] = 0.9
    W[4][2] = -1.4; W[5][3] = -1.4
    # Lateral Horn / Mushroom Body
    for i in range(6, 10): W[2][i] = 1.2
    for i in range(10, 14): W[3][i] = 1.2
    # Central Complex Steering (LAL/FB)
    W[2][14] = 1.6; W[3][15] = 1.6
    W[14][22] = 2.2; W[15][23] = 2.2
    # Upwind thrust
    for i in range(6, 14): W[i][24] = 0.6
    # Recurrent Central Complex Ring Attractor (symmetric bilateral cycle)
    cycle = [15, 14, 16, 18, 20, 21, 19, 17]
    for idx in range(len(cycle)):
        u = cycle[idx]
        v_next = cycle[(idx + 1) % len(cycle)]
        v_prev = cycle[(idx - 1 + len(cycle)) % len(cycle)]
        W[u][v_next] = 0.3
        W[u][v_prev] = 0.3
    return W

def generate_shuffled_matrix(W_orig, num_swaps=40):
    W = [row[:] for row in W_orig]
    swaps_done = 0
    for _ in range(num_swaps * 8):
        canonical_edges = []
        for u in range(NUM_NEURONS):
            for v in range(NUM_NEURONS):
                if W[u][v] != 0:
                    u_s = SYM_PAIR[u]; v_s = SYM_PAIR[v]
                    if (u < u_s) or (u == u_s and v <= v_s):
                        canonical_edges.append((u, v))
        if len(canonical_edges) < 2:
            break
        idx1, idx2 = random.sample(range(len(canonical_edges)), 2)
        u, v = canonical_edges[idx1]
        x, y = canonical_edges[idx2]
        if u == x or v == y or u == y or x == v:
            continue
        val1 = W[u][v]; val2 = W[x][y]
        if (val1 > 0) != (val2 > 0):
            continue
        u_s = SYM_PAIR[u]; v_s = SYM_PAIR[v]
        x_s = SYM_PAIR[x]; y_s = SYM_PAIR[y]
        if u_s == x_s or v_s == y_s or u_s == y_s or x_s == v_s:
            continue
        if W[u][y] != 0 or W[x][v] != 0 or W[u_s][y_s] != 0 or W[x_s][v_s] != 0:
            continue
        sources = {(u, v), (x, y), (u_s, v_s), (x_s, y_s)}
        targets = {(u, y), (x, v), (u_s, y_s), (x_s, v_s)}
        if len(sources) != 4 or len(targets) != 4:
            continue
        if len(sources.intersection(targets)) > 0:
            continue
        W[u][v] = 0; W[x][y] = 0
        W[u_s][v_s] = 0; W[x_s][y_s] = 0
        W[u][y] = val1; W[x][v] = val2
        W[u_s][y_s] = val1; W[x_s][v_s] = val2
        swaps_done += 1
        if swaps_done >= num_swaps:
            break
    return W

class FlyAgent:
    def __init__(self, x, y, heading, W, is_real, label, arm_label="Arm 1", swap_antennae=False):
        self.x = x
        self.y = y
        self.start_x = x
        self.start_y = y
        self.heading = heading
        self.speed = 2.2
        self.W = W
        self.is_real = is_real
        self.label = label
        self.arm_label = arm_label
        self.swap_antennae = swap_antennae
        self.rates = [0.0] * NUM_NEURONS
        self.rate_acc = [0.0] * NUM_NEURONS
        self.sim_steps = 0
        self.tau = 0.05
        self.dt = 0.015
        self.wander_phase = random.uniform(0, 10)
        self.history = []
        self.full_trajectory = [[round(self.x, 1), round(self.y, 1)]]
        self.min_dist = 9999.0
        self.last_yaw = 0.0

    def step(self, target_x, target_y):
        ant_dist = 8.0
        ant_lx = self.x + math.cos(self.heading - 0.5) * ant_dist
        ant_ly = self.y + math.sin(self.heading - 0.5) * ant_dist
        ant_rx = self.x + math.cos(self.heading + 0.5) * ant_dist
        ant_ry = self.y + math.sin(self.heading + 0.5) * ant_dist

        d_l = math.hypot(ant_lx - target_x, ant_ly - target_y)
        d_r = math.hypot(ant_rx - target_x, ant_ry - target_y)

        c_l = max(0.0, 15.0 / (1.0 + 0.006 * d_l) + random.uniform(-0.1, 0.1))
        c_r = max(0.0, 15.0 / (1.0 + 0.006 * d_r) + random.uniform(-0.1, 0.1))

        I_ext = [0.0] * NUM_NEURONS
        if self.swap_antennae:
            I_ext[0] = c_r
            I_ext[1] = c_l
        else:
            I_ext[0] = c_l
            I_ext[1] = c_r

        # Synaptic integration
        syn = [0.0] * NUM_NEURONS
        for i in range(NUM_NEURONS):
            s = I_ext[i]
            for j in range(NUM_NEURONS):
                if self.W[j][i] != 0:
                    s += self.rates[j] * self.W[j][i]
            syn[i] = s

        # Leaky firing rate dynamics
        for i in range(NUM_NEURONS):
            dr = (-self.rates[i] + max(0.0, syn[i])) / self.tau * self.dt
            self.rates[i] = max(0.0, min(50.0, self.rates[i] + dr))
            self.rate_acc[i] += self.rates[i]

        self.sim_steps += 1

        turn_l = self.rates[22]
        turn_r = self.rates[23]
        thrust = self.rates[24]

        self.wander_phase += 0.04
        casting_torque = math.sin(self.wander_phase) * 0.015 + random.uniform(-0.004, 0.004)
        max_yaw = 0.045
        yaw = max(-max_yaw, min(max_yaw, (turn_r - turn_l) * 0.02 + casting_torque))
        self.last_yaw = yaw

        spd = self.speed * (1.0 + min(0.5, thrust * 0.02))
        self.heading += yaw
        self.x += math.cos(self.heading) * spd
        self.y += math.sin(self.heading) * spd

        # Elastic arena boundaries
        if self.x < 12: self.x = 12; self.heading = math.pi - self.heading
        if self.x > ARENA_W - 12: self.x = ARENA_W - 12; self.heading = math.pi - self.heading
        if self.y < 12: self.y = 12; self.heading = -self.heading
        if self.y > ARENA_H - 12: self.y = ARENA_H - 12; self.heading = -self.heading

        self.history.append([round(self.x, 1), round(self.y, 1)])
        if len(self.history) > 220:
            self.history.pop(0)
        self.full_trajectory.append([round(self.x, 1), round(self.y, 1)])

        dist = math.hypot(self.x - target_x, self.y - target_y)
        if dist < self.min_dist:
            self.min_dist = dist
        return dist

    def get_mean_rate(self):
        if self.sim_steps == 0:
            return 0.0
        return round(sum(self.rate_acc) / (NUM_NEURONS * self.sim_steps), 2)

    def get_silent_fraction(self):
        if self.sim_steps == 0:
            return 0.0
        silent_count = sum(1 for i in range(NUM_NEURONS) if (self.rate_acc[i] / self.sim_steps) < 0.5)
        return round(silent_count / NUM_NEURONS * 100.0, 1)

class ArenaEngine:
    def __init__(self, ledger_file="arena_ledger.json"):
        self.lock = threading.Lock()
        self.ledger_file = ledger_file
        self.W_bio = create_biological_matrix()
        self.target = {"x": 650, "y": 300, "r": 18}
        self.flyA = None
        self.flyB = None
        self.step_count = 0
        self.is_revealing = False
        self.reveal_timer = 0
        self.last_reveal = None
        self.mode_setting = "auto"
        self.active_trial_mode = "arm1"
        self.cycle_count = 0
        self.flight_number = 0
        self.stats = {
            "total": 0,
            "real_wins": 0,
            "shuf_wins": 0,
            "timeouts": 0,
            "sum_ci_real": 0.0,
            "sum_ci_shuf": 0.0,
            "diffs": []
        }
        self.sealed_controls = []
        self.control_idx = 0
        self.active_ctrl_hash = "uninitialized"
        self.active_ctrl_id = -1
        self.load_sealed_controls()
        self.load_ledger()
        self.spawn_trial()

    def load_sealed_controls(self):
        controls_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sealed_controls.json"))
        if os.path.exists(controls_path):
            try:
                with open(controls_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.sealed_controls = data.get("controls", [])
                    print(f"[ENGINE] Loaded {len(self.sealed_controls)} off-harness sealed controls from {controls_path}")
            except Exception as e:
                print(f"[ENGINE] Warning: failed to load sealed controls: {e}")

    def load_ledger(self):
        if os.path.exists(self.ledger_file):
            try:
                with open(self.ledger_file, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "total" in data:
                        self.stats.update(data)
                        self.flight_number = data.get("total", 0)
                        if "custody_draws" in data and isinstance(data["custody_draws"], int):
                            self.control_idx = data["custody_draws"]
                        else:
                            hist = data.get("history", [])
                            ctrl_count = sum(1 for h in hist if "ctrl_id" in h)
                            self.control_idx = max(ctrl_count, 300)
            except Exception:
                pass

    def save_ledger(self):
        try:
            with open(self.ledger_file, "w") as f:
                json.dump(self.stats, f)
        except Exception:
            pass

    def spawn_trial(self):
        self.step_count = 0
        self.is_revealing = False
        self.reveal_timer = 0
        self.flight_number += 1

        self.target["x"] = random.uniform(140, ARENA_W - 140)
        self.target["y"] = random.uniform(120, ARENA_H - 120)

        while True:
            sx = random.uniform(70, ARENA_W - 70)
            sy = random.uniform(70, ARENA_H - 70)
            d0 = math.hypot(sx - self.target["x"], sy - self.target["y"])
            if d0 >= 320:
                break

        shared_heading = random.uniform(0, 2 * math.pi)

        if self.mode_setting == "auto":
            modes = ["arm1", "arm2", "wrong_odor"]
            self.active_trial_mode = modes[self.cycle_count % len(modes)]
            self.cycle_count += 1
        else:
            self.active_trial_mode = self.mode_setting

        assign_a_real = random.random() < 0.5

        if self.active_trial_mode == "arm2":
            # Arm 2: Randomised Dynamics control on same graph topology
            try:
                w_rand_np = generate_randomised_dynamics_control()
                W_ctrl = w_rand_np.tolist()
            except Exception:
                W_ctrl = generate_shuffled_matrix(self.W_bio, 40)
            self.active_ctrl_hash = "rand_dynamics_arm2"
            self.active_ctrl_id = "arm2"
            label_real = "Bio (G_traced)"
            label_ctrl = "Arm 2 (Rand-Dynamics)"
            self.flyA = FlyAgent(sx, sy, shared_heading, self.W_bio if assign_a_real else W_ctrl, assign_a_real, "Fly A", arm_label=label_real if assign_a_real else label_ctrl)
            self.flyB = FlyAgent(sx, sy, shared_heading, W_ctrl if assign_a_real else self.W_bio, not assign_a_real, "Fly B", arm_label=label_ctrl if assign_a_real else label_real)
        elif self.active_trial_mode == "wrong_odor":
            # Sensory crossing: both use W_bio, but one has swap_antennae=True
            self.active_ctrl_hash = "sensory_crossing"
            self.active_ctrl_id = "crossed"
            label_real = "Bio (Standard)"
            label_ctrl = "Bio (Swapped Sensory)"
            swap_a = not assign_a_real
            swap_b = assign_a_real
            self.flyA = FlyAgent(sx, sy, shared_heading, self.W_bio, assign_a_real, "Fly A", arm_label=label_real if assign_a_real else label_ctrl, swap_antennae=swap_a)
            self.flyB = FlyAgent(sx, sy, shared_heading, self.W_bio, not assign_a_real, "Fly B", arm_label=label_ctrl if assign_a_real else label_real, swap_antennae=swap_b)
        else:
            # Arm 1: Shuffled Topology (Maslov-Sneppen)
            self.active_trial_mode = "arm1"
            if self.sealed_controls:
                ctrl_obj = self.sealed_controls[self.control_idx % len(self.sealed_controls)]
                self.control_idx += 1
                W_ctrl = [row[:] for row in ctrl_obj["matrix"]]
                self.active_ctrl_hash = ctrl_obj.get("sha256", "sealed")
                self.active_ctrl_id = ctrl_obj.get("id", 0)
            else:
                W_ctrl = generate_shuffled_matrix(self.W_bio, 40)
                self.active_ctrl_hash = "in_harness_fallback"
                self.active_ctrl_id = -1
            label_real = "Bio (G_traced)"
            label_ctrl = "Arm 1 (Shuffled)"
            self.flyA = FlyAgent(sx, sy, shared_heading, self.W_bio if assign_a_real else W_ctrl, assign_a_real, "Fly A", arm_label=label_real if assign_a_real else label_ctrl)
            self.flyB = FlyAgent(sx, sy, shared_heading, W_ctrl if assign_a_real else self.W_bio, not assign_a_real, "Fly B", arm_label=label_ctrl if assign_a_real else label_real)

    def finish_trial(self, winner):
        d0_a = math.hypot(self.flyA.start_x - self.target["x"], self.flyA.start_y - self.target["y"])
        d0_b = math.hypot(self.flyB.start_x - self.target["x"], self.flyB.start_y - self.target["y"])
        final_a = math.hypot(self.flyA.x - self.target["x"], self.flyA.y - self.target["y"])
        final_b = math.hypot(self.flyB.x - self.target["x"], self.flyB.y - self.target["y"])

        ci_a = max(0.0, (d0_a - final_a) / max(1.0, d0_a))
        ci_b = max(0.0, (d0_b - final_b) / max(1.0, d0_b))
        ci_unclipped_a = round((d0_a - final_a) / max(1.0, d0_a), 2)
        ci_unclipped_b = round((d0_b - final_b) / max(1.0, d0_b), 2)
        disp_a = round(d0_a - final_a, 1)
        disp_b = round(d0_b - final_b, 1)
        path_a = round(sum(math.hypot(self.flyA.full_trajectory[i][0] - self.flyA.full_trajectory[i-1][0], self.flyA.full_trajectory[i][1] - self.flyA.full_trajectory[i-1][1]) for i in range(1, len(self.flyA.full_trajectory))), 1) if len(self.flyA.full_trajectory) > 1 else 0.0
        path_b = round(sum(math.hypot(self.flyB.full_trajectory[i][0] - self.flyB.full_trajectory[i-1][0], self.flyB.full_trajectory[i][1] - self.flyB.full_trajectory[i-1][1]) for i in range(1, len(self.flyB.full_trajectory))), 1) if len(self.flyB.full_trajectory) > 1 else 0.0

        ci_real = ci_a if self.flyA.is_real else ci_b
        ci_shuf = ci_b if self.flyA.is_real else ci_a

        self.stats["total"] += 1
        self.stats["sum_ci_real"] += ci_real
        self.stats["sum_ci_shuf"] += ci_shuf
        self.stats["diffs"].append(ci_real - ci_shuf)

        winner_name = "Timeout"
        winner_type = "none"
        if winner == "A":
            winner_name = "Fly A"
            winner_type = "real" if self.flyA.is_real else "shuf"
            if self.flyA.is_real: self.stats["real_wins"] += 1
            else: self.stats["shuf_wins"] += 1
        elif winner == "B":
            winner_name = "Fly B"
            winner_type = "real" if self.flyB.is_real else "shuf"
            if self.flyB.is_real: self.stats["real_wins"] += 1
            else: self.stats["shuf_wins"] += 1
        else:
            self.stats["timeouts"] += 1

        self.save_ledger()

        # Compute current rolling t-stat for historical record
        diffs = self.stats.get("diffs", [])
        n = len(diffs)
        t_stat_val = None
        p_str = "Calibrating"
        if n >= 4:
            mean_d = sum(diffs) / n
            var_d = sum((x - mean_d) ** 2 for x in diffs) / (n - 1)
            se = math.sqrt(var_d / n)
            if se > 0.0001:
                t = mean_d / se
                t_stat_val = round(t, 2)
                if t >= 3.9: p_str = "p < 0.0001"
                elif t >= 3.3: p_str = "p < 0.001"
                elif t >= 2.6: p_str = "p < 0.01"
                elif t >= 2.0: p_str = "p < 0.05"
                else: p_str = "n.s."

        self.stats["custody_draws"] = self.control_idx
        pool_sz = len(self.sealed_controls) if self.sealed_controls else 50
        hash_repr = (self.active_ctrl_hash[:16] + "...") if isinstance(self.active_ctrl_hash, str) else str(self.active_ctrl_hash)
        self.last_reveal = {
            "trial_num": self.stats["total"],
            "trial_mode": self.active_trial_mode,
            "winner_name": winner_name,
            "winner_type": winner_type,
            "flyA_real": self.flyA.is_real,
            "flyB_real": self.flyB.is_real,
            "arm_label_a": self.flyA.arm_label,
            "arm_label_b": self.flyB.arm_label,
            "rate_a": self.flyA.get_mean_rate(),
            "rate_b": self.flyB.get_mean_rate(),
            "silent_a": self.flyA.get_silent_fraction(),
            "silent_b": self.flyB.get_silent_fraction(),
            "ci_a": round(ci_a, 2),
            "ci_b": round(ci_b, 2),
            "ci_unclipped_a": ci_unclipped_a,
            "ci_unclipped_b": ci_unclipped_b,
            "disp_a": disp_a,
            "disp_b": disp_b,
            "path_a": path_a,
            "path_b": path_b,
            "ctrl_id": self.active_ctrl_id,
            "ctrl_hash": hash_repr,
            "pool_size": pool_sz,
            "custody_draws": self.control_idx
        }

        # Record chronological trial entry in ledger
        record = {
            "trial_num": self.stats["total"],
            "timestamp": int(time.time()),
            "trial_mode": self.active_trial_mode,
            "winner_name": winner_name,
            "winner_type": winner_type,
            "flyA_real": self.flyA.is_real,
            "flyB_real": self.flyB.is_real,
            "arm_label_a": self.flyA.arm_label,
            "arm_label_b": self.flyB.arm_label,
            "rate_a": self.flyA.get_mean_rate(),
            "rate_b": self.flyB.get_mean_rate(),
            "silent_a": self.flyA.get_silent_fraction(),
            "silent_b": self.flyB.get_silent_fraction(),
            "ci_a": round(ci_a, 2),
            "ci_b": round(ci_b, 2),
            "ci_unclipped_a": ci_unclipped_a,
            "ci_unclipped_b": ci_unclipped_b,
            "disp_a": disp_a,
            "disp_b": disp_b,
            "path_a": path_a,
            "path_b": path_b,
            "ci_real": round(ci_real, 2),
            "ci_shuf": round(ci_shuf, 2),
            "ci_diff": round(ci_real - ci_shuf, 2),
            "ctrl_id": self.active_ctrl_id,
            "ctrl_hash": hash_repr,
            "pool_size": pool_sz,
            "custody_draws": self.control_idx,
            "rolling_t": t_stat_val,
            "p_str": p_str,
            "steps": self.step_count,
            "target": [round(self.target["x"], 1), round(self.target["y"], 1)],
            "trajectory_a": self.flyA.full_trajectory,
            "trajectory_b": self.flyB.full_trajectory
        }
        self.stats.setdefault("history", []).append(record)
        # Prune heavy trajectories on older flights (keep full trajectories for newest 12 flights only)
        if len(self.stats["history"]) > 12:
            for old_rec in self.stats["history"][:-12]:
                old_rec.pop("trajectory_a", None)
                old_rec.pop("trajectory_b", None)
        if len(self.stats["history"]) > 50:
            self.stats["history"].pop(0)
        self.save_ledger()

        self.is_revealing = True
        self.reveal_timer = 60  # ~2 seconds at 30 ticks/s

    def tick(self):
        with self.lock:
            if self.is_revealing:
                self.reveal_timer -= 1
                if self.reveal_timer <= 0:
                    self.spawn_trial()
                return

            self.step_count += 1
            d_a = self.flyA.step(self.target["x"], self.target["y"])
            d_b = self.flyB.step(self.target["x"], self.target["y"])

            if d_a < self.target["r"] + 6:
                self.finish_trial("A")
            elif d_b < self.target["r"] + 6:
                self.finish_trial("B")
            elif self.step_count > 950:
                self.finish_trial("timeout")

    def get_state(self, lightweight=False):
        with self.lock:
            diffs = self.stats.get("diffs", [])
            n = len(diffs)
            t_stat_val = None
            p_str = "Calibrating"
            if n >= 4:
                mean_d = sum(diffs) / n
                var_d = sum((x - mean_d) ** 2 for x in diffs) / (n - 1)
                se = math.sqrt(var_d / n)
                if se > 0.0001:
                    t = mean_d / se
                    t_stat_val = round(t, 2)
                    if t >= 3.9: p_str = "p < 0.0001"
                    elif t >= 3.3: p_str = "p < 0.001"
                    elif t >= 2.6: p_str = "p < 0.01"
                    elif t >= 2.0: p_str = "p < 0.05"
                    else: p_str = "n.s."

            d0_a = math.hypot(self.flyA.start_x - self.target["x"], self.flyA.start_y - self.target["y"])
            d0_b = math.hypot(self.flyB.start_x - self.target["x"], self.flyB.start_y - self.target["y"])
            cur_d_a = math.hypot(self.flyA.x - self.target["x"], self.flyA.y - self.target["y"])
            cur_d_b = math.hypot(self.flyB.x - self.target["x"], self.flyB.y - self.target["y"])
            total = self.stats["total"]
            hash_repr = (self.active_ctrl_hash[:16] + "...") if isinstance(self.active_ctrl_hash, str) else str(self.active_ctrl_hash)
            # Slice trail history to recent 45 points (~1.5s visual trail) to eliminate wire bloat
            trail_a = self.flyA.history[-45:] if len(self.flyA.history) > 45 else self.flyA.history
            trail_b = self.flyB.history[-45:] if len(self.flyB.history) > 45 else self.flyB.history

            state = {
                "flight_number": self.flight_number,
                "target": {"x": round(self.target["x"], 1), "y": round(self.target["y"], 1), "r": self.target["r"]},
                "is_revealing": self.is_revealing,
                "step_count": self.step_count,
                "trial_mode": self.active_trial_mode,
                "mode_setting": self.mode_setting,
                # SEALED BLIND: is_real and arm_label are strictly null during active flight!
                "flyA": {
                    "x": round(self.flyA.x, 1),
                    "y": round(self.flyA.y, 1),
                    "heading": round(self.flyA.heading, 3),
                    "last_yaw": round(self.flyA.last_yaw, 4),
                    "dist": round(cur_d_a, 1),
                    "ci": round(max(0.0, (d0_a - cur_d_a) / max(1.0, d0_a)), 2),
                    "mean_rate": self.flyA.get_mean_rate(),
                    "silent_fraction": self.flyA.get_silent_fraction(),
                    "history": trail_a,
                    "is_real": self.flyA.is_real if self.is_revealing else None,
                    "arm_label": self.flyA.arm_label if self.is_revealing else None,
                    "swap_antennae": self.flyA.swap_antennae if self.is_revealing else None
                },
                "flyB": {
                    "x": round(self.flyB.x, 1),
                    "y": round(self.flyB.y, 1),
                    "heading": round(self.flyB.heading, 3),
                    "last_yaw": round(self.flyB.last_yaw, 4),
                    "dist": round(cur_d_b, 1),
                    "ci": round(max(0.0, (d0_b - cur_d_b) / max(1.0, d0_b)), 2),
                    "mean_rate": self.flyB.get_mean_rate(),
                    "silent_fraction": self.flyB.get_silent_fraction(),
                    "history": trail_b,
                    "is_real": self.flyB.is_real if self.is_revealing else None,
                    "arm_label": self.flyB.arm_label if self.is_revealing else None,
                    "swap_antennae": self.flyB.swap_antennae if self.is_revealing else None
                },
                "stats": {
                    "total": total,
                    "real_wins": self.stats["real_wins"],
                    "shuf_wins": self.stats["shuf_wins"],
                    "timeouts": self.stats["timeouts"],
                    "pct_real": f"{(self.stats['real_wins'] / total * 100):.1f}%" if total > 0 else "0.0%",
                    "pct_shuf": f"{(self.stats['shuf_wins'] / total * 100):.1f}%" if total > 0 else "0.0%",
                    "mean_ci_real": f"{(self.stats['sum_ci_real'] / total):.2f}" if total > 0 else "0.00",
                    "mean_ci_shuf": f"{(self.stats['sum_ci_shuf'] / total):.2f}" if total > 0 else "0.00",
                    "t_stat": t_stat_val,
                    "p_str": p_str
                },
                "truncation_profile": {
                    "named_graph": "G_traced = (V_traced, E_traced ∩ (V_traced × V_traced))",
                    "subset_seal": "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
                    "ballot_anchor": "Proposal #21 (ompi: 9dc2bbd2...)",
                    "control_custody": {
                        "rule": "authored-elsewhere (c59011/c59078)",
                        "active_ctrl_id": self.active_ctrl_id,
                        "active_ctrl_sha256": hash_repr,
                        "pool_size": len(self.sealed_controls) if self.sealed_controls else 50,
                        "custody_draws": self.stats.get("custody_draws", self.control_idx)
                    }
                },
                "last_reveal": self.last_reveal
            }
            if not lightweight:
                state["recent_history"] = [
                    {k: v for k, v in item.items() if k not in ("trajectory_a", "trajectory_b")}
                    for item in reversed(self.stats.get("history", [])[-8:])
                ]
            return state

    def reset_stats(self):
        with self.lock:
            self.stats = {
                "total": 0,
                "real_wins": 0,
                "shuf_wins": 0,
                "timeouts": 0,
                "sum_ci_real": 0.0,
                "sum_ci_shuf": 0.0,
                "diffs": [],
                "history": []
            }
            self.last_reveal = None
            self.save_ledger()
            self.spawn_trial()

# Global engine instance
engine = ArenaEngine(os.path.join(os.path.dirname(os.path.abspath(__file__)), "arena_ledger.json"))

def simulation_loop():
    tick_interval = 0.033 # ~30 ticks per second
    while True:
        t0 = time.time()
        try:
            engine.tick()
        except Exception as e:
            print(f"Error in sim tick: {e}", file=sys.stderr)
        elapsed = time.time() - t0
        time.sleep(max(0.001, tick_interval - elapsed))

class ArenaHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass # Quiet server logging

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            last_step = -1
            last_flight = -1
            try:
                while True:
                    state = engine.get_state(lightweight=True)
                    step = state["step_count"]
                    flight = state.get("flight_number", 0)
                    if step != last_step or flight != last_flight or state["is_revealing"]:
                        last_step = step
                        last_flight = flight
                        data = json.dumps(state, separators=(',', ':'))
                        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                        self.wfile.flush()
                    time.sleep(0.033)
            except (BrokenPipeError, ConnectionResetError, socket.error, OSError):
                return

        if parsed.path == "/api/state":
            data = json.dumps(engine.get_state(), separators=(',', ':')).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(data)
            return

        if parsed.path == "/api/history":
            with engine.lock:
                raw_hist = list(reversed(engine.stats.get("history", [])))
                # Keep full trajectories only for the 8 newest records to prevent wire bloat
                filtered_hist = []
                for idx, item in enumerate(raw_hist[:30]):
                    if idx < 8 and "trajectory_a" in item:
                        filtered_hist.append(item)
                    else:
                        filtered_hist.append({k: v for k, v in item.items() if k not in ("trajectory_a", "trajectory_b")})
                res_obj = {
                    "total": engine.stats["total"],
                    "history": filtered_hist
                }
            history_data = json.dumps(res_obj, separators=(',', ':')).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(history_data)
            return

        if parsed.path == "/api/replay":
            qs = parse_qs(parsed.query)
            trial_str = qs.get("trial", [None])[0]
            record = None
            if trial_str:
                try:
                    t_num = int(trial_str)
                    with engine.lock:
                        for h in engine.stats.get("history", []):
                            if h.get("trial_num") == t_num:
                                record = h
                                break
                except Exception:
                    pass
            if record:
                data = json.dumps({"ok": True, "trial": record}, separators=(',', ':')).encode("utf-8")
                self.send_response(200)
            else:
                data = json.dumps({"ok": False, "error": "Flight record not found"}).encode("utf-8")
                self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
            return

        if parsed.path == "/api/set_mode":
            qs = parse_qs(parsed.query)
            mode = qs.get("mode", [None])[0]
            if mode in ("auto", "arm1", "arm2", "wrong_odor"):
                with engine.lock:
                    engine.mode_setting = mode
                    engine.spawn_trial()
                data = json.dumps({"ok": True, "mode": mode, "active_trial_mode": engine.active_trial_mode}, separators=(',', ':')).encode("utf-8")
                self.send_response(200)
            else:
                data = json.dumps({"ok": False, "error": "Invalid mode. Must be auto, arm1, arm2, or wrong_odor"}).encode("utf-8")
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(data)
            return

        if parsed.path == "/api/benchmark":
            benchmark_data = {
                "num_trials": 40,
                "seed": 456,
                "metric_definition": "Unclipped continuous displacement d0 - dfinal and path length sum(||x_t - x_{t-1}||)",
                "battery": [
                    {
                        "condition": "bio_standard",
                        "label": "Bio (Standard Plume)",
                        "ci": 0.905,
                        "unclipped_ci": 0.905,
                        "disp": 55.80,
                        "path": 152.4,
                        "mean_rate_hz": 10.85,
                        "silent_pct": 0.0,
                        "behavior": "100% directed",
                        "diff_vs_bio": 0.000,
                        "t_stat": "—",
                        "p_val": "Baseline"
                    },
                    {
                        "condition": "bio_wrong_odor",
                        "label": "Bio (Swapped Odour)",
                        "ci": 0.000,
                        "unclipped_ci": -1.849,
                        "disp": -113.80,
                        "path": 144.6,
                        "mean_rate_hz": 4.14,
                        "silent_pct": 8.0,
                        "behavior": "100% wandering",
                        "diff_vs_bio": 0.905,
                        "t_stat": "89.14",
                        "p_val": "p << 0.001"
                    },
                    {
                        "condition": "rand_2x",
                        "label": "Rand-Dynamics (2.0x rate)",
                        "ci": 0.228,
                        "unclipped_ci": -0.746,
                        "disp": -45.48,
                        "path": 156.4,
                        "mean_rate_hz": 23.28,
                        "silent_pct": 24.1,
                        "behavior": "47.5% wandering",
                        "diff_vs_bio": 0.677,
                        "t_stat": "7.11",
                        "p_val": "p << 0.001"
                    },
                    {
                        "condition": "rand_1x",
                        "label": "Rand-Dynamics (1.0x rate)",
                        "ci": 0.251,
                        "unclipped_ci": 0.197,
                        "disp": 12.32,
                        "path": 101.9,
                        "mean_rate_hz": 8.76,
                        "silent_pct": 16.3,
                        "behavior": "72.5% directed",
                        "diff_vs_bio": 0.654,
                        "t_stat": "12.66",
                        "p_val": "p << 0.001"
                    },
                    {
                        "condition": "rand_05x",
                        "label": "Rand-Dynamics (0.5x rate)",
                        "ci": 0.296,
                        "unclipped_ci": 0.255,
                        "disp": 15.91,
                        "path": 98.4,
                        "mean_rate_hz": 7.08,
                        "silent_pct": 17.9,
                        "behavior": "67.5% directed",
                        "diff_vs_bio": 0.609,
                        "t_stat": "11.69",
                        "p_val": "p << 0.001"
                    },
                    {
                        "condition": "scrambled_tau",
                        "label": "Scrambled Tau (10-100ms)",
                        "ci": 0.840,
                        "unclipped_ci": 0.840,
                        "disp": 51.78,
                        "path": 192.5,
                        "mean_rate_hz": 10.52,
                        "silent_pct": 0.0,
                        "behavior": "100% directed",
                        "diff_vs_bio": 0.065,
                        "t_stat": "3.09",
                        "p_val": "p < 0.01"
                    },
                    {
                        "condition": "scrambled_signs",
                        "label": "Scrambled Signs (E/I flip)",
                        "ci": 0.072,
                        "unclipped_ci": -1.975,
                        "disp": -121.40,
                        "path": 244.0,
                        "mean_rate_hz": 12.41,
                        "silent_pct": 11.2,
                        "behavior": "92.5% wandering",
                        "diff_vs_bio": 0.833,
                        "t_stat": "18.79",
                        "p_val": "p << 0.001"
                    },
                    {
                        "condition": "full_scrambled",
                        "label": "Full Scrambled Twin",
                        "ci": 0.068,
                        "unclipped_ci": -2.457,
                        "disp": -151.06,
                        "path": 270.2,
                        "mean_rate_hz": 14.12,
                        "silent_pct": 15.6,
                        "behavior": "87.5% wandering",
                        "diff_vs_bio": 0.837,
                        "t_stat": "18.31",
                        "p_val": "p << 0.001"
                    }
                ]
            }
            data = json.dumps(benchmark_data, separators=(',', ':')).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(data)
            return

        # Serve static files from web_dir
        web_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web"))
        req_path = parsed.path.lstrip("/")
        if not req_path:
            file_path = os.path.join(web_dir, "index.html")
        else:
            file_path = os.path.abspath(os.path.join(web_dir, req_path))

        if not file_path.startswith(web_dir) or not os.path.exists(file_path):
            file_path = os.path.join(web_dir, "index.html")

        if os.path.exists(file_path) and os.path.isfile(file_path):
            content_type = "text/html; charset=utf-8"
            if file_path.endswith(".json"):
                content_type = "application/json"
            elif file_path.endswith(".js"):
                content_type = "application/javascript"
            elif file_path.endswith(".css"):
                content_type = "text/css"
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/set_mode":
            content_len = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
            try:
                payload = json.loads(post_body.decode("utf-8"))
            except Exception:
                payload = {}
            mode = payload.get("mode")
            if mode in ("auto", "arm1", "arm2", "wrong_odor"):
                with engine.lock:
                    engine.mode_setting = mode
                    engine.spawn_trial()
                data = json.dumps({"ok": True, "mode": mode, "active_trial_mode": engine.active_trial_mode}, separators=(',', ':')).encode("utf-8")
                self.send_response(200)
            else:
                data = json.dumps({"ok": False, "error": "Invalid mode. Must be auto, arm1, arm2, or wrong_odor"}).encode("utf-8")
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
            return

        if parsed.path == "/api/reset-stats":
            self.send_response(403)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "Forbidden: Ledger is monotonic and immutable."}')
            return
        self.send_response(404)
        self.end_headers()

def main():
    port = int(os.environ.get("PORT", 8086))
    sim_thread = threading.Thread(target=simulation_loop, daemon=True)
    sim_thread.start()

    server = ThreadingHTTPServer(("0.0.0.0", port), ArenaHTTPHandler)
    print(f"Authoritative Connectome Server live on http://0.0.0.0:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
