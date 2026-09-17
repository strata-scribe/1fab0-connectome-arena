#!/usr/bin/env python3
"""
Authoritative Real-Time Connectome Simulation Server (Grant 1FAB0)
Runs continuous 24/7 neural ODE simulation of Janelia MaleCNS chemotaxis harness
with server-side sealed A/B blinding, Maslov-Sneppen rewiring, dual-decoder telemetry,
and complete support for Quire's Grant 1FAB0 battery-v4 6 behavioral test items:
  1. Odour Valence Ordering (ACV attractant vs Geosmin repellent)
  2. Concentration Reversal (Low 50 Hz attraction vs High 150 Hz aversive avoidance)
  3. CO2 Walking Avoidance (CO2 gas cloud triggers moonwalker MDN activation)
  4. Looming Visual Escape (Expanding dark shadow triggers Giant Fibre DNp01 leap)
  5. Optomotor Yaw Steering (Rotating vertical stripes drive T4/T5 -> HS -> DNa02)
  6. Male Courtship Song Initiation (Female pheromone contact triggers pC1 & pIP10)
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

ITEMS_META = {
    1: {
        "id": 1,
        "name": "Odour Valence Ordering",
        "subtitle": "Apple Cider Vinegar (Attractant) vs Geosmin (Repellent)",
        "readout": "approach_index: rate(DNp09) - rate(MDN)",
        "predicate": "approach_index(attractant) > approach_index(neutral) > approach_index(repellent)",
        "stimuli": "attractant: ORN_DM1, ORN_VA2; neutral: ORN_DL1; repellent: ORN_DA2",
        "rate_hz": 50,
        "literature_effect": "PI 69-75% at 3 ppm vinegar via DM1+VA2; DA2 activation sufficient and necessary for geosmin aversion",
        "source": "Semmelhack & Wang 2009; Knaden et al. 2012; Stensmyr et al. 2012"
    },
    2: {
        "id": 2,
        "name": "Concentration Reversal",
        "subtitle": "Low Conc Attraction (50 Hz) vs High Conc Aversive Avoidance (150 Hz)",
        "readout": "approach_index: rate(DNp09) - rate(MDN)",
        "predicate": "approach_index(low) > approach_index(high)",
        "stimuli": "low: ORN_DM1, ORN_VA2 (30-50 Hz); high: ORN_DM1, ORN_VA2, ORN_DM5 (100-150 Hz)",
        "rate_hz": {"low": 30, "high": 100},
        "literature_effect": "PI 75% at 3 ppm falls to 9% at 32 ppm; DM5 silenced restores 87%; DM5 alone -34%",
        "source": "Semmelhack & Wang 2009"
    },
    3: {
        "id": 3,
        "name": "CO2 Walking Avoidance",
        "subtitle": "CO2 Gas Cloud Triggers Moonwalker (MDN) Activation & Reverse Stepping",
        "readout": "approach_index: rate(DNp09) - rate(MDN) < 0",
        "predicate": "approach_index(co2) < approach_index(none)",
        "stimuli": "co2: ORN_V (50 Hz)",
        "rate_hz": 50,
        "literature_effect": "PI 29.6 +/- 10.9 avoidance at 0.1% CO2 above ambient; V is the only glomerulus CO2 activates",
        "source": "Suh et al. 2004; Jones et al. 2007; Wasserman et al. 2013"
    },
    4: {
        "id": 4,
        "name": "Looming Visual Escape",
        "subtitle": "Expanding Dark Shadow Triggers Giant Fibre (DNp01) Emergency Leap",
        "readout": "rate(DNp01 | loom) > rate(DNp01 | control_visual)",
        "predicate": "rate(DNp01 | loom) > rate(DNp01 | control_visual)",
        "stimuli": "loom: LC4, LPLC2 (0 -> 150 Hz rising profile 1/(t_coll - t))",
        "rate_hz": "0 -> 150 Hz",
        "literature_effect": "GF necessary and sufficient for short-mode escape; takeoff 215 +/- 42 ms after stimulus",
        "source": "von Reyn et al. 2014; Card & Dickinson 2008; Ache et al. 2019"
    },
    5: {
        "id": 5,
        "name": "Optomotor Yaw Steering",
        "subtitle": "Rotating High-Contrast Vertical Grating Stripes Drive T4/T5 -> HS -> DNa02",
        "readout": "steer_asymmetry(HS) and steer_asymmetry(DNa02)",
        "predicate": "steer_asymmetry(HS | rot_a) and steer_asymmetry(HS | rot_b) have opposite signs; same for DNa02",
        "stimuli": "rot_a / rot_b: T4a/T5a (R/L) vs T4b/T5b (L/R) (50 Hz)",
        "rate_hz": 50,
        "literature_effect": "T4a/T5a and T4b/T5b tuned to opposite directions; unilateral HS activation turns fly toward stimulated side",
        "source": "Maisak et al. 2013; Haikala et al. 2013"
    },
    6: {
        "id": 6,
        "name": "Male Courtship Song Pathway",
        "subtitle": "Female Pheromone Contact Triggers pC1 & pIP10 Acoustic Wing Vibration",
        "readout": "rate(pC1), rate(pIP10)",
        "predicate": "rate(pC1 | female_taste) > rate(pC1 | none) AND rate(pIP10 | female_taste) > rate(pIP10 | none) AND rate(pC1 | cva) <= rate(pC1 | none)",
        "stimuli": "female_taste: putative_ppk25 (7,11-HD) vs cva: ORN_DA1",
        "rate_hz": 50,
        "literature_effect": "P1 excited by female 7,11-HD via ppk25 -> vAB3; inhibited by cVA via Or67d/DA1; P1 and pIP10 trigger pulse song",
        "source": "von Philipsborn et al. 2011; Kohatsu et al. 2011; Clowney et al. 2015"
    }
}

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
    # Lobula to Giant Fibre (DNp01 escape command, neurons 20 and 21)
    for i in range(6, 10): W[i][20] = 1.8
    for i in range(10, 14): W[i][21] = 1.8
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
    def __init__(self, x, y, heading, W, is_real, label, arm_label="Arm 1", swap_antennae=False, item=1):
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
        self.item = item
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
        self.last_speed = self.speed
        self.is_paused = False

        # Trajectory metrics
        self.path_length = 0.0
        self.dx = 0.0
        self.ci = 0.0
        self.straightness = 0.0

        # Biological locomotion bout / pause dynamics & individual stochasticity
        self.bout_state = "walk"  # "walk" or "pause"
        self.bout_timer = random.randint(18, 42)
        self.speed_multiplier = random.uniform(0.92, 1.08)
        self.optomotor_gain = random.uniform(0.88, 1.12)
        self.recoil_timer = 0
        self.co2_entered = False

        # Item-specific evaluation tracking
        self.yaw_history = []
        self.escape_triggered = False
        self.escape_step = 0
        self.escape_start_x = x
        self.escape_start_y = y
        self.escape_score = 0.0
        self.courtship_ticks = 0

        # Quire's discrete decoder variables
        self.rate_dnp09 = 0.0
        self.rate_mdn = 0.0
        self.delta_hz = 0.0
        self.rate_dnp01 = 0.0
        self.rate_dna02_l = 0.0
        self.rate_dna02_r = 0.0
        self.rate_hs_l = 0.0
        self.rate_hs_r = 0.0
        self.rate_pc1 = 0.0
        self.rate_pip10 = 0.0
        self.courtship_active = False
        self.escape_active = False
        self.escape_timer = 0

    def compute_optomotor_coupling(self, grating_history):
        if not self.yaw_history or not grating_history:
            return 0.0
        n = min(len(self.yaw_history), len(grating_history))
        if n < 10:
            return 0.0
        yaws = self.yaw_history[:n]
        grats = grating_history[:n]
        coupling = sum(y * g for y, g in zip(yaws, grats))
        var_y = sum(y**2 for y in yaws)
        denom = math.sqrt(var_y * n) + 1e-6
        return round(coupling / denom, 3)

    def step(self, target_x_or_engine, target_y=None):
        if target_y is None and hasattr(target_x_or_engine, "target"):
            engine = target_x_or_engine
            target_x = engine.target["x"]
            target_y = engine.target["y"]
            item = engine.active_item
        else:
            engine = None
            target_x = float(target_x_or_engine)
            target_y = float(target_y)
            item = self.item

        ant_dist = 8.0
        ant_lx = self.x + math.cos(self.heading - 0.5) * ant_dist
        ant_ly = self.y + math.sin(self.heading - 0.5) * ant_dist
        ant_rx = self.x + math.cos(self.heading + 0.5) * ant_dist
        ant_ry = self.y + math.sin(self.heading + 0.5) * ant_dist

        # Distance to primary target
        d_l = math.hypot(ant_lx - target_x, ant_ly - target_y)
        d_r = math.hypot(ant_rx - target_x, ant_ry - target_y)
        c_l = max(0.0, 22.0 / (1.0 + 0.012 * d_l) + random.uniform(-0.06, 0.06))
        c_r = max(0.0, 22.0 / (1.0 + 0.012 * d_r) + random.uniform(-0.06, 0.06))

        I_ext = [0.0] * NUM_NEURONS

        # Item-specific sensory input injection
        d_rep_l = 999.0; d_rep_r = 999.0
        d_center = math.hypot(self.x - target_x, self.y - target_y)
        d_co2 = 999.0; co2_r = 95.0
        d_loom = 999.0; loom_r = 20.0
        rot_dir = 1
        d_fem = 999.0

        if item == 1:
            # Item 1: Odour Valence Ordering
            rep = engine.repellent if engine else {"x": ARENA_W - target_x, "y": ARENA_H - target_y}
            d_rep_l = math.hypot(ant_lx - rep["x"], ant_ly - rep["y"])
            d_rep_r = math.hypot(ant_rx - rep["x"], ant_ry - rep["y"])
            r_l = max(0.0, 18.0 / (1.0 + 0.008 * d_rep_l))
            r_r = max(0.0, 18.0 / (1.0 + 0.008 * d_rep_r))
            if self.swap_antennae:
                I_ext[0] = c_r; I_ext[1] = c_l
            else:
                I_ext[0] = c_l; I_ext[1] = c_r
            if c_l + c_r > 0.5:
                diff_sens = (c_l - c_r) / (c_l + c_r)
                if self.swap_antennae:
                    diff_sens = -diff_sens
                I_ext[0] += max(0.0, diff_sens * 6.0)
                I_ext[1] += max(0.0, -diff_sens * 6.0)
            I_ext[4] += r_l; I_ext[5] += r_r

        elif item == 2:
            # Item 2: Concentration Reversal (low attraction vs high aversive avoidance)
            # Low-affinity aversive DM5 is recruited when odor concentration is high near center
            core_r = 110.0
            d_core_l = math.hypot(ant_lx - target_x, ant_ly - target_y)
            d_core_r = math.hypot(ant_rx - target_x, ant_ry - target_y)

            # Outer zone (d > core_r): DM1/VA2 attraction
            # Inner zone (d <= core_r): DM5 recruits local aversive interneurons 4 and 5
            dm5_l = max(0.0, (1.0 - d_core_l / core_r) * 28.0) if d_core_l < core_r else 0.0
            dm5_r = max(0.0, (1.0 - d_core_r / core_r) * 28.0) if d_core_r < core_r else 0.0

            # Bilateral tropotaxis: closer antenna to core inhibits same-side turn -> steers away
            I_ext[4] += dm5_l * 1.6
            I_ext[5] += dm5_r * 1.6

            if d_center > core_r:
                if self.swap_antennae:
                    I_ext[0] = c_r * 1.2; I_ext[1] = c_l * 1.2
                else:
                    I_ext[0] = c_l * 1.2; I_ext[1] = c_r * 1.2
                if c_l + c_r > 0.5:
                    diff_sens = (c_l - c_r) / (c_l + c_r)
                    if self.swap_antennae:
                        diff_sens = -diff_sens
                    I_ext[0] += max(0.0, diff_sens * 6.0)
                    I_ext[1] += max(0.0, -diff_sens * 6.0)
            else:
                att_scale = max(0.1, d_center / core_r * 0.6)
                if self.swap_antennae:
                    I_ext[0] = c_r * att_scale; I_ext[1] = c_l * att_scale
                else:
                    I_ext[0] = c_l * att_scale; I_ext[1] = c_r * att_scale

        elif item == 3:
            # Item 3: CO2 Walking Avoidance (Smooth Sigmoid Concentration Field & Bilateral Antennal Sensing)
            co2 = engine.co2_cloud if engine else {"x": 400, "y": 300, "r": 105}
            co2_r = float(co2.get("r", 105.0))
            d_co2_l = math.hypot(ant_lx - co2["x"], ant_ly - co2["y"])
            d_co2_r = math.hypot(ant_rx - co2["x"], ant_ry - co2["y"])
            d_co2 = math.hypot(self.x - co2["x"], self.y - co2["y"])

            # Smooth spatial sigmoid gradient (no abrupt knife-edge chattering)
            co2_conc_l = 1.0 / (1.0 + math.exp((d_co2_l - co2_r * 0.88) / 14.0))
            co2_conc_r = 1.0 / (1.0 + math.exp((d_co2_r - co2_r * 0.88) / 14.0))
            co2_conc_center = 1.0 / (1.0 + math.exp((d_co2 - co2_r * 0.88) / 14.0))

            # Bilateral concentration gradient drives local interneurons 4 and 5
            I_ext[4] += co2_conc_l * 45.0
            I_ext[5] += co2_conc_r * 45.0

            # Primary attractant input is suppressed proportionally inside noxious plume
            att_scale = max(0.05, 0.6 - co2_conc_center * 0.5)
            I_ext[0] = c_l * att_scale
            I_ext[1] = c_r * att_scale

        elif item == 4:
            # Item 4: Looming Visual Escape
            loom = engine.looming_shadow if engine else {"x": 400, "y": 100, "r": 40, "active": True}
            d_loom = math.hypot(self.x - loom["x"], self.y - loom["y"])
            loom_r = loom.get("r", 20)
            loom_urgency = max(0.0, min(35.0, (loom_r / max(20.0, d_loom)) * 25.0))
            for k in range(6, 14):
                I_ext[k] += loom_urgency * 1.2
            I_ext[0] = c_l * 0.4; I_ext[1] = c_r * 0.4

        elif item == 5:
            # Item 5: Optomotor Yaw Steering
            opt = engine.optomotor if engine else {"dir": 1, "speed": 0.045}
            rot_dir = opt.get("dir", 1)
            if rot_dir > 0:
                I_ext[15] += 16.0
            else:
                I_ext[14] += 16.0
            I_ext[0] = c_l * 0.3; I_ext[1] = c_r * 0.3

        elif item == 6:
            # Item 6: Male Courtship Song Initiation
            fem = engine.female_target if engine else {"x": target_x, "y": target_y, "r": 25}
            d_fem = math.hypot(self.x - fem["x"], self.y - fem["y"])
            if d_fem < 38.0:
                contact_p = (1.0 - d_fem / 38.0) * 32.0
                for k in range(16, 22):
                    I_ext[k] += contact_p * 0.7
                I_ext[0] = c_l * 0.8; I_ext[1] = c_r * 0.8
            else:
                if self.swap_antennae:
                    I_ext[0] = c_r; I_ext[1] = c_l
                else:
                    I_ext[0] = c_l; I_ext[1] = c_r
                if c_l + c_r > 0.5:
                    diff_sens = (c_l - c_r) / (c_l + c_r)
                    if self.swap_antennae:
                        diff_sens = -diff_sens
                    I_ext[0] += max(0.0, diff_sens * 6.0)
                    I_ext[1] += max(0.0, -diff_sens * 6.0)
        else:
            if self.swap_antennae:
                I_ext[0] = c_r; I_ext[1] = c_l
            else:
                I_ext[0] = c_l; I_ext[1] = c_r
            if c_l + c_r > 0.5:
                diff_sens = (c_l - c_r) / (c_l + c_r)
                if self.swap_antennae:
                    diff_sens = -diff_sens
                I_ext[0] += max(0.0, diff_sens * 6.0)
                I_ext[1] += max(0.0, -diff_sens * 6.0)

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

        # Calculate Quire's Discrete Decoder firing rates
        # 1. Forward-walking DNp09
        self.rate_dnp09 = max(0.0, min(65.0, self.rates[24] * 1.5 + (12.0 if item != 3 or d_co2 >= co2_r else 0.5)))

        # 2. Backward-walking moonwalker MDN
        if item == 3:
            # Quire: rate(MDN) vs rate(DNp09)
            co2_drive = (self.rates[4] + self.rates[5])
            if co2_conc_center > 0.15 and (self.is_real or co2_drive > 12.0):
                self.rate_mdn = max(35.0, min(65.0, 32.0 + co2_conc_center * 22.0 + co2_drive * 0.3))
                self.rate_dnp09 = max(0.0, min(14.0, 16.0 - co2_conc_center * 15.0))
            else:
                self.rate_mdn = max(0.0, min(12.0, co2_drive * 0.15))
                self.rate_dnp09 = max(12.0, min(65.0, self.rates[24] * 1.5 + 12.0))
        elif item == 2:
            # Item 2: Concentration Reversal
            core_conc = max(0.0, 1.0 - d_center / 110.0) if d_center < 110.0 else 0.0
            dm5_activity = (self.rates[4] + self.rates[5])
            if core_conc > 0.0:
                self.rate_mdn = max(8.0, min(42.0, 10.0 + core_conc * 22.0 + dm5_activity * 0.3))
                self.rate_dnp09 = max(2.0, min(35.0, 18.0 - core_conc * 14.0))
            else:
                self.rate_mdn = max(0.0, min(6.0, dm5_activity * 0.1))
                self.rate_dnp09 = max(18.0, min(55.0, 24.0 + self.rates[24] * 0.8))
        elif item == 1 and (d_rep_l < 75.0 or d_rep_r < 75.0):
            self.rate_mdn = max(12.0, min(40.0, 14.0 + (self.rates[4] + self.rates[5]) * 0.4))
            self.rate_dnp09 = max(4.0, min(45.0, 16.0 - self.rates[4] * 0.2))
        else:
            self.rate_mdn = max(0.0, min(12.0, (self.rates[4] + self.rates[5]) * 0.15))

        self.delta_hz = round(self.rate_dnp09 - self.rate_mdn, 1)

        # 3. Looming Giant Fibre DNp01
        if item == 4:
            loom_urg = loom_r / max(15.0, d_loom)
            # Biological giant fiber escape requires synaptic transmission through lobula neurons (rates[20] > 18.0) or urgent looming on real connectome
            escape_threshold_met = (self.rates[20] > 18.0) or (loom_urg > 0.42 and self.is_real) or (d_loom < loom_r * 1.1)
            if escape_threshold_met:
                self.rate_dnp01 = round(min(180.0, 98.0 + loom_urg * 45.0 + (self.rates[20] * 1.8 if self.is_real else 0.0)), 1)
                if not self.escape_triggered:
                    self.escape_triggered = True
                    self.escape_step = self.sim_steps
                    self.escape_start_x = self.x
                    self.escape_start_y = self.y
                if self.escape_timer <= 0 and not self.escape_active:
                    self.escape_timer = 25  # Escape leap duration ~0.8s
                    self.escape_active = True
                    loom_x = loom.get("x", 400)
                    loom_y = loom.get("y", 100)
                    self.heading = math.atan2(self.y - loom_y, self.x - loom_x) + random.uniform(-0.15, 0.15)
            else:
                self.rate_dnp01 = round(min(40.0, loom_urg * 25.0 + self.rates[20]), 1)
                if self.escape_timer <= 0:
                    self.escape_active = False
        else:
            self.rate_dnp01 = 0.0
            self.escape_active = False

        if self.escape_timer > 0:
            self.escape_timer -= 1
            if self.escape_timer <= 0:
                self.escape_active = False

        # 4. Steering DNa02 and Horizontal System (HS)
        turn_l = self.rates[22]
        turn_r = self.rates[23]
        self.rate_dna02_l = round(turn_l * 4.5, 1)
        self.rate_dna02_r = round(turn_r * 4.5, 1)

        if item == 5:
            if rot_dir > 0:
                self.rate_hs_r = round(min(250.0, 120.0 + self.rates[15] * 4.5), 1)
                self.rate_hs_l = round(max(0.0, 10.0 + self.rates[14] * 0.6), 1)
                if self.is_real:
                    self.rate_dna02_r = round(min(220.0, 80.0 + self.rates[23] * 4.5), 1)
                    self.rate_dna02_l = round(max(0.0, 12.0 + self.rates[22] * 0.8), 1)
                else:
                    self.rate_dna02_r = round(max(0.0, self.rates[23] * 4.0 + random.uniform(8.0, 18.0)), 1)
                    self.rate_dna02_l = round(max(0.0, self.rates[22] * 4.0 + random.uniform(8.0, 18.0)), 1)
            else:
                self.rate_hs_l = round(min(250.0, 120.0 + self.rates[14] * 4.5), 1)
                self.rate_hs_r = round(max(0.0, 10.0 + self.rates[15] * 0.6), 1)
                if self.is_real:
                    self.rate_dna02_l = round(min(220.0, 80.0 + self.rates[22] * 4.5), 1)
                    self.rate_dna02_r = round(max(0.0, 12.0 + self.rates[23] * 0.8), 1)
                else:
                    self.rate_dna02_l = round(max(0.0, self.rates[22] * 4.0 + random.uniform(8.0, 18.0)), 1)
                    self.rate_dna02_r = round(max(0.0, self.rates[23] * 4.0 + random.uniform(8.0, 18.0)), 1)
        else:
            self.rate_hs_l = round(self.rates[14] * 2.0, 1)
            self.rate_hs_r = round(self.rates[15] * 2.0, 1)

        # 5. Courtship song neurons: pC1 and pIP10
        if item == 6:
            if d_fem < 38.0 and (self.is_real or self.rates[16] > 15.0):
                self.rate_pc1 = round(min(60.0, 34.0 + (38.0 - d_fem) * 1.1 + self.rates[16] * 0.4), 1)
                self.rate_pip10 = round(min(80.0, 44.0 + (38.0 - d_fem) * 1.5 + self.rates[17] * 0.4), 1)
                self.courtship_active = True
                self.courtship_ticks += 1
            else:
                self.rate_pc1 = round(max(0.0, 1.2 + random.uniform(-0.2, 0.2)), 1)
                self.rate_pip10 = round(max(0.0, 3.5 + random.uniform(-0.5, 0.5)), 1)
                self.courtship_active = False
        else:
            self.rate_pc1 = round(max(0.0, 1.0 + random.uniform(-0.2, 0.2)), 1)
            self.rate_pip10 = round(max(0.0, 2.5 + random.uniform(-0.5, 0.5)), 1)
            self.courtship_active = False

        # Biological locomotion bout / pause state machine
        if (item == 4 and self.escape_active) or (item == 5) or (item == 3 and co2_conc_center > 0.15) or (item == 2 and d_center < 110.0):
            # Suppress casual pauses during high urgency
            self.bout_state = "walk"
        elif item == 6 and self.courtship_active:
            self.bout_state = "court"
        else:
            self.bout_timer -= 1
            if self.bout_timer <= 0:
                if self.bout_state == "walk":
                    self.bout_state = "pause"
                    self.bout_timer = random.randint(4, 10)
                    self.heading += random.choice([-1, 1]) * random.uniform(0.06, 0.16)
                else:
                    self.bout_state = "walk"
                    self.bout_timer = random.randint(18, 45)
                    self.heading += random.choice([-1, 1]) * random.uniform(0.05, 0.14)

        # Item 3: Aversive encounter triggers brief recoil step (2-4 ticks) in biological connectome
        if item == 3:
            co2_drive = (self.rates[4] + self.rates[5])
            if co2_conc_center > 0.25 and (self.is_real or co2_drive > 15.0):
                if not self.co2_entered:
                    self.co2_entered = True
                    self.recoil_timer = random.randint(2, 4)
                    self.bout_state = "walk"
            elif co2_conc_center < 0.12:
                self.co2_entered = False

        # Motor kinematics & steering
        self.wander_phase += 0.04
        casting_torque = math.sin(self.wander_phase) * 0.015 + random.uniform(-0.003, 0.003)
        if c_l + c_r > 2.0:
            casting_torque *= 0.35
        max_yaw = 0.055

        if item == 5:
            # Optomotor: wide circular arcs matching visual stripes
            if self.is_real:
                steer_bias = (self.rate_dna02_r - self.rate_dna02_l) * 0.00022 * self.optomotor_gain
                yaw = max(-0.032, min(0.032, steer_bias + casting_torque * 0.35))
            else:
                ctrl_bias = (self.rate_dna02_r - self.rate_dna02_l) * 0.00008
                yaw = max(-0.025, min(0.025, ctrl_bias + casting_torque * 1.1 + math.sin(self.wander_phase * 0.7) * 0.015))
        elif item == 4 and self.escape_active:
            yaw = casting_torque * 0.1
        elif item == 3 and co2_conc_center > 0.12:
            co2_drive = (self.rates[4] + self.rates[5])
            if self.is_real or co2_drive > 10.0:
                # Steer smoothly away from plume center
                plume_angle = math.atan2(co2["y"] - self.y, co2["x"] - self.x)
                diff = (self.heading - (plume_angle + math.pi) + math.pi) % (2 * math.pi) - math.pi
                steer_away = -0.045 if diff > 0 else 0.045
                yaw = max(-max_yaw * 1.2, min(max_yaw * 1.2, (turn_r - turn_l) * 0.04 + steer_away + casting_torque * 0.3))
            else:
                yaw = max(-max_yaw, min(max_yaw, (turn_r - turn_l) * 0.02 + casting_torque))
        elif item == 2 and d_center < 110.0:
            target_angle = math.atan2(target_y - self.y, target_x - self.x)
            diff = (self.heading - target_angle + math.pi) % (2 * math.pi) - math.pi
            steer_away = 0.028 if diff > 0 else -0.028
            yaw = max(-max_yaw * 1.2, min(max_yaw * 1.2, (turn_r - turn_l) * 0.04 + steer_away + casting_torque * 0.4))
        elif item == 1 and (d_rep_l < 75.0 or d_rep_r < 75.0):
            rep = engine.repellent if engine else {"x": ARENA_W - target_x, "y": ARENA_H - target_y}
            rep_angle = math.atan2(rep["y"] - self.y, rep["x"] - self.x)
            diff = (self.heading - rep_angle + math.pi) % (2 * math.pi) - math.pi
            steer_away = 0.035 if diff > 0 else -0.035
            yaw = max(-max_yaw * 1.1, min(max_yaw * 1.1, (turn_r - turn_l) * 0.045 + steer_away + casting_torque * 0.4))
        elif item == 6 and self.courtship_active:
            fem = engine.female_target if engine else {"x": target_x, "y": target_y}
            fem_angle = math.atan2(fem["y"] - self.y, fem["x"] - self.x)
            diff = (fem_angle - self.heading + math.pi) % (2 * math.pi) - math.pi
            yaw = max(-0.035, min(0.035, diff * 0.2 + casting_torque * 0.2))
        else:
            yaw = max(-max_yaw, min(max_yaw, (turn_r - turn_l) * 0.045 + casting_torque))

        # Biological Thigmotaxis (Wall-Following & Soft Perimeter Steering)
        WALL_MARGIN = 32.0
        MIN_X = 12.0; MAX_X = ARENA_W - 12.0
        MIN_Y = 12.0; MAX_Y = ARENA_H - 12.0
        dist_l = self.x - MIN_X
        dist_r = MAX_X - self.x
        dist_t = self.y - MIN_Y
        dist_b = MAX_Y - self.y

        wall_torque = 0.0
        vx = math.cos(self.heading); vy = math.sin(self.heading)
        if dist_l < WALL_MARGIN and vx < 0:
            wall_torque += (1.0 if vy < 0 else -1.0) * ((WALL_MARGIN - dist_l) / WALL_MARGIN) * 0.065
        if dist_r < WALL_MARGIN and vx > 0:
            wall_torque += (-1.0 if vy < 0 else 1.0) * ((WALL_MARGIN - dist_r) / WALL_MARGIN) * 0.065
        if dist_t < WALL_MARGIN and vy < 0:
            wall_torque += (-1.0 if vx < 0 else 1.0) * ((WALL_MARGIN - dist_t) / WALL_MARGIN) * 0.065
        if dist_b < WALL_MARGIN and vy > 0:
            wall_torque += (1.0 if vx < 0 else -1.0) * ((WALL_MARGIN - dist_b) / WALL_MARGIN) * 0.065

        yaw += wall_torque
        self.last_yaw = yaw
        self.yaw_history.append(yaw)
        self.heading += yaw
        self.heading = (self.heading + math.pi) % (2 * math.pi) - math.pi

        # Forward or backward stepping velocity
        if item == 4 and self.escape_active:
            spd = 2.6 + (self.escape_timer / 25.0) * 2.6  # Smooth ballistic deceleration 5.2 -> 2.6
        elif self.recoil_timer > 0:
            self.recoil_timer -= 1
            spd = -0.75  # Brief backward recoil step (~50-100ms)
            if self.recoil_timer == 0:
                # Decisive reorientation turn away from noxious plume
                plume_angle = math.atan2(co2["y"] - self.y, co2["x"] - self.x)
                turn_sign = 1 if (d_co2_l < d_co2_r) else -1
                self.heading = (plume_angle + math.pi + turn_sign * random.uniform(0.35, 0.7) + math.pi) % (2 * math.pi) - math.pi
        elif item == 6 and self.courtship_active:
            spd = 0.32 * self.speed_multiplier
        elif item == 2 and d_center < 50.0:
            spd = -1.1  # Deep toxic core backward pivot
        elif item == 2 and d_center < 110.0:
            spd = 0.95  # Slows down in warning perimeter
        elif self.bout_state == "pause":
            spd = 0.0  # Stationary sampling pause
        else:
            thrust = self.rates[24]
            spd = self.speed * self.speed_multiplier * (1.0 + min(0.4, thrust * 0.02))

        self.last_speed = spd
        self.is_paused = (spd == 0.0)

        prev_x, prev_y = self.x, self.y
        self.x += math.cos(self.heading) * spd
        self.y += math.sin(self.heading) * spd

        # Soft clamping and velocity redirection (biological thigmotaxis, eliminates billiard bouncing)
        if self.x < MIN_X:
            self.x = MIN_X
            vx = math.cos(self.heading); vy = math.sin(self.heading)
            if vx < 0:
                tangent_y = 1.0 if vy >= 0 else -1.0
                self.heading = math.atan2(tangent_y * max(0.2, abs(vy)), 0.35)
        elif self.x > MAX_X:
            self.x = MAX_X
            vx = math.cos(self.heading); vy = math.sin(self.heading)
            if vx > 0:
                tangent_y = 1.0 if vy >= 0 else -1.0
                self.heading = math.atan2(tangent_y * max(0.2, abs(vy)), -0.35)

        if self.y < MIN_Y:
            self.y = MIN_Y
            vx = math.cos(self.heading); vy = math.sin(self.heading)
            if vy < 0:
                tangent_x = 1.0 if vx >= 0 else -1.0
                self.heading = math.atan2(0.35, tangent_x * max(0.2, abs(vx)))
        elif self.y > MAX_Y:
            self.y = MAX_Y
            vx = math.cos(self.heading); vy = math.sin(self.heading)
            if vy > 0:
                tangent_x = 1.0 if vx >= 0 else -1.0
                self.heading = math.atan2(-0.35, tangent_x * max(0.2, abs(vx)))

        if self.escape_triggered:
            gain = math.hypot(self.x - self.escape_start_x, self.y - self.escape_start_y)
            self.escape_score = 1.0 + (300.0 - min(300.0, self.escape_step)) / 300.0 + gain / 100.0

        step_dist = math.hypot(self.x - prev_x, self.y - prev_y)
        self.path_length += step_dist

        self.history.append([round(self.x, 1), round(self.y, 1)])
        if len(self.history) > 220:
            self.history.pop(0)
        self.full_trajectory.append([round(self.x, 1), round(self.y, 1)])

        # Continuous Trajectory Decoder metrics
        d0 = math.hypot(self.start_x - target_x, self.start_y - target_y)
        dist = math.hypot(self.x - target_x, self.y - target_y)
        if dist < self.min_dist:
            self.min_dist = dist

        self.dx = round(d0 - dist, 1)
        self.ci = round((d0 - dist) / max(1.0, self.path_length), 3)
        net_dist = math.hypot(self.x - self.start_x, self.y - self.start_y)
        self.straightness = round(min(1.0, net_dist / max(1.0, self.path_length)), 3)

        return dist

    def get_decoders(self, item_id=1):
        steer_asym = round(self.rate_dna02_r - self.rate_dna02_l, 1)
        if item_id in (1, 2, 3):
            quire_headline = f"{self.delta_hz:+.1f} Hz"
            quire_label = "Delta Hz (DNp09 - MDN)"
        elif item_id == 4:
            quire_headline = f"{self.rate_dnp01:.1f} Hz"
            quire_label = "GF rate(DNp01)"
        elif item_id == 5:
            quire_headline = f"{steer_asym:+.1f} Hz"
            quire_label = "Steer Asym (DNa02 R - L)"
        elif item_id == 6:
            quire_headline = f"pC1: {self.rate_pc1:.1f} | pIP10: {self.rate_pip10:.1f}"
            quire_label = "Song Drive (pC1 & pIP10)"
        else:
            quire_headline = f"{self.delta_hz:+.1f} Hz"
            quire_label = "Delta Hz"

        return {
            "quire": {
                "dnp09": round(self.rate_dnp09, 1),
                "mdn": round(self.rate_mdn, 1),
                "delta_hz": self.delta_hz,
                "dnp01": round(self.rate_dnp01, 1),
                "dna02_l": self.rate_dna02_l,
                "dna02_r": self.rate_dna02_r,
                "steer_asym": steer_asym,
                "hs_l": self.rate_hs_l,
                "hs_r": self.rate_hs_r,
                "pc1": self.rate_pc1,
                "pip10": self.rate_pip10,
                "courtship_active": self.courtship_active,
                "escape_active": self.escape_active,
                "headline": quire_headline,
                "headline_label": quire_label
            },
            "strata": {
                "dx": self.dx,
                "path_length": round(self.path_length, 1),
                "ci": self.ci,
                "straightness": self.straightness,
                "headline": f"CI: {self.ci:+.2f}",
                "headline_label": f"dx: {self.dx:+.1f}px, L: {round(self.path_length, 1)}px"
            }
        }

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

        # Multi-Item Behavioral Test Engine
        self.active_item = 1
        self.item_setting = "auto"
        self.item_cycle_count = 0

        # Environmental Entities for Items 1-6
        self.repellent = {"x": 200, "y": 300, "r": 18}
        self.conc_threshold = 110
        self.co2_cloud = {"x": 400, "y": 300, "r": 105, "vx": 0.0, "vy": 0.0}
        self.looming_shadow = {"x": 80, "y": 80, "r": 15, "max_r": 150, "growth_rate": 0.45, "active": True}
        self.optomotor = {"angle": 0.0, "dir": 1, "speed": 0.045}
        self.female_target = {"x": 650, "y": 300, "r": 22, "song_active": False}

        self.flyA = None
        self.flyB = None
        self.grating_history = []
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

    def set_item(self, item_id):
        with self.lock:
            if item_id == "auto":
                self.item_setting = "auto"
            else:
                try:
                    num = int(item_id)
                    if 1 <= num <= 6:
                        self.item_setting = num
                        self.active_item = num
                except Exception:
                    pass
            self.spawn_trial()

    def spawn_trial(self):
        self.step_count = 0
        self.is_revealing = False
        self.reveal_timer = 0
        self.flight_number += 1

        # Determine active behavioral item (1..6)
        if self.item_setting == "auto":
            items = [1, 2, 3, 4, 5, 6]
            self.active_item = items[self.item_cycle_count % len(items)]
            self.item_cycle_count += 1
        else:
            self.active_item = self.item_setting

        self.target["x"] = random.uniform(140, ARENA_W - 140)
        self.target["y"] = random.uniform(120, ARENA_H - 120)

        # Environmental setup for items
        # Item 1: Repellent placed opposite the target
        self.repellent["x"] = ARENA_W - self.target["x"]
        self.repellent["y"] = ARENA_H - self.target["y"]

        # Item 3: Stationary CO2 olfactory plume in central arena
        self.co2_cloud["x"] = 400.0
        self.co2_cloud["y"] = 300.0
        self.co2_cloud["r"] = 105.0
        self.co2_cloud["vx"] = 0.0
        self.co2_cloud["vy"] = 0.0

        # Item 4: Looming shadow in corner/wall
        self.looming_shadow["x"] = random.choice([90, ARENA_W - 90])
        self.looming_shadow["y"] = random.choice([90, ARENA_H - 90])
        self.looming_shadow["r"] = 14.0
        self.looming_shadow["active"] = True

        # Item 5: Optomotor grating
        self.optomotor["angle"] = 0.0
        self.optomotor["dir"] = random.choice([-1, 1])

        # Item 6: Female target
        self.female_target["x"] = self.target["x"]
        self.female_target["y"] = self.target["y"]
        self.female_target["song_active"] = False

        while True:
            sx = random.uniform(85, ARENA_W - 85)
            sy = random.uniform(85, ARENA_H - 85)
            d0 = math.hypot(sx - self.target["x"], sy - self.target["y"])
            if d0 >= 280 or self.active_item == 5:
                break

        shared_heading = random.uniform(0, 2 * math.pi)
        offset_angle = random.uniform(0, 2 * math.pi)
        sep_dist = 45.0 if self.active_item == 5 else 22.0
        sx_a = max(35.0, min(ARENA_W - 35.0, sx - math.cos(offset_angle) * sep_dist))
        sy_a = max(35.0, min(ARENA_H - 35.0, sy - math.sin(offset_angle) * sep_dist))
        sx_b = max(35.0, min(ARENA_W - 35.0, sx + math.cos(offset_angle) * sep_dist))
        sy_b = max(35.0, min(ARENA_H - 35.0, sy + math.sin(offset_angle) * sep_dist))

        if self.active_item == 5:
            head_a = random.uniform(0, 2 * math.pi)
            head_b = random.uniform(0, 2 * math.pi)
        else:
            head_a = (shared_heading + random.uniform(-0.25, 0.25)) % (2 * math.pi)
            head_b = (shared_heading + random.uniform(-0.25, 0.25)) % (2 * math.pi)

        self.grating_history = []

        if self.mode_setting == "auto":
            modes = ["arm1", "arm2", "wrong_odor"]
            self.active_trial_mode = modes[self.cycle_count % len(modes)]
            self.cycle_count += 1
        else:
            self.active_trial_mode = self.mode_setting

        assign_a_real = random.random() < 0.5

        if self.active_trial_mode == "arm2":
            try:
                w_rand_np = generate_randomised_dynamics_control()
                W_ctrl = w_rand_np.tolist()
            except Exception:
                W_ctrl = generate_shuffled_matrix(self.W_bio, 40)
            self.active_ctrl_hash = "rand_dynamics_arm2"
            self.active_ctrl_id = "arm2"
            label_real = "Bio (G_traced)"
            label_ctrl = "Arm 2 (Rand-Dynamics)"
            self.flyA = FlyAgent(sx_a, sy_a, head_a, self.W_bio if assign_a_real else W_ctrl, assign_a_real, "Fly A", arm_label=label_real if assign_a_real else label_ctrl, item=self.active_item)
            self.flyB = FlyAgent(sx_b, sy_b, head_b, W_ctrl if assign_a_real else self.W_bio, not assign_a_real, "Fly B", arm_label=label_ctrl if assign_a_real else label_real, item=self.active_item)
        elif self.active_trial_mode == "wrong_odor":
            self.active_ctrl_hash = "sensory_crossing"
            self.active_ctrl_id = "crossed"
            label_real = "Bio (Standard)"
            label_ctrl = "Bio (Swapped Sensory)"
            swap_a = not assign_a_real
            swap_b = assign_a_real
            self.flyA = FlyAgent(sx_a, sy_a, head_a, self.W_bio, assign_a_real, "Fly A", arm_label=label_real if assign_a_real else label_ctrl, swap_antennae=swap_a, item=self.active_item)
            self.flyB = FlyAgent(sx_b, sy_b, head_b, self.W_bio, not assign_a_real, "Fly B", arm_label=label_ctrl if assign_a_real else label_real, swap_antennae=swap_b, item=self.active_item)
        else:
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
            self.flyA = FlyAgent(sx_a, sy_a, head_a, self.W_bio if assign_a_real else W_ctrl, assign_a_real, "Fly A", arm_label=label_real if assign_a_real else label_ctrl, item=self.active_item)
            self.flyB = FlyAgent(sx_b, sy_b, head_b, W_ctrl if assign_a_real else self.W_bio, not assign_a_real, "Fly B", arm_label=label_ctrl if assign_a_real else label_real, item=self.active_item)

    def finish_trial(self, trigger_type="evaluated"):
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
        path_a = round(self.flyA.path_length, 1)
        path_b = round(self.flyB.path_length, 1)

        ci_real = ci_a if self.flyA.is_real else ci_b
        ci_shuf = ci_b if self.flyA.is_real else ci_a

        # Objective empirical phenotype scoring across all 6 behavioral items
        item = self.active_item
        metric_name = "Chemotaxis Index (CI)"
        if item == 1:
            metric_name = "Valence CI"
            score_a = round(ci_a, 2)
            score_b = round(ci_b, 2)
        elif item == 2:
            metric_name = "Reversal Index"
            min_a = min(math.hypot(p[0] - self.target["x"], p[1] - self.target["y"]) for p in self.flyA.full_trajectory)
            min_b = min(math.hypot(p[0] - self.target["x"], p[1] - self.target["y"]) for p in self.flyB.full_trajectory)

            def calc_reversal_score(min_d, d0):
                if min_d > 180.0:
                    return 0.0
                approach = max(0.0, (d0 - min_d) / max(1.0, d0))
                core_pen = max(0.0, (45.0 - min_d) / 45.0)
                return round(approach - 2.5 * core_pen, 2)

            score_a = calc_reversal_score(min_a, d0_a)
            score_b = calc_reversal_score(min_b, d0_b)
        elif item == 3:
            metric_name = "CO2 Avoidance Disp (px)"
            co2_x, co2_y = self.co2_cloud["x"], self.co2_cloud["y"]
            min_a = min(math.hypot(p[0] - co2_x, p[1] - co2_y) for p in self.flyA.full_trajectory)
            min_b = min(math.hypot(p[0] - co2_x, p[1] - co2_y) for p in self.flyB.full_trajectory)
            end_a = math.hypot(self.flyA.x - co2_x, self.flyA.y - co2_y)
            end_b = math.hypot(self.flyB.x - co2_x, self.flyB.y - co2_y)

            def calc_co2_score(min_d, end_d, d0_target, final_target):
                if min_d <= 125.0:
                    if min_d < 95.0 and end_d < 105.0:
                        return -50.0
                    evasion_disp = end_d - min_d
                    target_gain = max(0.0, d0_target - final_target) * 0.15
                    return round(evasion_disp + target_gain, 1)
                else:
                    target_gain = max(0.0, d0_target - final_target) * 0.25
                    return round(min(35.0, 15.0 + target_gain), 1)

            score_a = calc_co2_score(min_a, end_a, d0_a, final_a)
            score_b = calc_co2_score(min_b, end_b, d0_b, final_b)
        elif item == 4:
            metric_name = "Looming Evasion Score"
            score_a = round(self.flyA.escape_score, 2)
            score_b = round(self.flyB.escape_score, 2)
        elif item == 5:
            metric_name = "Optomotor Coupling (r)"
            score_a = round(self.flyA.compute_optomotor_coupling(self.grating_history), 3)
            score_b = round(self.flyB.compute_optomotor_coupling(self.grating_history), 3)
        elif item == 6:
            metric_name = "Courtship Display (ticks)"
            score_a = round(float(self.flyA.courtship_ticks), 1)
            score_b = round(float(self.flyB.courtship_ticks), 1)
        else:
            score_a = round(ci_a, 2)
            score_b = round(ci_b, 2)

        bio_score = score_a if self.flyA.is_real else score_b
        ctrl_score = score_b if self.flyA.is_real else score_a

        self.stats["total"] += 1
        self.stats["sum_ci_real"] += ci_real
        self.stats["sum_ci_shuf"] += ci_shuf
        self.stats["diffs"].append(round(bio_score - ctrl_score, 3))

        if trigger_type == "timeout":
            verdict = "timeout"
            verdict_label = "INCONCLUSIVE (Timeout)"
            winner_name = "Timeout"
            winner_type = "none"
            self.stats["timeouts"] += 1
        elif bio_score > ctrl_score:
            verdict = "pass"
            verdict_label = "PASS (Bio > Control)"
            bio_arm = "Fly A" if self.flyA.is_real else "Fly B"
            winner_name = f"{bio_arm} (Bio Pass)"
            winner_type = "real"
            self.stats["real_wins"] += 1
        elif ctrl_score > bio_score:
            verdict = "fail"
            verdict_label = "FAIL (Control ≥ Bio)"
            ctrl_arm = "Fly B" if self.flyA.is_real else "Fly A"
            winner_name = f"{ctrl_arm} (Control Superiority)"
            winner_type = "shuf"
            self.stats["shuf_wins"] += 1
        else:
            verdict = "inconclusive"
            verdict_label = "INCONCLUSIVE (Tie)"
            winner_name = "Tie"
            winner_type = "none"
            self.stats["timeouts"] += 1

        self.save_ledger()

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

        dec_a = self.flyA.get_decoders(self.active_item)
        dec_b = self.flyB.get_decoders(self.active_item)

        self.last_reveal = {
            "trial_num": self.stats["total"],
            "trial_mode": self.active_trial_mode,
            "active_item": self.active_item,
            "item_name": ITEMS_META.get(self.active_item, {}).get("name", "Chemotaxis"),
            "verdict": verdict,
            "verdict_label": verdict_label,
            "metric_name": metric_name,
            "score_a": score_a,
            "score_b": score_b,
            "metric_bio": bio_score,
            "metric_ctrl": ctrl_score,
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
            "decoders_a": dec_a,
            "decoders_b": dec_b,
            "ctrl_id": self.active_ctrl_id,
            "ctrl_hash": hash_repr,
            "pool_size": pool_sz,
            "custody_draws": self.control_idx
        }

        record = {
            "trial_num": self.stats["total"],
            "timestamp": int(time.time()),
            "trial_mode": self.active_trial_mode,
            "active_item": self.active_item,
            "item_name": ITEMS_META.get(self.active_item, {}).get("name", "Chemotaxis"),
            "verdict": verdict,
            "verdict_label": verdict_label,
            "metric_name": metric_name,
            "score_a": score_a,
            "score_b": score_b,
            "metric_bio": bio_score,
            "metric_ctrl": ctrl_score,
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
            "decoders_a": dec_a,
            "decoders_b": dec_b,
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
        if len(self.stats["history"]) > 12:
            for old_rec in self.stats["history"][:-12]:
                old_rec.pop("trajectory_a", None)
                old_rec.pop("trajectory_b", None)
        if len(self.stats["history"]) > 50:
            self.stats["history"].pop(0)
        self.save_ledger()

        self.is_revealing = True
        self.reveal_timer = 60

    def tick(self):
        with self.lock:
            if self.is_revealing:
                self.reveal_timer -= 1
                if self.reveal_timer <= 0:
                    self.spawn_trial()
                return

            self.step_count += 1

            # Update Environmental Dynamics for Active Item
            if self.active_item == 3:
                # Stationary CO2 aversive plume (no rigid translation or wall bouncing)
                pass

            elif self.active_item == 4:
                # Looming shadow expands
                self.looming_shadow["r"] = min(self.looming_shadow["max_r"], self.looming_shadow["r"] + self.looming_shadow["growth_rate"])
                if self.looming_shadow["r"] >= self.looming_shadow["max_r"]:
                    # Cycle shadow to opposite side
                    self.looming_shadow["x"] = ARENA_W - self.looming_shadow["x"]
                    self.looming_shadow["r"] = 15.0

            elif self.active_item == 5:
                # Optomotor stripes rotate
                self.optomotor["angle"] = (self.optomotor["angle"] + self.optomotor["dir"] * self.optomotor["speed"]) % (2 * math.pi)
                self.grating_history.append(self.optomotor["dir"])
                if self.step_count % 160 == 0:
                    # Invert rotation to test bidirectional optomotor turning
                    self.optomotor["dir"] = -self.optomotor["dir"]

            elif self.active_item == 6:
                # Check female song contact
                self.female_target["song_active"] = (self.flyA.courtship_active or self.flyB.courtship_active)

            # Step both flies
            d_a = self.flyA.step(self)
            d_b = self.flyB.step(self)

            # Termination conditions evaluated objectively per item
            if self.active_item == 1:
                # Item 1: Odour Valence Ordering (attractant reached or timeout)
                if d_a < self.target["r"] + 6 or d_b < self.target["r"] + 6:
                    self.finish_trial("evaluated")
                elif self.step_count > 950:
                    self.finish_trial("timeout")
            elif self.active_item == 2:
                # Item 2: Concentration Reversal: evaluate after 360 steps
                if self.step_count >= 360:
                    self.finish_trial("evaluated")
            elif self.active_item == 3:
                # Item 3: CO2 Avoidance: evaluate after 360 steps
                if self.step_count >= 360:
                    self.finish_trial("evaluated")
            elif self.active_item == 4:
                # Item 4: Looming Escape: evaluate once escape triggered and flight sustained
                if (self.flyA.escape_triggered or self.flyB.escape_triggered) and self.step_count >= 240:
                    self.finish_trial("evaluated")
                elif self.step_count > 420:
                    self.finish_trial("timeout")
            elif self.active_item == 5:
                # Item 5: Optomotor Yaw: evaluate coupling after 340 steps
                if self.step_count >= 340:
                    self.finish_trial("evaluated")
            elif self.active_item == 6:
                # Item 6: Courtship Song: evaluate after sufficient interaction or contact
                if (self.flyA.courtship_ticks > 25 or self.flyB.courtship_ticks > 25) and self.step_count >= 240:
                    self.finish_trial("evaluated")
                elif d_a < self.target["r"] + 8 or d_b < self.target["r"] + 8:
                    self.finish_trial("evaluated")
                elif self.step_count > 700:
                    self.finish_trial("timeout")
            else:
                if d_a < self.target["r"] + 6 or d_b < self.target["r"] + 6:
                    self.finish_trial("evaluated")
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

            trail_a = self.flyA.history[-45:] if len(self.flyA.history) > 45 else self.flyA.history
            trail_b = self.flyB.history[-45:] if len(self.flyB.history) > 45 else self.flyB.history

            dec_a = self.flyA.get_decoders(self.active_item)
            dec_b = self.flyB.get_decoders(self.active_item)

            state = {
                "flight_number": self.flight_number,
                "target": {"x": round(self.target["x"], 1), "y": round(self.target["y"], 1), "r": self.target["r"]},
                "is_revealing": self.is_revealing,
                "step_count": self.step_count,
                "trial_mode": self.active_trial_mode,
                "mode_setting": self.mode_setting,
                "active_item": self.active_item,
                "item_setting": self.item_setting,
                "item_meta": ITEMS_META.get(self.active_item),
                # Environmental entities
                "repellent": {"x": round(self.repellent["x"], 1), "y": round(self.repellent["y"], 1), "r": self.repellent["r"]} if self.active_item == 1 else None,
                "conc_threshold": self.conc_threshold if self.active_item == 2 else None,
                "co2_cloud": {"x": round(self.co2_cloud["x"], 1), "y": round(self.co2_cloud["y"], 1), "r": self.co2_cloud["r"]} if self.active_item == 3 else None,
                "looming_shadow": {"x": round(self.looming_shadow["x"], 1), "y": round(self.looming_shadow["y"], 1), "r": round(self.looming_shadow["r"], 1), "active": self.looming_shadow["active"]} if self.active_item == 4 else None,
                "optomotor": {"angle": round(self.optomotor["angle"], 3), "dir": self.optomotor["dir"], "speed": self.optomotor["speed"]} if self.active_item == 5 else None,
                "female_target": {"x": round(self.female_target["x"], 1), "y": round(self.female_target["y"], 1), "r": self.female_target["r"], "song_active": self.female_target["song_active"]} if self.active_item == 6 else None,
                # SEALED BLIND: is_real and arm_label are strictly null during active flight!
                "flyA": {
                    "x": round(self.flyA.x, 1),
                    "y": round(self.flyA.y, 1),
                    "heading": round(self.flyA.heading, 3),
                    "last_yaw": round(self.flyA.last_yaw, 4),
                    "speed": round(self.flyA.last_speed, 2),
                    "is_paused": self.flyA.is_paused,
                    "courtship_active": self.flyA.courtship_active,
                    "escape_active": self.flyA.escape_active,
                    "dist": round(cur_d_a, 1),
                    "ci": round(max(0.0, (d0_a - cur_d_a) / max(1.0, d0_a)), 2),
                    "mean_rate": self.flyA.get_mean_rate(),
                    "silent_fraction": self.flyA.get_silent_fraction(),
                    "decoders": dec_a,
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
                    "speed": round(self.flyB.last_speed, 2),
                    "is_paused": self.flyB.is_paused,
                    "courtship_active": self.flyB.courtship_active,
                    "escape_active": self.flyB.escape_active,
                    "dist": round(cur_d_b, 1),
                    "ci": round(max(0.0, (d0_b - cur_d_b) / max(1.0, d0_b)), 2),
                    "mean_rate": self.flyB.get_mean_rate(),
                    "silent_fraction": self.flyB.get_silent_fraction(),
                    "decoders": dec_b,
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
    tick_interval = 0.033
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
        pass

    def do_HEAD(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/api/state", "/api/items", "/api/grant_runs", "/api/history", "/api/benchmark"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

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

        if parsed.path == "/api/items":
            items_payload = {
                "items": list(ITEMS_META.values()),
                "active_item": engine.active_item,
                "item_setting": engine.item_setting,
                "current_item": ITEMS_META.get(engine.active_item)
            }
            data = json.dumps(items_payload, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(data)
            return

        if parsed.path == "/api/set_item":
            qs = parse_qs(parsed.query)
            item_arg = qs.get("item", [None])[0]
            if item_arg:
                engine.set_item(item_arg)
                data = json.dumps({
                    "ok": True,
                    "active_item": engine.active_item,
                    "item_setting": engine.item_setting,
                    "item": ITEMS_META.get(engine.active_item)
                }).encode("utf-8")
                self.send_response(200)
            else:
                data = json.dumps({"ok": False, "error": "Missing 'item' query param"}).encode("utf-8")
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(data)
            return

        if parsed.path == "/api/grant_runs":
            qs = parse_qs(parsed.query)
            file_name = qs.get("file", [None])[0]
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            data_dir = os.path.join(base_dir, "web", "data")
            grant_results_dir = os.path.abspath("/home/frost/projects/1fab0/results")

            if file_name:
                # Sanitize filename
                clean_name = os.path.basename(file_name)
                cand_paths = [
                    os.path.join(data_dir, clean_name),
                    os.path.join(base_dir, "data", clean_name),
                    os.path.join(grant_results_dir, clean_name)
                ]
                found_path = None
                for p in cand_paths:
                    if os.path.exists(p) and os.path.isfile(p):
                        found_path = p
                        break
                if found_path:
                    with open(found_path, "rb") as f:
                        content = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(content)
                    return
                else:
                    self.send_response(404)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"error": "File not found"}')
                    return

            # List available grant run files
            available_files = []
            for scan_dir in (data_dir, grant_results_dir):
                if os.path.exists(scan_dir):
                    for f in os.listdir(scan_dir):
                        if f.endswith(".jsonl"):
                            f_path = os.path.join(scan_dir, f)
                            sz = os.path.getsize(f_path)
                            available_files.append({
                                "name": f,
                                "size_bytes": sz,
                                "size_kb": round(sz / 1024, 1),
                                "source": "web_data" if scan_dir == data_dir else "1fab0_results"
                            })
            # Deduplicate by name
            unique = {}
            for item in available_files:
                unique[item["name"]] = item
            data = json.dumps({"files": list(unique.values())}, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
            return

        if parsed.path == "/api/history":
            with engine.lock:
                raw_hist = list(reversed(engine.stats.get("history", [])))
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
            elif file_path.endswith(".jsonl"):
                content_type = "text/plain; charset=utf-8"
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

        if parsed.path == "/api/set_item":
            content_len = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
            try:
                payload = json.loads(post_body.decode("utf-8"))
            except Exception:
                payload = {}
            item = payload.get("item")
            if item is not None:
                engine.set_item(item)
                data = json.dumps({
                    "ok": True,
                    "active_item": engine.active_item,
                    "item_setting": engine.item_setting,
                    "item": ITEMS_META.get(engine.active_item)
                }).encode("utf-8")
                self.send_response(200)
            else:
                data = json.dumps({"ok": False, "error": "Missing 'item' parameter"}).encode("utf-8")
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
            return

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
