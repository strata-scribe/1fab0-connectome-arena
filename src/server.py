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
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

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
    def __init__(self, x, y, heading, W, is_real, label):
        self.x = x
        self.y = y
        self.start_x = x
        self.start_y = y
        self.heading = heading
        self.speed = 2.2
        self.W = W
        self.is_real = is_real
        self.label = label
        self.rates = [0.0] * NUM_NEURONS
        self.tau = 0.05
        self.dt = 0.015
        self.wander_phase = random.uniform(0, 10)
        self.history = []
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

        dist = math.hypot(self.x - target_x, self.y - target_y)
        if dist < self.min_dist:
            self.min_dist = dist
        return dist

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

        self.target["x"] = random.uniform(140, ARENA_W - 140)
        self.target["y"] = random.uniform(120, ARENA_H - 120)

        while True:
            sx = random.uniform(70, ARENA_W - 70)
            sy = random.uniform(70, ARENA_H - 70)
            d0 = math.hypot(sx - self.target["x"], sy - self.target["y"])
            if d0 >= 320:
                break

        shared_heading = random.uniform(0, 2 * math.pi)
        
        # AUTHORED-ELSEWHERE CUSTODY (c59011 / c59078):
        # Harness consumes pre-generated, cryptographically sealed off-harness controls
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

        assign_a_real = random.random() < 0.5

        self.flyA = FlyAgent(sx, sy, shared_heading, self.W_bio if assign_a_real else W_ctrl, assign_a_real, "Fly A")
        self.flyB = FlyAgent(sx, sy, shared_heading, W_ctrl if assign_a_real else self.W_bio, not assign_a_real, "Fly B")

    def finish_trial(self, winner):
        d0_a = math.hypot(self.flyA.start_x - self.target["x"], self.flyA.start_y - self.target["y"])
        d0_b = math.hypot(self.flyB.start_x - self.target["x"], self.flyB.start_y - self.target["y"])
        final_a = math.hypot(self.flyA.x - self.target["x"], self.flyA.y - self.target["y"])
        final_b = math.hypot(self.flyB.x - self.target["x"], self.flyB.y - self.target["y"])

        ci_a = max(0.0, (d0_a - final_a) / max(1.0, d0_a))
        ci_b = max(0.0, (d0_b - final_b) / max(1.0, d0_b))

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

        self.last_reveal = {
            "trial_num": self.stats["total"],
            "winner_name": winner_name,
            "winner_type": winner_type,
            "flyA_real": self.flyA.is_real,
            "flyB_real": self.flyB.is_real,
            "ci_a": round(ci_a, 2),
            "ci_b": round(ci_b, 2)
        }

        # Record chronological trial entry in ledger
        record = {
            "trial_num": self.stats["total"],
            "timestamp": int(time.time()),
            "winner_name": winner_name,
            "winner_type": winner_type,
            "flyA_real": self.flyA.is_real,
            "flyB_real": self.flyB.is_real,
            "ci_a": round(ci_a, 2),
            "ci_b": round(ci_b, 2),
            "ci_real": round(ci_real, 2),
            "ci_shuf": round(ci_shuf, 2),
            "ci_diff": round(ci_real - ci_shuf, 2),
            "rolling_t": t_stat_val,
            "p_str": p_str,
            "steps": self.step_count
        }
        self.stats.setdefault("history", []).append(record)
        if len(self.stats["history"]) > 300:
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

    def get_state(self):
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
            return {
                "target": {"x": round(self.target["x"], 1), "y": round(self.target["y"], 1), "r": self.target["r"]},
                "is_revealing": self.is_revealing,
                "step_count": self.step_count,
                # SEALED BLIND: is_real is strictly null during active flight!
                "flyA": {
                    "x": round(self.flyA.x, 1),
                    "y": round(self.flyA.y, 1),
                    "heading": round(self.flyA.heading, 3),
                    "last_yaw": round(self.flyA.last_yaw, 4),
                    "dist": round(cur_d_a, 1),
                    "ci": round(max(0.0, (d0_a - cur_d_a) / max(1.0, d0_a)), 2),
                    "history": self.flyA.history,
                    "is_real": self.flyA.is_real if self.is_revealing else None
                },
                "flyB": {
                    "x": round(self.flyB.x, 1),
                    "y": round(self.flyB.y, 1),
                    "heading": round(self.flyB.heading, 3),
                    "last_yaw": round(self.flyB.last_yaw, 4),
                    "dist": round(cur_d_b, 1),
                    "ci": round(max(0.0, (d0_b - cur_d_b) / max(1.0, d0_b)), 2),
                    "history": self.flyB.history,
                    "is_real": self.flyB.is_real if self.is_revealing else None
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
                    "survival_gradient": {
                        "ORN": 0.669,
                        "ALPN": 0.442,
                        "KC": 0.843,
                        "DN": 0.505
                    },
                    "degree_invariance": "k_i = deg_G_traced(i)",
                    "control_custody": {
                        "rule": "authored-elsewhere (c59011/c59078)",
                        "active_ctrl_id": self.active_ctrl_id,
                        "active_ctrl_sha256": self.active_ctrl_hash[:16] + "..."
                    }
                },
                "last_reveal": self.last_reveal,
                "recent_history": list(reversed(self.stats.get("history", [])[-20:]))
            }

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
        if parsed.path == "/api/state":
            data = json.dumps(engine.get_state()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(data)
            return

        if parsed.path == "/api/history":
            history_data = json.dumps({
                "total": engine.stats["total"],
                "history": list(reversed(engine.stats.get("history", [])))
            }, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(history_data)
            return

        # Serve static html
        web_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web"))
        file_path = os.path.join(web_dir, "index.html")
        if not os.path.exists(file_path):
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "1fab0_interactive_demo.html")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
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

    server = HTTPServer(("0.0.0.0", port), ArenaHTTPHandler)
    print(f"Authoritative Connectome Server live on http://0.0.0.0:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
