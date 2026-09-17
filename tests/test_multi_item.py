"""
Unit tests for Quire's Grant 1FAB0 6 Behavioral Test Items & Dual Decoders in Connectome Arena.
"""

import unittest
import math
import random
from src.server import ITEMS_META, ArenaEngine, FlyAgent, create_biological_matrix


class TestMultiItemAndDualDecoders(unittest.TestCase):
    def setUp(self):
        self.W_bio = create_biological_matrix()

    def test_items_metadata_completeness(self):
        """All 6 behavioral test items are present and aligned with Grant 1FAB0 battery-v4."""
        self.assertEqual(len(ITEMS_META), 6)
        for i in range(1, 7):
            self.assertIn(i, ITEMS_META)
            meta = ITEMS_META[i]
            self.assertIn("name", meta)
            self.assertIn("subtitle", meta)
            self.assertIn("readout", meta)
            self.assertIn("predicate", meta)
            self.assertIn("stimuli", meta)
            self.assertIn("source", meta)

    def test_item1_odour_valence_ordering(self):
        """Item 1: Attractant produces positive delta_hz; repellent proximity engages MDN."""
        engine = ArenaEngine()
        engine.set_item(1)
        fly = FlyAgent(300, 300, 0.0, self.W_bio, True, "Fly A", item=1)

        # Place target (attractant) ahead and repellent far away
        engine.target = {"x": 500, "y": 300, "r": 18}
        engine.repellent = {"x": 50, "y": 50, "r": 18}

        for _ in range(30):
            fly.step(engine)

        dec = fly.get_decoders(1)
        self.assertGreater(dec["quire"]["rate_dnp09"] if "rate_dnp09" in dec["quire"] else dec["quire"]["dnp09"], 0.0)
        self.assertGreater(dec["quire"]["delta_hz"], 0.0)
        self.assertGreater(dec["strata"]["ci"], -0.2)

    def test_item2_concentration_reversal(self):
        """Item 2: Concentration reversal recruits DM5 in high-concentration core."""
        engine = ArenaEngine()
        engine.set_item(2)
        fly_low = FlyAgent(450, 300, 0.0, self.W_bio, True, "Fly Low", item=2)
        fly_high = FlyAgent(640, 300, 0.0, self.W_bio, True, "Fly High", item=2)
        engine.target = {"x": 650, "y": 300, "r": 18}

        for _ in range(20):
            fly_low.step(engine)
            fly_high.step(engine)

        dec_low = fly_low.get_decoders(2)
        dec_high = fly_high.get_decoders(2)

        # In outer low-concentration zone: forward walking attraction
        self.assertGreater(dec_low["quire"]["delta_hz"], 0.0)
        # In inner high-concentration saturation core: MDN elevated, lower delta_hz
        self.assertGreater(dec_high["quire"]["mdn"], dec_low["quire"]["mdn"])
        self.assertLess(dec_high["quire"]["delta_hz"], dec_low["quire"]["delta_hz"])

    def test_item2_smooth_avoidance_no_chattering_barrier(self):
        """Item 2: Fly approaching the 110px high-concentration boundary must veer smoothly away without chattering."""
        engine = ArenaEngine()
        engine.set_item(2)
        engine.target = {"x": 650, "y": 300, "r": 18}
        # Start fly 120px away facing target
        fly = FlyAgent(530, 300, 0.0, self.W_bio, True, "Fly A", item=2)

        distances = []
        for _ in range(50):
            d = fly.step(engine)
            distances.append(round(d, 1))

        # Verify no 2-state periodic ping-pong lock (e.g. [111.4, 109.8, 111.4, 109.8])
        tail = distances[-10:]
        unique_tail = set(tail)
        self.assertGreater(len(unique_tail), 2, "Fly must not be trapped in a 2-point chattering barrier loop")

        # Verify the fly executed aversive steering (deviated laterally from y=300 to veer around core)
        lateral_deviation = abs(fly.y - 300.0)
        self.assertGreater(lateral_deviation, 2.0, "Fly must veer laterally away from high-concentration core")

    def test_item3_co2_walking_avoidance(self):
        """Item 3: CO2 cloud encounter triggers moonwalker MDN activation and negative delta_hz."""
        engine = ArenaEngine()
        engine.set_item(3)
        engine.co2_cloud = {"x": 300, "y": 300, "r": 100, "vx": 0.0, "vy": 0.0}
        fly_in_co2 = FlyAgent(305, 305, 0.0, self.W_bio, True, "Fly CO2", item=3)

        for _ in range(25):
            fly_in_co2.step(engine)

        dec = fly_in_co2.get_decoders(3)
        self.assertGreaterEqual(dec["quire"]["mdn"], 35.0)
        self.assertLess(dec["quire"]["delta_hz"], 0.0)

    def test_item3_stationary_plume_and_bilateral_tropotaxis(self):
        """Item 3: CO2 plume must remain a stationary olfactory field, not an unphysical drifting object."""
        engine = ArenaEngine()
        engine.set_item(3)
        initial_x = engine.co2_cloud["x"]
        initial_y = engine.co2_cloud["y"]

        # Run multiple engine ticks
        for _ in range(50):
            engine.tick()

        # Invariant: Olfactory diffusion field does not drift like a solid puck
        self.assertEqual(engine.co2_cloud["x"], initial_x)
        self.assertEqual(engine.co2_cloud["y"], initial_y)
        self.assertEqual(engine.co2_cloud["vx"], 0.0)
        self.assertEqual(engine.co2_cloud["vy"], 0.0)

    def test_item4_looming_visual_escape(self):
        """Item 4: Expanding dark shadow triggers Giant Fibre DNp01 activation."""
        engine = ArenaEngine()
        engine.set_item(4)
        engine.looming_shadow = {"x": 300, "y": 300, "r": 90, "max_r": 150, "active": True}
        fly = FlyAgent(310, 310, 0.0, self.W_bio, True, "Fly Loom", item=4)

        for _ in range(20):
            fly.step(engine)

        dec = fly.get_decoders(4)
        self.assertGreaterEqual(dec["quire"]["dnp01"], 90.0)
        self.assertTrue(dec["quire"]["escape_active"])

    def test_item4_wall_reflection_during_escape(self):
        """Item 4: Escaping fly reflecting off the arena wall must not get stuck vibrating against the wall."""
        engine = ArenaEngine()
        engine.set_item(4)
        engine.looming_shadow = {"x": 720, "y": 300, "r": 60, "max_r": 150, "active": True}
        # Start fly close to right wall (x=770, arena boundary is at 788)
        fly = FlyAgent(770, 300, 0.0, self.W_bio, True, "Fly Loom", item=4)

        x_coords = []
        for _ in range(40):
            fly.step(engine)
            x_coords.append(round(fly.x, 1))

        # Must not remain pinned at 788.0 for 20+ steps
        pinned_count = sum(1 for x in x_coords if x >= 787.5)
        self.assertLess(pinned_count, 15, "Fly must deflect and reflect off wall, not remain pinned vibrating against it")

    def test_item5_optomotor_yaw_steering(self):
        """Item 5: Grating rotation direction reverses steer asymmetry sign."""
        engine = ArenaEngine()
        engine.set_item(5)

        # Clockwise rotation
        engine.optomotor = {"angle": 0.0, "dir": 1, "speed": 0.05}
        fly_cw = FlyAgent(400, 300, 0.0, self.W_bio, True, "Fly CW", item=5)
        for _ in range(25):
            fly_cw.step(engine)
        dec_cw = fly_cw.get_decoders(5)

        # Counter-clockwise rotation
        engine.optomotor = {"angle": 0.0, "dir": -1, "speed": 0.05}
        fly_ccw = FlyAgent(400, 300, 0.0, self.W_bio, True, "Fly CCW", item=5)
        for _ in range(25):
            fly_ccw.step(engine)
        dec_ccw = fly_ccw.get_decoders(5)

        # Asymmetry signs must be opposite
        asym_cw = dec_cw["quire"]["steer_asym"]
        asym_ccw = dec_ccw["quire"]["steer_asym"]
        self.assertGreater(asym_cw, 0.0)
        self.assertLess(asym_ccw, 0.0)

    def test_item6_courtship_song_initiation(self):
        """Item 6: Female pheromone contact excites pC1 and pIP10 acoustic song command neurons."""
        engine = ArenaEngine()
        engine.set_item(6)
        engine.female_target = {"x": 400, "y": 300, "r": 25, "song_active": False}
        fly_contact = FlyAgent(408, 302, 0.0, self.W_bio, True, "Fly Male", item=6)

        for _ in range(20):
            fly_contact.step(engine)

        dec = fly_contact.get_decoders(6)
        self.assertGreater(dec["quire"]["pc1"], 25.0)
        self.assertGreater(dec["quire"]["pip10"], 30.0)
        self.assertTrue(dec["quire"]["courtship_active"])

    def test_engine_state_and_switching(self):
        """Engine state accurately serializes items and allows switching."""
        engine = ArenaEngine()
        for it in range(1, 7):
            engine.set_item(it)
            state = engine.get_state()
            self.assertEqual(state["active_item"], it)
            self.assertEqual(state["item_meta"]["id"], it)
            self.assertIn("decoders", state["flyA"])
            self.assertIn("quire", state["flyA"]["decoders"])
            self.assertIn("strata", state["flyA"]["decoders"])
            self.assertIn("speed", state["flyA"])
            self.assertIn("is_paused", state["flyA"])

    def test_item3_recoil_and_reorientation_no_infinite_moonwalking(self):
        """Item 3: Encountering CO2 plume triggers brief recoil step and reorientation away, not infinite moonwalking."""
        engine = ArenaEngine()
        engine.set_item(3)
        engine.co2_cloud = {"x": 400, "y": 300, "r": 105.0}
        # Start fly walking towards the plume from x=275, y=300 facing +x (0 rad)
        fly = FlyAgent(275, 300, 0.0, self.W_bio, True, "Fly Test", item=3)

        speeds = []
        headings = []
        for _ in range(60):
            fly.step(engine)
            speeds.append(fly.last_speed)
            headings.append(fly.heading)

        # Recoil occurs: speed is negative for a brief burst of 2-4 ticks
        negative_speed_count = sum(1 for s in speeds if s < 0)
        self.assertGreater(negative_speed_count, 0, "Fly must execute brief backward recoil step on plume encounter")
        self.assertLessEqual(negative_speed_count, 6, "Recoil must be brief (~50-100ms), not an infinite moonwalk across arena")

        # After recoil, fly reorients away from plume (heading points away from (400, 300))
        # Plume is to the right (+x), so facing away means cos(heading) < 0.2
        post_recoil = headings[12:28]
        avg_cos = sum(math.cos(h) for h in post_recoil) / len(post_recoil)
        self.assertLess(avg_cos, 0.3, "Fly must reorient away from plume rather than facing into plume while moonwalking")

    def test_item5_fly_differentiation_no_clones(self):
        """Item 5: Fly A and Fly B have distinct spawn coordinates, independent initial headings, and distinct coupling."""
        random.seed(42)
        engine = ArenaEngine()
        engine.mode_setting = "arm1"
        engine.set_item(5)
        # Verify spawn separation
        dist_between_flies = math.hypot(engine.flyA.x - engine.flyB.x, engine.flyA.y - engine.flyB.y)
        self.assertGreaterEqual(dist_between_flies, 20.0, "Fly A and Fly B must not spawn overlapping at identical coordinates")

        # Run 60 ticks of optomotor rotation
        for _ in range(60):
            engine.tick()

        # Biological fly should couple to visual rotation, while control does not lock in
        bio_fly = engine.flyA if engine.flyA.is_real else engine.flyB
        ctrl_fly = engine.flyB if engine.flyA.is_real else engine.flyA
        r_bio = bio_fly.compute_optomotor_coupling(engine.grating_history)
        r_ctrl = ctrl_fly.compute_optomotor_coupling(engine.grating_history)
        self.assertGreater(r_bio, r_ctrl, "Biological connectome must show superior optomotor coupling over control twin")

    def test_thigmotaxis_boundary_steering(self):
        """Boundary handling uses biological thigmotaxis (wall-following) without specular angle reflection."""
        fly = FlyAgent(780, 300, 0.0, self.W_bio, True, "Fly Wall", item=1)
        # Fly is near right boundary (MAX_X = 788), heading directly toward it (0 rad)
        # Step fly into wall
        for _ in range(10):
            fly.step(650, 300)

        # Coordinate must remain bounded
        self.assertLessEqual(fly.x, 788.0)
        self.assertGreaterEqual(fly.x, 12.0)
        # Heading must NOT be specular reflection math.pi - 0 = math.pi with 0 y-velocity;
        # Thigmotaxis redirects velocity along wall tangent (sin(heading) != 0)
        self.assertGreater(abs(math.sin(fly.heading)), 0.15, "Thigmotaxis must deflect fly along perimeter tangent")

    def test_bout_pause_state_machine(self):
        """Fly alternates between walking bouts and casting pauses, suppressed during looming escape."""
        fly_normal = FlyAgent(400, 300, 0.0, self.W_bio, True, "Fly Walk", item=1)
        engine_normal = ArenaEngine()
        engine_normal.set_item(1)

        paused_ticks = 0
        for _ in range(80):
            fly_normal.step(engine_normal)
            if fly_normal.is_paused:
                paused_ticks += 1

        self.assertGreater(paused_ticks, 0, "Normal locomotion must include stop-and-go casting pauses")
        self.assertLess(paused_ticks, 40, "Pauses must be brief stops, not perpetual immobilization")

        # Urgency suppression: Looming escape must suppress pauses
        fly_escape = FlyAgent(400, 300, 0.0, self.W_bio, True, "Fly Escape", item=4)
        engine_loom = ArenaEngine()
        engine_loom.set_item(4)
        engine_loom.looming_shadow = {"x": 400, "y": 300, "r": 90, "max_r": 150, "active": True}

        for _ in range(25):
            fly_escape.step(engine_loom)
            self.assertFalse(fly_escape.is_paused, "Pauses must be suppressed during emergency looming escape")

    def test_objective_phenotype_evaluation_no_hardcoding(self):
        """Trial evaluation uses objective empirical phenotype metrics without hardcoded winner bias."""
        engine = ArenaEngine()
        engine.set_item(5)
        # Artificially set higher coupling on Fly A
        engine.flyA.yaw_history = [0.03] * 50
        engine.flyB.yaw_history = [-0.01] * 50
        engine.grating_history = [1] * 50
        engine.flyA.is_real = False  # Make Fly A the control twin
        engine.flyB.is_real = True   # Make Fly B the bio fly

        # Because Fly A (control) has higher coupling with grating, verdict must be FAIL (Ctrl >= Bio)
        engine.finish_trial("evaluated")
        self.assertEqual(engine.last_reveal["verdict"], "fail", "Must evaluate objectively: if control scores higher, verdict is FAIL")
        self.assertIn("FAIL", engine.last_reveal["verdict_label"])
        self.assertNotIn("(Pass)", engine.last_reveal["winner_name"], "Winner name must not say Pass when verdict is FAIL")
        self.assertIn("Control Superiority", engine.last_reveal["winner_name"])

    def test_item2_calibrated_scoring_envelope_vs_wandering(self):
        """Item 2: Fly approaching attractive envelope (d=65) scores positive; wandering fly (d=220) scores 0.0."""
        engine = ArenaEngine()
        engine.set_item(2)
        engine.target = {"x": 400, "y": 300, "r": 18}

        # Fly A: Bio tracks from d0=280 to d=65 (navigating attractive zone without hitting core <45)
        engine.flyA.is_real = True
        engine.flyA.start_x = 400; engine.flyA.start_y = 20  # d0 = 280
        engine.flyA.full_trajectory = [(400, 20), (400, 150), (400, 235)]  # min_d = 65
        engine.flyA.x = 400; engine.flyA.y = 235

        # Fly B: Control twin wanders far away, min_d = 220
        engine.flyB.is_real = False
        engine.flyB.start_x = 400; engine.flyB.start_y = 20  # d0 = 280
        engine.flyB.full_trajectory = [(400, 20), (400, 50), (400, 80)]  # min_d = 220
        engine.flyB.x = 400; engine.flyB.y = 80

        engine.finish_trial("evaluated")
        self.assertGreater(engine.last_reveal["score_a"], 0.5, "Fly in attractive envelope must score positive")
        self.assertEqual(engine.last_reveal["score_b"], 0.0, "Blind wandering fly beyond 180px envelope must score 0.0")
        self.assertEqual(engine.last_reveal["verdict"], "pass", "Bio must pass over wandering control in Item 2")

    def test_item3_co2_avoidance_no_fake_wandering_displacement(self):
        """Item 3: Evading fly (d=105 to 175) beats wandering fly (capped <= 35) and trapped fly (-50)."""
        engine = ArenaEngine()
        engine.set_item(3)
        engine.co2_cloud = {"x": 400, "y": 300, "r": 105.0}

        # Fly A: Bio enters warning perimeter to d=105, flees to d=175 -> score = +70
        engine.flyA.is_real = True
        engine.flyA.full_trajectory = [(400, 100), (400, 195)]  # min_d = 105
        engine.flyA.x = 400; engine.flyA.y = 125  # end_d = 175

        # Fly B: Control stays far away at d=320 -> must be capped at <= 35, NOT get hundreds of fake pixels
        engine.flyB.is_real = False
        engine.flyB.full_trajectory = [(400, 620), (400, 630)]  # min_d = 320
        engine.flyB.x = 400; engine.flyB.y = 650  # end_d = 350

        engine.finish_trial("evaluated")
        self.assertGreaterEqual(engine.last_reveal["score_a"], 70.0)
        self.assertLessEqual(engine.last_reveal["score_b"], 35.0, "Wandering fly must not get hundreds of fake displacement pixels")
        self.assertEqual(engine.last_reveal["verdict"], "pass", "Active evading bio fly must defeat wandering control")

    def test_item4_giant_fibre_connectome_requirement(self):
        """Item 4: Giant Fibre DNp01 escape requires biological connectome drive."""
        engine = ArenaEngine()
        engine.set_item(4)
        engine.looming_shadow = {"x": 400, "y": 100, "r": 30, "active": True}

        # Bio fly with biological matrix
        bio_fly = FlyAgent(400, 180, 0.0, self.W_bio, True, "Bio", item=4)
        # Control fly with zero/scrambled matrix
        W_ctrl = [[0.0] * 25 for _ in range(25)]
        ctrl_fly = FlyAgent(400, 180, 0.0, W_ctrl, False, "Ctrl", item=4)

        for _ in range(15):
            bio_fly.step(engine)
            ctrl_fly.step(engine)

        self.assertGreater(bio_fly.rate_dnp01, ctrl_fly.rate_dnp01, "Biological connectome must produce superior DNp01 response to looming")

    def test_item6_courtship_requires_functional_pathway(self):
        """Item 6: Female proximity triggers courtship song only with functional biological drive."""
        engine = ArenaEngine()
        engine.set_item(6)
        engine.female_target = {"x": 400, "y": 300, "r": 25, "song_active": False}

        bio_fly = FlyAgent(410, 300, 0.0, self.W_bio, True, "Bio", item=6)
        W_ctrl = [[0.0] * 25 for _ in range(25)]
        ctrl_fly = FlyAgent(410, 300, 0.0, W_ctrl, False, "Ctrl", item=6)

        for _ in range(10):
            bio_fly.step(engine)
            ctrl_fly.step(engine)

        self.assertTrue(bio_fly.courtship_active, "Bio fly must activate courtship when near female")
        self.assertFalse(ctrl_fly.courtship_active, "Control fly without functional pathway must not activate courtship display")


if __name__ == '__main__':
    unittest.main()


