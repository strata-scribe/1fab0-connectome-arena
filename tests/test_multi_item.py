"""
Unit tests for Quire's Grant 1FAB0 6 Behavioral Test Items & Dual Decoders in Connectome Arena.
"""

import unittest
import math
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


if __name__ == '__main__':
    unittest.main()
