import unittest

import simulate_2026_season as sim2026


class Simulate2026SeasonTests(unittest.TestCase):
    def test_find_consensus_picks_nearest_tradable_anchor(self):
        strike, price = sim2026.find_consensus({7.5: 0.64, 8.5: 0.52, 9.5: 0.43})
        self.assertEqual(strike, 8.5)
        self.assertEqual(price, 0.52)

    def test_find_consensus_falls_back_to_closest_to_fifty(self):
        strike, price = sim2026.find_consensus({7.5: 0.61, 8.5: 0.57, 9.5: 0.54})
        self.assertEqual(strike, 9.5)
        self.assertEqual(price, 0.54)


if __name__ == "__main__":
    unittest.main()
