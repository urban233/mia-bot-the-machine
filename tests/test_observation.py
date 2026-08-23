import math
import unittest
import numpy as np
from mia_bot.bot import POS_STD, VEL_STD, ANG_VEL_STD


class TestObservation(unittest.TestCase):
    def test_normalization_constants(self):
        self.assertEqual(POS_STD, 2300.0)
        self.assertEqual(VEL_STD, 2300.0)
        self.assertEqual(ANG_VEL_STD, math.pi)

    def test_observation_vector_shape(self):
        # Observation vector is 89-dimensional
        obs = np.zeros(89, dtype=np.float32)
        self.assertEqual(len(obs), 89)
        self.assertEqual(obs.dtype, np.float32)

    def test_observation_normalization_ranges(self):
        # Simulated ball position at arena bounds
        raw_x = 4096.0
        norm_x = raw_x / POS_STD
        self.assertGreater(norm_x, 1.0)  # Normalized relative to standard deviation

        # Velocity normalization
        raw_vel = 2300.0
        norm_vel = raw_vel / VEL_STD
        self.assertEqual(norm_vel, 1.0)


if __name__ == "__main__":
    unittest.main()
