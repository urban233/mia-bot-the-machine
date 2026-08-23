import math
import unittest
import numpy as np
from mia_bot.bot import get_rotation_matrix


class TestMath(unittest.TestCase):
    def test_rotation_matrix_zero_angles(self):
        rot = get_rotation_matrix(0.0, 0.0, 0.0)
        expected = np.eye(3, dtype=np.float32).flatten()
        np.testing.assert_allclose(rot, expected, atol=1e-5)

    def test_rotation_matrix_orthogonality():
        pass

    def test_rotation_matrix_orthogonality(self):
        pitch, yaw, roll = 0.3, -0.7, 1.2
        rot = get_rotation_matrix(pitch, yaw, roll).reshape(3, 3)

        # R * R^T should be Identity
        identity = np.dot(rot, rot.T)
        np.testing.assert_allclose(identity, np.eye(3), atol=1e-5)

        # Determinant of proper rotation matrix is +1
        det = np.linalg.det(rot)
        self.assertTrue(math.isclose(det, 1.0, rel_tol=1e-5))

    def test_rotation_matrix_shape(self):
        rot = get_rotation_matrix(0.1, 0.2, 0.3)
        self.assertEqual(rot.shape, (9,))
        self.assertEqual(rot.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
