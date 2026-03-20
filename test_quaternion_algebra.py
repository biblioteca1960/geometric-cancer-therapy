#!/usr/bin/env python3
"""
test_quaternion_algebra.py

Unit tests for quaternion algebra operations.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quaternion_algebra import (
    qmul, qconj, qnorm, qnormalize, qinv,
    Q_ONE, Q_I, Q_J, Q_K
)

class TestQuaternionAlgebra(unittest.TestCase):
    
    def test_basis_multiplication(self):
        """Test quaternion basis multiplication rules."""
        # i·i = -1
        self.assertTrue(np.allclose(qmul(Q_I, Q_I), -Q_ONE))
        # j·j = -1
        self.assertTrue(np.allclose(qmul(Q_J, Q_J), -Q_ONE))
        # k·k = -1
        self.assertTrue(np.allclose(qmul(Q_K, Q_K), -Q_ONE))
        # i·j = k
        self.assertTrue(np.allclose(qmul(Q_I, Q_J), Q_K))
        # j·k = i
        self.assertTrue(np.allclose(qmul(Q_J, Q_K), Q_I))
        # k·i = j
        self.assertTrue(np.allclose(qmul(Q_K, Q_I), Q_J))
    
    def test_conjugation(self):
        """Test quaternion conjugation."""
        q = np.array([1, 2, 3, 4])
        conj = qconj(q)
        self.assertEqual(conj[0], 1)
        self.assertEqual(conj[1], -2)
        self.assertEqual(conj[2], -3)
        self.assertEqual(conj[3], -4)
    
    def test_norm(self):
        """Test quaternion norm calculation."""
        # |1| = 1
        self.assertAlmostEqual(qnorm(Q_ONE), 1.0)
        # |i| = 1
        self.assertAlmostEqual(qnorm(Q_I), 1.0)
        # |j| = 1
        self.assertAlmostEqual(qnorm(Q_J), 1.0)
        # |k| = 1
        self.assertAlmostEqual(qnorm(Q_K), 1.0)
        # |a + bi + cj + dk| = sqrt(a²+b²+c²+d²)
        q = np.array([1, 2, 3, 4])
        self.assertAlmostEqual(qnorm(q), np.sqrt(1+4+9+16))
    
    def test_normalization(self):
        """Test quaternion normalization."""
        q = np.array([1, 2, 3, 4])
        q_norm = qnormalize(q)
        self.assertAlmostEqual(qnorm(q_norm), 1.0)
    
    def test_inverse(self):
        """Test quaternion inverse."""
        q = np.array([1, 2, 3, 4])
        q_inv = qinv(q)
        # q * q^{-1} = 1
        self.assertTrue(np.allclose(qmul(q, q_inv), Q_ONE, atol=1e-10))

if __name__ == '__main__':
    unittest.main()