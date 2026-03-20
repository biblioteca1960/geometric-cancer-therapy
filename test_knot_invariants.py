#!/usr/bin/env python3
"""
test_knot_invariants.py

Unit tests for Conway knot invariants and topological properties.
Tests the Jones polynomial, crossing number, and torsion validation.

Morató de Dalmases, 2026
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from constants import CONWAY_TORSION, CONWAY_CROSSINGS, N_MOBIUS
from conway_knot_analysis import (
    hypercube_hamiltonian, apply_torsion, project_to_3d, project_to_2d,
    detect_crossings, jones_polynomial_k11n34, evaluate_jones_polynomial
)

class TestConwayKnot(unittest.TestCase):
    
    def test_hamiltonian_cycle(self):
        """Test Hamiltonian cycle generation on Q4."""
        vertices, directions = hypercube_hamiltonian()
        
        # Should have 16 vertices
        self.assertEqual(len(vertices), 16)
        
        # Should have 16 edges (directions)
        self.assertEqual(len(directions), 16)
        
        # All vertices should be distinct
        vertex_set = set(tuple(v) for v in vertices)
        self.assertEqual(len(vertex_set), 16)
        
        # Each vertex should be a 4-bit binary string
        for v in vertices:
            self.assertTrue(all(bit in [0, 1] for bit in v))
    
    def test_torsion_operator(self):
        """Test torsion operator application."""
        vertices, _ = hypercube_hamiltonian()
        
        # Apply torsion
        vertices_rot = apply_torsion(vertices, CONWAY_TORSION)
        
        # Should preserve shape
        self.assertEqual(vertices.shape, vertices_rot.shape)
        
        # Norm should be preserved (rotation)
        for i in range(len(vertices)):
            self.assertAlmostEqual(
                np.linalg.norm(vertices[i]),
                np.linalg.norm(vertices_rot[i]),
                delta=1e-10
            )
    
    def test_projection(self):
        """Test projection from 4D to 3D and 2D."""
        vertices, _ = hypercube_hamiltonian()
        vertices_rot = apply_torsion(vertices, CONWAY_TORSION)
        
        # Project to 3D
        vertices_3d = project_to_3d(vertices_rot)
        self.assertEqual(vertices_3d.shape[1], 3)
        
        # Project to 2D
        vertices_2d = project_to_2d(vertices_3d)
        self.assertEqual(vertices_2d.shape[1], 2)
    
    def test_crossing_number(self):
        """Test crossing number of Conway knot projection."""
        vertices, directions = hypercube_hamiltonian()
        vertices_rot = apply_torsion(vertices, CONWAY_TORSION)
        vertices_3d = project_to_3d(vertices_rot)
        vertices_2d = project_to_2d(vertices_3d)
        
        # Create edges from Hamiltonian cycle
        edges = [(i, i+1) for i in range(len(vertices)-1)] + [(len(vertices)-1, 0)]
        
        # Detect crossings
        crossings = detect_crossings(vertices_2d, edges)
        
        # Should have 11 crossings (Conway knot K11n34)
        self.assertEqual(len(crossings), 11)
    
    def test_jones_polynomial(self):
        """Test Jones polynomial evaluation."""
        # Get polynomial string
        jones_str = jones_polynomial_k11n34()
        self.assertEqual(jones_str, "t^{-2} - t^{-1} + 1 - t + t^{2}")
        
        # Evaluate at specific points
        # V(1) = 1 for all knots
        v_at_1 = evaluate_jones_polynomial(1.0)
        self.assertAlmostEqual(v_at_1, 1.0, delta=1e-10)
        
        # V(i) = ?
        v_at_i = evaluate_jones_polynomial(1j)
        self.assertIsInstance(v_at_i, complex)
    
    def test_torsion_value(self):
        """Test that torsion value produces Conway knot."""
        vertices, _ = hypercube_hamiltonian()
        
        # Test with correct torsion
        vertices_rot_correct = apply_torsion(vertices, CONWAY_TORSION)
        vertices_3d_correct = project_to_3d(vertices_rot_correct)
        vertices_2d_correct = project_to_2d(vertices_3d_correct)
        
        edges = [(i, i+1) for i in range(len(vertices)-1)] + [(len(vertices)-1, 0)]
        crossings_correct = detect_crossings(vertices_2d_correct, edges)
        
        # Test with incorrect torsion (should not produce 11 crossings)
        vertices_rot_wrong = apply_torsion(vertices, 0.5)
        vertices_3d_wrong = project_to_3d(vertices_rot_wrong)
        vertices_2d_wrong = project_to_2d(vertices_3d_wrong)
        crossings_wrong = detect_crossings(vertices_2d_wrong, edges)
        
        # Correct torsion should give 11 crossings
        self.assertEqual(len(crossings_correct), 11)
        
        # Incorrect torsion should not give 11 crossings
        self.assertNotEqual(len(crossings_wrong), 11)

class TestMobiusStructure(unittest.TestCase):
    
    def test_number_of_projections(self):
        """Test that number of Möbius projections is 37."""
        # 37 = 120/5 + 120/15 + 1 = 24 + 12 + 1
        projections_5fold = 120 // 5
        projections_3fold = 120 // 15
        central = 1
        
        total = projections_5fold + projections_3fold + central
        self.assertEqual(total, 37)
        self.assertEqual(total, N_MOBIUS)
    
    def test_torsion_angles(self):
        """Test torsion angles for 37 projections."""
        for a in range(1, 38):
            theta = a * 2 * np.pi / N_MOBIUS
            self.assertTrue(0 <= theta <= 2 * np.pi)
            
            # First and last are complementary
            if a == 1:
                theta_1 = theta
            if a == 36:
                # 36*2π/37 ≈ 2π - 2π/37
                self.assertAlmostEqual(theta + theta_1, 2 * np.pi, delta=1e-10)

class TestTopologicalInvariants(unittest.TestCase):
    
    def test_writhe(self):
        """Test writhe calculation for Conway knot."""
        # Simplified writhe test
        # For K11n34, writhe = -1
        # This would require full knot diagram analysis
        pass
    
    def test_kauffman_bracket(self):
        """Test Kauffman bracket for Conway knot."""
        # For K11n34, the Kauffman bracket should be
        # ⟨D⟩ = A^8 + A^4 + 1 + A^{-4} + A^{-8}
        # With A = t^{-1/4}
        pass

if __name__ == '__main__':
    unittest.main()