#!/usr/bin/env python3
"""
test_peptide_folding.py

Unit tests for peptide folding analysis.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stress_test_amyloid import analyze_folding, compare_mutations

class TestPeptideFolding(unittest.TestCase):
    
    def test_amyloid_critical_region(self):
        """Test amyloid-β critical region analysis."""
        analysis = analyze_folding("KLVFFAE", crowding=0.3)
        
        # Norm should be close to 0.5 (Z0 point)
        self.assertAlmostEqual(analysis.norm, 0.484, delta=0.05)
        
        # Should be classified as METASTABLE (Z0 point)
        self.assertIn("METASTABLE", analysis.stability)
    
    def test_alpha_helix_stability(self):
        """Test alpha helix stability."""
        analysis = analyze_folding("A" * 20)
        
        # Norm should be close to 1
        self.assertAlmostEqual(analysis.norm, 1.0, delta=0.1)
        
        # Should be STABLE
        self.assertIn("STABLE", analysis.stability)
    
    def test_beta_sheet_stability(self):
        """Test beta sheet stability."""
        analysis = analyze_folding("VKVKVKVKVK" * 2)
        
        # Norm should be close to sqrt(3) ≈ 1.732
        self.assertAlmostEqual(analysis.norm, 1.732, delta=0.1)
        
        # Should be STABLE
        self.assertIn("STABLE", analysis.stability)
    
    def test_mutation_analysis(self):
        """Test mutation effect prediction."""
        wt = "KLVFFAE"
        mutant = "KLVFGFE"  # A21G mutation
        
        comparison = compare_mutations(wt, mutant)
        
        # Mutation should increase pathogenicity
        self.assertTrue(comparison['is_pathogenic'])
        
        # Norm should change
        self.assertNotEqual(comparison['wildtype_norm'], comparison['mutant_norm'])

if __name__ == '__main__':
    unittest.main()