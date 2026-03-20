#!/usr/bin/env python3
"""
05_nras_hras_simulation.py

Geometric analysis of NRAS and HRAS mutations for melanoma, leukemia, and bladder cancer.

Morató de Dalmases, 2026
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from constants import DELTA_0, PHI, N_MOBIUS, ETA_GAMMA
from quaternion_algebra import qmul, qnorm, product_sequence
from kras_g12d_simulation import compute_geometric_efficacy, compute_binding_energy
from braf_v600e_simulation import compute_torsion_from_mutation

# ============================================================================
# AMINO ACID QUATERNIONS
# ============================================================================

AMINO_ACIDS = {
    'G': np.array([0.125, 0.125, 0.125, 0.125]),  # Glycine
    'D': np.array([0.375, 0.125, 0.250, 0.125]),  # Aspartic acid
    'V': np.array([0.500, 0.500, 0.125, 0.500]),  # Valine
    'Q': np.array([0.500, 0.375, 0.250, 0.375]),  # Glutamine
    'K': np.array([0.625, 0.500, 0.375, 0.500]),  # Lysine
    'L': np.array([0.625, 0.250, 0.250, 0.250]),  # Leucine
    'R': np.array([0.375, 0.375, 0.375, 0.375]),  # Arginine
    'A': np.array([0.125, 0.250, 0.125, 0.250]),  # Alanine
    'N': np.array([0.250, 0.125, 0.375, 0.125]),  # Asparagine
}

# ============================================================================
# MUTATION DATA
# ============================================================================

@dataclass
class MutationInfo:
    """Information about an oncogenic mutation."""
    protein: str
    mutation: str
    wt_aa: str
    mut_aa: str
    cancer_type: str
    torsion: float = 0.0
    gei: float = 0.0
    binding_energy: float = 0.0
    tap_sequence: str = ""

# Known mutations
MUTATIONS = [
    MutationInfo("NRAS", "Q61K", "Q", "K", "Melanoma, Leukemia"),
    MutationInfo("NRAS", "Q61L", "Q", "L", "Melanoma"),
    MutationInfo("NRAS", "Q61R", "Q", "R", "Melanoma"),
    MutationInfo("HRAS", "G12V", "G", "V", "Bladder Cancer"),
    MutationInfo("HRAS", "G12D", "G", "D", "Bladder Cancer"),
    MutationInfo("HRAS", "Q61L", "Q", "L", "Head and Neck Cancer"),
]

# ============================================================================
# TAP DESIGN FOR NRAS/HRAS
# ============================================================================

def design_tap_for_mutation(torsion: float) -> List[str]:
    """
    Design Nanopor-Tap sequence for a given pathological torsion.
    
    This is a simplified design algorithm. In practice, this would use
    the Hamiltonian cycle optimization.
    
    Args:
        torsion: Pathological torsion in rad/Å
    
    Returns:
        List of amino acids for the Tap sequence
    """
    # Base Tap sequence (scaled by torsion ratio)
    base_sequence = ['C', 'Y', 'G', 'D', 'W', 'C']
    base_torsion = 0.34  # KRAS G12D torsion
    
    scale_factor = torsion / base_torsion
    
    # Adjust sequence based on scale factor
    if scale_factor < 0.9:
        # Shorter or modified sequence for smaller torsion
        return ['C', 'Y', 'G', 'D', 'C']
    elif scale_factor > 1.1:
        # Longer sequence for larger torsion
        return ['C', 'Y', 'G', 'D', 'W', 'P', 'C']
    else:
        return base_sequence

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_all_mutations() -> List[MutationInfo]:
    """
    Analyze all NRAS and HRAS mutations.
    
    Returns:
        List of MutationInfo with calculated values
    """
    print("=" * 70)
    print("NRAS/HRAS MUTATION ANALYSIS")
    print("Based on Conway Knot K11n34 Topology")
    print("Morató de Dalmases, 2026")
    print("=" * 70)
    
    results = []
    
    for mut in MUTATIONS:
        print(f"\n{mut.protein} {mut.mutation} ({mut.cancer_type})")
        print("-" * 40)
        
        # Calculate torsion
        mut.torsion = compute_torsion_from_mutation(mut.wt_aa, mut.mut_aa)
        print(f"  Pathological torsion: {mut.torsion:.3f} rad/Å")
        
        # Design Tap
        tap_seq = design_tap_for_mutation(mut.torsion)
        mut.tap_sequence = ''.join(tap_seq)
        print(f"  Proposed Tap sequence: {mut.tap_sequence}")
        
        # Calculate GEI (assuming perfect inverse torsion)
        tap_torsion = -mut.torsion
        mut.gei = compute_geometric_efficacy(mut.torsion, tap_torsion)
        print(f"  Geometric Efficacy Index: {mut.gei:.4f}")
        
        # Estimate binding energy
        q_mut = AMINO_ACIDS.get(mut.mut_aa, np.array([0.5, 0.5, 0.5, 0.5]))
        q_tap = AMINO_ACIDS.get(tap_seq[2] if len(tap_seq) > 2 else 'G', 
                                 np.array([0.125, 0.125, 0.125, 0.125]))
        mut.binding_energy = compute_binding_energy(q_mut, q_tap)
        print(f"  Estimated binding energy: {mut.binding_energy:.1f} kcal/mol")
        
        # Z0 point analysis
        quaternions = [AMINO_ACIDS.get(aa, np.array([0.5, 0.5, 0.5, 0.5])) 
                       for aa in [mut.wt_aa, mut.mut_aa]]
        norm = qnorm(product_sequence(quaternions)) if len(quaternions) > 1 else 1.0
        z0_deviation = abs(norm - 0.5)
        print(f"  Z0 point deviation: {z0_deviation:.4f}")
        
        if z0_deviation < 0.1:
            print("  → Region is at Z0 point: MAXIMUM TOPOLOGICAL INSTABILITY")
        
        results.append(mut)
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Mutation':<12} {'Cancer':<20} {'Torsion':<10} {'GEI':<8} {'Tap':<10}")
    print("-" * 70)
    for mut in results:
        print(f"{mut.protein} {mut.mutation:<6} {mut.cancer_type:<20} {mut.torsion:.3f}     {mut.gei:.4f}  {mut.tap_sequence}")
    
    return results

if __name__ == "__main__":
    analyze_all_mutations()