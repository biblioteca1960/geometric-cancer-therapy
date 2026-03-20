#!/usr/bin/env python3
"""
06_stress_test_amyloid.py

Stress test analysis of amyloid-β misfolding in Alzheimer's disease.
Based on Conway knot topology and Z0 point instability.

Morató de Dalmases, 2026
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from constants import DELTA_0, ETA_GAMMA, K_B
from quaternion_algebra import qmul, qnorm, product_sequence

# ============================================================================
# AMINO ACID QUATERNIONS
# ============================================================================

AMINO_ACIDS = {
    'A': np.array([0.125, 0.250, 0.125, 0.250]),  # Alanine
    'C': np.array([0.250, 0.250, 0.125, 0.375]),  # Cysteine
    'D': np.array([0.375, 0.125, 0.250, 0.125]),  # Aspartic acid
    'E': np.array([0.500, 0.125, 0.375, 0.125]),  # Glutamic acid
    'F': np.array([0.750, 0.375, 0.125, 0.375]),  # Phenylalanine
    'G': np.array([0.125, 0.125, 0.125, 0.125]),  # Glycine
    'H': np.array([0.375, 0.250, 0.500, 0.250]),  # Histidine
    'I': np.array([0.625, 0.375, 0.125, 0.375]),  # Isoleucine
    'K': np.array([0.625, 0.500, 0.375, 0.500]),  # Lysine
    'L': np.array([0.625, 0.250, 0.250, 0.250]),  # Leucine
    'M': np.array([0.500, 0.250, 0.125, 0.250]),  # Methionine
    'N': np.array([0.250, 0.125, 0.375, 0.125]),  # Asparagine
    'P': np.array([0.375, 0.125, 0.250, 0.125]),  # Proline
    'Q': np.array([0.500, 0.375, 0.250, 0.375]),  # Glutamine
    'R': np.array([0.375, 0.375, 0.375, 0.375]),  # Arginine
    'S': np.array([0.250, 0.250, 0.250, 0.250]),  # Serine
    'T': np.array([0.375, 0.375, 0.125, 0.375]),  # Threonine
    'V': np.array([0.500, 0.500, 0.125, 0.500]),  # Valine
    'W': np.array([0.875, 0.500, 0.250, 0.500]),  # Tryptophan
    'Y': np.array([0.750, 0.250, 0.375, 0.250]),  # Tyrosine
}

# ============================================================================
# AMYLOID-β SEQUENCES
# ============================================================================

# Amyloid-β critical region (residues 16-22) - KLVFFAE
ABETA_CRITICAL = "KLVFFAE"

# Full amyloid-β (1-42)
ABETA_FULL = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"

# ============================================================================
# FOLDING ANALYSIS
# ============================================================================

@dataclass
class FoldingAnalysis:
    """Results of protein folding analysis."""
    sequence: str
    norm: float
    curvature: float
    stability: str
    transition_probability: float
    is_z0: bool

def analyze_folding(sequence: str, temperature: float = 310, crowding: float = 0.3) -> FoldingAnalysis:
    """
    Analyze protein folding stability using quaternion geometry.
    
    Args:
        sequence: Amino acid sequence (one-letter code)
        temperature: Temperature in Kelvin
        crowding: Molecular crowding fraction (0-1)
    
    Returns:
        FoldingAnalysis with stability metrics
    """
    # Convert sequence to quaternions
    quaternions = []
    for aa in sequence.upper():
        if aa in AMINO_ACIDS:
            quaternions.append(AMINO_ACIDS[aa])
        else:
            raise ValueError(f"Unknown amino acid: {aa}")
    
    # Compute total product and norm
    total = product_sequence(quaternions)
    norm = qnorm(total)
    
    # Compute local curvature
    curvatures = []
    for i in range(len(quaternions) - 1):
        local_product = qmul(quaternions[i], quaternions[i+1])
        local_norm = qnorm(local_product)
        curvature = abs(1 - local_norm) * DELTA_0
        curvatures.append(curvature)
    avg_curvature = np.mean(curvatures) if curvatures else 0
    
    # Determine stability
    if abs(norm - 1) < 0.2:
        stability = "STABLE (α-helix)"
    elif abs(norm - np.sqrt(3)) < 0.2:
        stability = "STABLE (β-sheet)"
    elif abs(norm - 0.5) < 0.2:
        stability = "METASTABLE (Z0 point - Conway knot instability)"
    else:
        stability = f"TRANSITIONAL (norm = {norm:.3f})"
    
    # Calculate transition probability
    delta_E = abs(norm - 1) * 0.5
    beta = 1 / (K_B * temperature)
    crowding_factor = 1 / (1 - crowding)
    delta_E_eff = delta_E / crowding_factor
    transition_probability = np.exp(-beta * delta_E_eff)
    
    return FoldingAnalysis(
        sequence=sequence,
        norm=norm,
        curvature=avg_curvature,
        stability=stability,
        transition_probability=transition_probability,
        is_z0=abs(norm - 0.5) < 0.1
    )

def compare_mutations(wildtype: str, mutant: str) -> Dict:
    """Compare folding stability between wildtype and mutant."""
    wt_analysis = analyze_folding(wildtype)
    mut_analysis = analyze_folding(mutant)
    
    return {
        'wildtype_norm': wt_analysis.norm,
        'mutant_norm': mut_analysis.norm,
        'norm_difference': mut_analysis.norm - wt_analysis.norm,
        'wildtype_stability': wt_analysis.stability,
        'mutant_stability': mut_analysis.stability,
        'is_pathogenic': abs(mut_analysis.norm - np.sqrt(3)) < abs(wt_analysis.norm - 1)
    }

# ============================================================================
# MAIN STRESS TEST
# ============================================================================

def run_stress_test():
    """Run complete stress test on amyloid-β."""
    print("=" * 70)
    print("STRESS TEST: Amyloid-β Folding Analysis")
    print("Based on Conway Knot K11n34 Topology")
    print("Morató de Dalmases, 2026")
    print("=" * 70)
    
    # 1. Critical region analysis
    print(f"\n1. Amyloid-β Critical Region: {ABETA_CRITICAL}")
    analysis = analyze_folding(ABETA_CRITICAL, crowding=0.3)
    print(f"   Total quaternion norm: {analysis.norm:.4f}")
    print(f"   Z0 point deviation: {abs(analysis.norm - 0.5):.4f}")
    print(f"   Stability: {analysis.stability}")
    print(f"   Transition probability: {analysis.transition_probability:.2e}")
    
    # 2. Full sequence analysis
    print(f"\n2. Amyloid-β Full Sequence (1-42):")
    full_analysis = analyze_folding(ABETA_FULL, crowding=0.3)
    print(f"   Total quaternion norm: {full_analysis.norm:.4f}")
    print(f"   Stability: {full_analysis.stability}")
    
    # 3. Effect of crowding
    print(f"\n3. Effect of Molecular Crowding on Aβ Transition:")
    crowding_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for crowd in crowding_levels:
        analysis_crowd = analyze_folding(ABETA_CRITICAL, crowding=crowd)
        print(f"   Crowding {crowd*100:.0f}%: norm = {analysis_crowd.norm:.4f}, "
              f"P(transition) = {analysis_crowd.transition_probability:.2e}")
    
    # 4. Mutation analysis
    print(f"\n4. Mutation Analysis (Alzheimer's Risk Factors):")
    wt = "KLVFFAE"
    mutations = [
        ("A21G", "KLVFGFE"),
        ("A21V", "KLVFFVE"),
        ("F20L", "KLVFLAE"),
    ]
    
    for name, mutant in mutations:
        comparison = compare_mutations(wt, mutant)
        print(f"\n   {name}: {mutant}")
        print(f"   Wildtype norm: {comparison['wildtype_norm']:.4f}")
        print(f"   Mutant norm: {comparison['mutant_norm']:.4f}")
        print(f"   Pathogenic prediction: {comparison['is_pathogenic']}")
    
    # 5. Energy barrier analysis
    print(f"\n5. α-helix ↔ β-sheet Transition Analysis:")
    alpha_helix = "A" * 20
    beta_sheet = "VKVKVKVKVK" * 2
    
    alpha = analyze_folding(alpha_helix)
    beta = analyze_folding(beta_sheet)
    
    print(f"   α-helix (polyA): norm = {alpha.norm:.4f}")
    print(f"   β-sheet (alternating): norm = {beta.norm:.4f}")
    
    delta_norm = abs(beta.norm - alpha.norm)
    barrier = delta_norm * 0.5
    print(f"   Energy barrier: {barrier:.3f} kcal/mol")
    print(f"   Thermal energy at 310K: 0.600 kcal/mol")
    
    if barrier < 0.6:
        print("   → BARRIER SURMOUNTABLE: Transition possible at body temperature")
    else:
        print("   → BARRIER INSURMOUNTABLE: Transition requires external stress")
    
    # 6. Final verdict
    print("\n" + "=" * 70)
    print("STRESS TEST VERDICT")
    print("=" * 70)
    
    if 0.8 < analysis.norm < 1.2:
        print("   Amyloid-β: STABLE in α-helix conformation at low crowding")
    else:
        print("   Amyloid-β: INTRINSICALLY UNSTABLE, prone to β-sheet transition")
        print(f"   Z0 point proximity: {abs(analysis.norm - 0.5):.3f} (Conway knot instability)")
    
    if 0.8 < full_analysis.norm < 1.2:
        print("   Full Aβ (1-42): PREDICTED STABLE under physiological conditions")
    else:
        print("   Full Aβ (1-42): PREDICTED METASTABLE - aggregation likely")
        print(f"   Z0 point proximity: {abs(full_analysis.norm - 0.5):.3f}")
    
    print("\n" + "=" * 70)
    print("STRESS TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    run_stress_test()