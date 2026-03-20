#!/usr/bin/env python3
"""
08_genetic_code_quaternion.py

Complete quaternion representation of the genetic code.
64 codons as triple products of nucleotide quaternions.

Morató de Dalmases, 2026
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from quaternion_algebra import qmul, qnorm, product_sequence, Q_ONE, Q_I, Q_J, Q_K

# ============================================================================
# NUCLEOTIDE QUATERNIONS
# ============================================================================

# Nucleotide to quaternion mapping (from Conway knot projection)
NUCLEOTIDES = {
    'A': Q_ONE,      # 1
    'C': Q_I,        # i
    'G': Q_J,        # j
    'U': Q_K,        # k
    'T': Q_K,        # Thymine (replaces U in DNA)
}

# Nucleotide indices for torsion phases
NUCLEOTIDE_INDEX = {
    'A': 1,
    'C': 2,
    'G': 3,
    'U': 4,
    'T': 4,
}

# ============================================================================
# AMINO ACID MAPPING
# ============================================================================

# Standard genetic code (64 codons -> amino acids)
GENETIC_CODE = {
    # Phenylalanine
    'UUU': 'Phe', 'UUC': 'Phe',
    # Leucine
    'UUA': 'Leu', 'UUG': 'Leu', 'CUU': 'Leu', 'CUC': 'Leu', 'CUA': 'Leu', 'CUG': 'Leu',
    # Isoleucine
    'AUU': 'Ile', 'AUC': 'Ile', 'AUA': 'Ile',
    # Methionine (Start)
    'AUG': 'Met',
    # Valine
    'GUU': 'Val', 'GUC': 'Val', 'GUA': 'Val', 'GUG': 'Val',
    # Serine
    'UCU': 'Ser', 'UCC': 'Ser', 'UCA': 'Ser', 'UCG': 'Ser', 'AGU': 'Ser', 'AGC': 'Ser',
    # Proline
    'CCU': 'Pro', 'CCC': 'Pro', 'CCA': 'Pro', 'CCG': 'Pro',
    # Threonine
    'ACU': 'Thr', 'ACC': 'Thr', 'ACA': 'Thr', 'ACG': 'Thr',
    # Alanine
    'GCU': 'Ala', 'GCC': 'Ala', 'GCA': 'Ala', 'GCG': 'Ala',
    # Tyrosine
    'UAU': 'Tyr', 'UAC': 'Tyr',
    # Histidine
    'CAU': 'His', 'CAC': 'His',
    # Glutamine
    'CAA': 'Gln', 'CAG': 'Gln',
    # Asparagine
    'AAU': 'Asn', 'AAC': 'Asn',
    # Lysine
    'AAA': 'Lys', 'AAG': 'Lys',
    # Aspartic acid
    'GAU': 'Asp', 'GAC': 'Asp',
    # Glutamic acid
    'GAA': 'Glu', 'GAG': 'Glu',
    # Cysteine
    'UGU': 'Cys', 'UGC': 'Cys',
    # Tryptophan
    'UGG': 'Trp',
    # Arginine
    'CGU': 'Arg', 'CGC': 'Arg', 'CGA': 'Arg', 'CGG': 'Arg', 'AGA': 'Arg', 'AGG': 'Arg',
    # Serine (continued)
    'AGU': 'Ser', 'AGC': 'Ser',
    # Glycine
    'GGU': 'Gly', 'GGC': 'Gly', 'GGA': 'Gly', 'GGG': 'Gly',
    # Stop codons
    'UAA': 'Stop', 'UAG': 'Stop', 'UGA': 'Stop',
}

# Amino acid quaternions (from matter 2.pdf)
AMINO_ACID_QUATERNIONS = {
    'Ala': np.array([0.125, 0.250, 0.125, 0.250]),
    'Arg': np.array([0.375, 0.375, 0.375, 0.375]),
    'Asn': np.array([0.250, 0.125, 0.375, 0.125]),
    'Asp': np.array([0.375, 0.125, 0.250, 0.125]),
    'Cys': np.array([0.250, 0.250, 0.125, 0.375]),
    'Gln': np.array([0.500, 0.375, 0.250, 0.375]),
    'Glu': np.array([0.500, 0.125, 0.375, 0.125]),
    'Gly': np.array([0.125, 0.125, 0.125, 0.125]),
    'His': np.array([0.375, 0.250, 0.500, 0.250]),
    'Ile': np.array([0.625, 0.375, 0.125, 0.375]),
    'Leu': np.array([0.625, 0.250, 0.250, 0.250]),
    'Lys': np.array([0.625, 0.500, 0.375, 0.500]),
    'Met': np.array([0.500, 0.250, 0.125, 0.250]),
    'Phe': np.array([0.750, 0.375, 0.125, 0.375]),
    'Pro': np.array([0.375, 0.125, 0.250, 0.125]),
    'Ser': np.array([0.250, 0.250, 0.250, 0.250]),
    'Thr': np.array([0.375, 0.375, 0.125, 0.375]),
    'Trp': np.array([0.875, 0.500, 0.250, 0.500]),
    'Tyr': np.array([0.750, 0.250, 0.375, 0.250]),
    'Val': np.array([0.500, 0.500, 0.125, 0.500]),
    'Stop': np.array([0, 0, 0, 0]),
}

# ============================================================================
# CODON QUATERNION CALCULATION
# ============================================================================

@dataclass
class CodonInfo:
    """Information about a codon."""
    sequence: str
    quaternion: np.ndarray
    norm: float
    amino_acid: str
    amino_acid_quaternion: np.ndarray
    family: str

def codon_quaternion(codon: str) -> np.ndarray:
    """
    Compute quaternion product for a codon.
    
    Args:
        codon: 3-letter codon string (e.g., "AUG")
    
    Returns:
        Quaternion product q1 * q2 * q3
    """
    q = NUCLEOTIDES[codon[0]]
    for base in codon[1:]:
        q = qmul(q, NUCLEOTIDES[base])
    return q

def analyze_codon(codon: str) -> CodonInfo:
    """
    Analyze a codon and return its quaternion properties.
    
    Args:
        codon: 3-letter codon string
    
    Returns:
        CodonInfo with quaternion and amino acid mapping
    """
    q = codon_quaternion(codon)
    norm = qnorm(q)
    aa = GENETIC_CODE.get(codon.upper(), 'Unknown')
    aa_q = AMINO_ACID_QUATERNIONS.get(aa, np.zeros(4))
    
    # Determine family by norm
    if abs(norm - 1) < 0.2:
        family = "Nonpolar (Ala, Val, Leu, Ile, Pro, Met, Phe, Trp)"
    elif abs(norm - np.sqrt(2)) < 0.2:
        family = "Polar uncharged (Ser, Thr, Cys, Asn, Gln, Tyr)"
    elif abs(norm - np.sqrt(3)) < 0.2:
        family = "Acidic/Basic (Asp, Glu, Lys, Arg, His)"
    elif abs(norm - 2) < 0.2:
        family = "Stop codons"
    else:
        family = f"Transitional (norm = {norm:.3f})"
    
    return CodonInfo(
        sequence=codon,
        quaternion=q,
        norm=norm,
        amino_acid=aa,
        amino_acid_quaternion=aa_q,
        family=family
    )

# ============================================================================
# COMPLETE CODON TABLE
# ============================================================================

def generate_codon_table() -> pd.DataFrame:
    """Generate complete codon table with quaternion products."""
    bases = ['A', 'C', 'G', 'U']
    codons = [b1 + b2 + b3 for b1 in bases for b2 in bases for b3 in bases]
    
    data = []
    for codon in codons:
        info = analyze_codon(codon)
        data.append({
            'Codon': codon,
            'Quaternion': info.quaternion,
            'Norm': info.norm,
            'Amino Acid': info.amino_acid,
            'Family': info.family
        })
    
    return pd.DataFrame(data)

def print_codon_table():
    """Print formatted codon table."""
    df = generate_codon_table()
    
    print("=" * 80)
    print("COMPLETE CODON TABLE (64 codons as quaternion products)")
    print("=" * 80)
    print(f"{'Codon':<6} {'Quaternion Product':<25} {'Norm':<8} {'Amino Acid':<12} {'Family'}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        q_str = quaternion_to_string(row['Quaternion'], precision=1)
        print(f"{row['Codon']:<6} {q_str:<25} {row['Norm']:<8.3f} {row['Amino Acid']:<12} {row['Family'][:40]}")
    
    print("-" * 80)
    print(f"Total codons: {len(df)}")

# ============================================================================
# QUATERNION TO STRING UTILITY
# ============================================================================

def quaternion_to_string(q: np.ndarray, precision: int = 2) -> str:
    """Convert quaternion to human-readable string."""
    a, b, c, d = q
    
    def fmt(x):
        if abs(x) < 1e-10:
            return "0"
        return f"{x:.{precision}f}"
    
    parts = []
    if abs(a) > 1e-10:
        parts.append(fmt(a))
    if abs(b) > 1e-10:
        parts.append(f"{fmt(b)}i" if b > 0 else f"-{fmt(-b)}i")
    if abs(c) > 1e-10:
        parts.append(f"{fmt(c)}j" if c > 0 else f"-{fmt(-c)}j")
    if abs(d) > 1e-10:
        parts.append(f"{fmt(d)}k" if d > 0 else f"-{fmt(-d)}k")
    
    if not parts:
        return "0"
    
    result = parts[0]
    for part in parts[1:]:
        if part.startswith('-'):
            result += f" {part}"
        else:
            result += f" + {part}"
    
    return result

# ============================================================================
# CODON USAGE ANALYSIS
# ============================================================================

def codon_usage_frequency(sequence: str) -> Dict[str, float]:
    """Calculate codon usage frequency in a DNA/RNA sequence."""
    seq = sequence.upper()
    codons = [seq[i:i+3] for i in range(0, len(seq)-2, 3)]
    
    freq = {}
    for codon in codons:
        freq[codon] = freq.get(codon, 0) + 1
    
    total = sum(freq.values())
    return {c: f/total for c, f in freq.items()}

def analyze_codon_families(sequence: str) -> Dict[str, float]:
    """Analyze distribution of codon families in a sequence."""
    freq = codon_usage_frequency(sequence)
    
    families = {
        'Nonpolar': 0,
        'Polar uncharged': 0,
        'Acidic/Basic': 0,
        'Stop': 0,
        'Transitional': 0
    }
    
    for codon, f in freq.items():
        info = analyze_codon(codon)
        if 'Nonpolar' in info.family:
            families['Nonpolar'] += f
        elif 'Polar uncharged' in info.family:
            families['Polar uncharged'] += f
        elif 'Acidic/Basic' in info.family:
            families['Acidic/Basic'] += f
        elif 'Stop' in info.family:
            families['Stop'] += f
        else:
            families['Transitional'] += f
    
    return families

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print_codon_table()
    
    # Example: Analyze a specific codon
    print("\n" + "=" * 80)
    print("EXAMPLE: AUG (Methionine start codon)")
    print("=" * 80)
    info = analyze_codon("AUG")
    print(f"Codon: {info.sequence}")
    print(f"Quaternion: {quaternion_to_string(info.quaternion)}")
    print(f"Norm: {info.norm:.4f}")
    print(f"Amino Acid: {info.amino_acid}")
    print(f"Family: {info.family}")
    
    # Example: Human genome codon usage (simulated)
    print("\n" + "=" * 80)
    print("HUMAN CODON USAGE (simulated)")
    print("=" * 80)
    human_sequence = "AUG" * 1000 + "CUG" * 850 + "GCC" * 720 + "AGA" * 680 + "GAG" * 620
    families = analyze_codon_families(human_sequence)
    for family, freq in families.items():
        print(f"  {family}: {freq:.1%}")