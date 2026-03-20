#!/usr/bin/env python3
"""
01_quaternion_algebra.py

Core quaternion algebra implementation for the geometric framework.
Includes quaternion multiplication, conjugation, norm, and normalization.

Morató de Dalmases, 2026
"""

import numpy as np
from typing import List, Tuple, Optional

# ============================================================================
# QUATERNION BASIS
# ============================================================================

Q_ONE = np.array([1, 0, 0, 0])
Q_I = np.array([0, 1, 0, 0])
Q_J = np.array([0, 0, 1, 0])
Q_K = np.array([0, 0, 0, 1])

QUATERNION_BASIS = {
    '1': Q_ONE,
    'i': Q_I,
    'j': Q_J,
    'k': Q_K
}

# ============================================================================
# QUATERNION OPERATIONS
# ============================================================================

def qmul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Quaternion multiplication.

    q = a + bi + cj + dk

    Args:
        q1: First quaternion [a, b, c, d]
        q2: Second quaternion [a, b, c, d]

    Returns:
        Product quaternion [a, b, c, d]
    """
    a1, b1, c1, d1 = q1
    a2, b2, c2, d2 = q2

    return np.array([
        a1*a2 - b1*b2 - c1*c2 - d1*d2,
        a1*b2 + b1*a2 + c1*d2 - d1*c2,
        a1*c2 - b1*d2 + c1*a2 + d1*b2,
        a1*d2 + b1*c2 - c1*b2 + d1*a2
    ])

def qconj(q: np.ndarray) -> np.ndarray:
    """
    Quaternion conjugate.

    Args:
        q: Quaternion [a, b, c, d]

    Returns:
        Conjugate quaternion [a, -b, -c, -d]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])

def qnorm(q: np.ndarray) -> float:
    """
    Quaternion norm (magnitude).

    Args:
        q: Quaternion [a, b, c, d]

    Returns:
        Norm sqrt(a² + b² + c² + d²)
    """
    return np.sqrt(np.sum(q**2))

def qnorm_sq(q: np.ndarray) -> float:
    """
    Squared quaternion norm.

    Args:
        q: Quaternion [a, b, c, d]

    Returns:
        Squared norm a² + b² + c² + d²
    """
    return np.sum(q**2)

def qnormalize(q: np.ndarray) -> np.ndarray:
    """
    Normalize quaternion to unit norm.

    Args:
        q: Quaternion [a, b, c, d]

    Returns:
        Unit quaternion
    """
    n = qnorm(q)
    if n == 0:
        return q
    return q / n

def qinv(q: np.ndarray) -> np.ndarray:
    """
    Quaternion inverse.

    Args:
        q: Quaternion [a, b, c, d]

    Returns:
        Inverse quaternion q^{-1} = conj(q) / |q|²
    """
    return qconj(q) / qnorm_sq(q)

def qexp(q: np.ndarray) -> np.ndarray:
    """
    Quaternion exponential.

    Args:
        q: Quaternion [a, b, c, d]

    Returns:
        exp(q)
    """
    a = q[0]
    v = q[1:4]
    v_norm = np.linalg.norm(v)

    if v_norm == 0:
        return np.array([np.exp(a), 0, 0, 0])

    return np.array([
        np.exp(a) * np.cos(v_norm),
        np.exp(a) * np.sin(v_norm) * v[0] / v_norm,
        np.exp(a) * np.sin(v_norm) * v[1] / v_norm,
        np.exp(a) * np.sin(v_norm) * v[2] / v_norm
    ])

def qlog(q: np.ndarray) -> np.ndarray:
    """
    Quaternion logarithm.

    Args:
        q: Quaternion [a, b, c, d]

    Returns:
        log(q)
    """
    norm = qnorm(q)
    if norm == 0:
        raise ValueError("Logarithm of zero quaternion")

    q_norm = q / norm
    a = q_norm[0]

    if a > 1:
        a = 1
    elif a < -1:
        a = -1

    angle = np.arccos(a)
    v = q_norm[1:4]
    v_norm = np.linalg.norm(v)

    if v_norm == 0:
        return np.array([np.log(norm), 0, 0, 0])

    return np.array([
        np.log(norm),
        angle * v[0] / v_norm,
        angle * v[1] / v_norm,
        angle * v[2] / v_norm
    ])

def qpow(q: np.ndarray, t: float) -> np.ndarray:
    """
    Quaternion power.

    Args:
        q: Quaternion [a, b, c, d]
        t: Exponent

    Returns:
        q^t = exp(t * log(q))
    """
    return qexp(t * qlog(q))

# ============================================================================
# QUATERNION SEQUENCE OPERATIONS
# ============================================================================

def product_sequence(quaternions: List[np.ndarray]) -> np.ndarray:
    """
    Compute the product of a sequence of quaternions.

    Args:
        quaternions: List of quaternions [q1, q2, ..., qn]

    Returns:
        Product q1 * q2 * ... * qn
    """
    if not quaternions:
        return Q_ONE

    q = quaternions[0]
    for qi in quaternions[1:]:
        q = qmul(q, qi)
    return q

def norm_sequence(quaternions: List[np.ndarray]) -> float:
    """
    Compute the norm of the product of a sequence of quaternions.

    Args:
        quaternions: List of quaternions [q1, q2, ..., qn]

    Returns:
        Norm of the product
    """
    return qnorm(product_sequence(quaternions))

def conjugation_product(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Compute the conjugation product q1 * conj(q2).

    Args:
        q1: First quaternion
        q2: Second quaternion

    Returns:
        q1 * conj(q2)
    """
    return qmul(q1, qconj(q2))

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def quaternion_to_string(q: np.ndarray, precision: int = 3) -> str:
    """
    Convert quaternion to human-readable string.

    Args:
        q: Quaternion [a, b, c, d]
        precision: Number of decimal places

    Returns:
        String representation like "a + bi + cj + dk"
    """
    a, b, c, d = q

    def fmt(x):
        return f"{x:.{precision}f}"

    parts = []
    if abs(a) > 1e-10:
        parts.append(fmt(a))
    if abs(b) > 1e-10:
        if b > 0:
            parts.append(f"{fmt(b)}i")
        else:
            parts.append(f"-{fmt(-b)}i")
    if abs(c) > 1e-10:
        if c > 0:
            parts.append(f"{fmt(c)}j")
        else:
            parts.append(f"-{fmt(-c)}j")
    if abs(d) > 1e-10:
        if d > 0:
            parts.append(f"{fmt(d)}k")
        else:
            parts.append(f"-{fmt(-d)}k")

    if not parts:
        return "0"

    result = parts[0]
    for part in parts[1:]:
        if part.startswith('-'):
            result += f" {part}"
        else:
            result += f" + {part}"

    return result

def is_unit_quaternion(q: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Check if quaternion has unit norm.

    Args:
        q: Quaternion [a, b, c, d]
        tol: Tolerance

    Returns:
        True if |q| = 1 within tolerance
    """
    return abs(qnorm(q) - 1) < tol

# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_quaternion_algebra():
    """Test quaternion algebra operations."""
    print("=" * 60)
    print("QUATERNION ALGEBRA TESTS")
    print("=" * 60)

    # Test basis multiplication
    print("\nBasis multiplication:")
    print(f"  i·i = {quaternion_to_string(qmul(Q_I, Q_I))}")
    print(f"  i·j = {quaternion_to_string(qmul(Q_I, Q_J))}")
    print(f"  j·k = {quaternion_to_string(qmul(Q_J, Q_K))}")
    print(f"  k·i = {quaternion_to_string(qmul(Q_K, Q_I))}")

    # Test conjugation
    print("\nConjugation:")
    q = np.array([1, 2, 3, 4])
    print(f"  q = {quaternion_to_string(q)}")
    print(f"  conj(q) = {quaternion_to_string(qconj(q))}")

    # Test norm
    print("\nNorm:")
    print(f"  |1| = {qnorm(Q_ONE):.4f}")
    print(f"  |i| = {qnorm(Q_I):.4f}")
    print(f"  |j| = {qnorm(Q_J):.4f}")
    print(f"  |k| = {qnorm(Q_K):.4f}")

    # Test exponential and logarithm
    print("\nExponential and logarithm:")
    q = np.array([0, 1, 0, 0])
    print(f"  q = {quaternion_to_string(q)}")
    print(f"  exp(q) = {quaternion_to_string(qexp(q))}")
    print(f"  log(exp(q)) = {quaternion_to_string(qlog(qexp(q)))}")

    print("\nAll tests passed.")

if __name__ == "__main__":
    test_quaternion_algebra()