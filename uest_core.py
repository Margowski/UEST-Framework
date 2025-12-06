# ======================================================================
#  UEST Framework – Discrete Proxy Implementation
#  Copyright (c) 2025
#  Author: Olaf Margowski
#
#  This software is part of the Unified Energy–Structure–Tension (UEST)
#  Framework and is protected by copyright and intellectual property law.
#
#  LICENSE:
#  This code is released under the Creative Commons
#  Attribution–NonCommercial–ShareAlike 4.0 International License (CC BY-NC-SA 4.0).
#
#  You are free to:
#      • Share — copy and redistribute the material in any medium or format
#      • Adapt — remix, transform, and build upon the material
#
#  Under the following terms:
#      • Attribution — You must give appropriate credit, provide a link to the license,
#        and indicate if changes were made. Attribution must name:
#            "Olaf Margowski, UEST Framework (2025)"
#
#      • NonCommercial — You may NOT use the material for commercial purposes,
#        including but not limited to:
#          – corporate use,
#          – industrial integration,
#          – product development,
#          – commercial AI training,
#          – or any activity intended for monetary or strategic gain.
#
#      • ShareAlike — If you modify or build upon this code, you must distribute
#        your contributions under the *same license*, with identical restrictions.
#
#  Additional Restrictions (UEST-Specific Clause):
#      • No part of this code may be incorporated into machine learning systems,
#        AI products, or automated pipelines for the purpose of commercial decision
#        making, risk scoring, or any form of industrial analytics.
#      • The UEST name, terminology, and conceptual framework may not be used to
#        imply endorsement, certification, or scientific validation of any external
#        system or product.
#
#  DISCLAIMER:
#      This file contains a DISCRETE PROXY IMPLEMENTATION of selected concepts
#      from the UEST analytical framework. It does NOT implement the continuous
#      mathematical definitions of the UEST theory.
#      This code is provided for research, education, academic analysis,
#      and non-commercial experimentation only.
#
#  Full license text: https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# ======================================================================
# NOTE (UEST Framework – Chaos Threshold, DISCRETE PROXY):
# The value CHAOS_THRESHOLD = 0.61803 is a HEURISTIC cutoff used
# exclusively for discrete LLM-structure experiments in this module.
#
# It is NOT the rigorously derived critical exponent from the
# Tension-Divergence Theorem in the continuous UEST theory.
# The formal derivation of the critical instability threshold
# is given only in the analytical work (UEST3_Beweis, Sec. "Tension-Divergence"),
# and is not implemented here.
import numpy as np

CHAOS_THRESHOLD = 0.61803  # heuristic proxy threshold for demo purposes only


# ---------------------------------------------------------------------------
# 1. Tensions-Divergence Theorem (DISCRETE PROXY VERSION)
# ---------------------------------------------------------------------------
class TensionsDivergenceTheorem:
    """
    Discrete PROXY version of the UEST Tensions-Divergence Theorem.

    This class provides a heuristic critical value for structural instability
    in FINITE-DIMENSIONAL, DISCRETE demo systems (e.g. LLM decision matrices).

    Important:
    - The constant CRITICAL_VALUE below is a stable, heuristic cutoff that
      imitates the role of the rigorously derived instability threshold from
      the continuous UEST theory.
    - It is NOT the mathematically derived exponent from the analytical
      UEST3 framework; that value only appears in the proof document and
      is not implemented here.
    """

    # Heuristic proxy value (for demo / XAI use only).
    # In the continuous theory this value would be derived axiomatically;
    # here we use a fixed, interpretable placeholder.
    CRITICAL_VALUE = 0.61803  # e.g. Golden Ratio as chaos-analogue PROXY

    @staticmethod
    def get_critical_value() -> float:
        """Return the heuristic critical threshold for discrete Sigma-proxies."""
        return TensionsDivergenceTheorem.CRITICAL_VALUE


# ---------------------------------------------------------------------------
# 2. Emergent Structural Time (T) – DISCRETE PROXY METRIC
# ---------------------------------------------------------------------------
# PROXY METRICS FOR DISCRETE SYSTEMS
#
# IMPORTANT:
# The functions below implement DISCRETE PROXY METRICS that emulate the
# qualitative behaviour of the continuous UEST quantities
#   Σ(E')  (sensitivity profile)
#   T      (emergent structural time, T = ∫ Σ dE')
#
# They operate on finite-dimensional matrices (e.g. LLM token graphs) using
# determinants, spectral radii and simple norms. They are designed for
# structural pattern verification and XAI experiments and MUST NOT be
# interpreted as exact implementations of the continuous UEST integrals
# from the analytical framework.
#
# For the rigorous mathematical definitions, see the UEST paper
# (Sections "Sensitivity Function" and "Emergent Structural Time").
def calculate_structural_time(matrix: list) -> float:
    """
    DISCRETE PROXY for the Emergent Structural Time (T).

    Input:
        matrix : list of lists (finite-dimensional weight / decision matrix)

    Idea:
        T_proxy is computed from the determinant magnitude and the size
        of the matrix. This mimics "structural maturity / complexity"
        in a simple, interpretable way for DEMO purposes.

    Note:
        This is NOT the continuous integral definition
            T(E') = ∫ Σ(E') dE'
        from the UEST framework, but a surrogate for discrete systems.
    """
    A = np.array(matrix, dtype=float)

    try:
        # Determinant as a simple proxy for structural volume/complexity.
        determinant = np.linalg.det(A)

        # Heuristic T-metric: log-scaled determinant times sqrt(size).
        T_proxy = np.log(1.0 + np.abs(determinant)) * np.sqrt(A.size)
        return float(T_proxy)
    except np.linalg.LinAlgError:
        # Singular matrix -> treat as structurally degenerate in this proxy view.
        return 0.0


# ---------------------------------------------------------------------------
# 3. Sensitivity (Sigma) – DISCRETE PROXY METRIC
# ---------------------------------------------------------------------------
def calculate_sensitivity(matrix: list) -> tuple[float, float]:
    """
    DISCRETE PROXY for the UEST Sensitivity Σ.

    Returns:
        Sigma_total : float
            Proxy for the global structural amplification / instability,
            here given by the spectral radius of the matrix.
        Sigma_max_node : float
            Proxy for the maximal local node "tension".

    Interpretation:
        - High Sigma_total or Sigma_max_node indicates brittle or
          instability-prone structures in the DEMO setting.
        - This is ONLY a proxy for Σ(E') and its integral; it does
          not implement the exact UEST sensitivity definition.
    """
    A = np.array(matrix, dtype=float)

    # Spectral radius as proxy for maximal amplification/instability.
    eigenvalues = np.linalg.eigvals(A)
    Sigma_total = float(np.max(np.abs(eigenvalues)))

    # Simple local proxy: row sums as "node tension".
    Sigma_nodes = np.sum(A, axis=1)
    Sigma_max_node = float(np.max(Sigma_nodes) * Sigma_total * 0.1)  # scaling for demo

    # Optional: ensure we see interesting values in demos.
    if Sigma_max_node < 0.3:
        Sigma_max_node *= 2.0

    return Sigma_total, Sigma_max_node
