"""
UEST Framework - Unified Eigenspace Structural Theory for Explainable AI

This module implements the core UEST framework for quantifying internal structural 
stability and maturity of Neural Network decision states, transforming black box 
outputs into structurally interpretable white box outputs.

The framework operates in a finite-dimensional, discrete setting and provides:
1. Structural stability metrics
2. Decision state maturity quantification
3. White box transformation of NN outputs
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings


@dataclass
class StructuralMetrics:
    """Container for structural stability and maturity metrics"""
    stability_score: float  # Overall stability measure [0, 1]
    maturity_index: float   # Decision maturity level [0, 1]
    eigenvalue_entropy: float  # Entropy of eigenvalue distribution
    coherence_measure: float   # Internal state coherence
    confidence_alignment: float  # Alignment between confidence and structure
    interpretability_score: float  # Overall interpretability measure
    
    def __str__(self) -> str:
        return (
            f"Structural Metrics:\n"
            f"  Stability Score: {self.stability_score:.4f}\n"
            f"  Maturity Index: {self.maturity_index:.4f}\n"
            f"  Eigenvalue Entropy: {self.eigenvalue_entropy:.4f}\n"
            f"  Coherence Measure: {self.coherence_measure:.4f}\n"
            f"  Confidence Alignment: {self.confidence_alignment:.4f}\n"
            f"  Interpretability Score: {self.interpretability_score:.4f}"
        )


@dataclass
class WhiteBoxOutput:
    """White box representation of neural network decision"""
    prediction: Any  # Original prediction
    confidence: float  # Prediction confidence
    structural_metrics: StructuralMetrics  # UEST structural analysis
    eigenspace_representation: np.ndarray  # Eigenspace decomposition
    stability_indicators: Dict[str, float]  # Detailed stability measures
    interpretation: str  # Human-readable interpretation
    
    def __str__(self) -> str:
        return (
            f"White Box Neural Network Output:\n"
            f"{'='*50}\n"
            f"Prediction: {self.prediction}\n"
            f"Confidence: {self.confidence:.4f}\n\n"
            f"{self.structural_metrics}\n\n"
            f"Interpretation:\n{self.interpretation}\n"
            f"{'='*50}"
        )


class UESTFramework:
    """
    Main UEST Framework class for analyzing neural network decision states.
    
    This framework quantifies the structural stability and maturity of NN decisions
    by analyzing their internal representations in eigenspace.
    """
    
    def __init__(self, dimensionality: int = 10, stability_threshold: float = 0.7):
        """
        Initialize the UEST Framework.
        
        Args:
            dimensionality: Expected dimensionality of decision state
            stability_threshold: Threshold for considering a state stable
        """
        self.dimensionality = dimensionality
        self.stability_threshold = stability_threshold
        
    def compute_eigenvalue_entropy(self, eigenvalues: np.ndarray) -> float:
        """
        Compute the entropy of eigenvalue distribution.
        
        Higher entropy indicates more distributed structure (less concentrated).
        Lower entropy indicates dominant eigenvalues (more concentrated).
        
        Args:
            eigenvalues: Array of eigenvalues
            
        Returns:
            Entropy value
        """
        # Normalize eigenvalues to form a probability distribution
        eigenvalues = np.abs(eigenvalues)
        if np.sum(eigenvalues) == 0:
            return 0.0
        
        probs = eigenvalues / np.sum(eigenvalues)
        probs = probs[probs > 1e-10]  # Remove near-zero values
        
        # Compute Shannon entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log(len(eigenvalues))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy
    
    def compute_stability_score(self, state_matrix: np.ndarray) -> float:
        """
        Compute structural stability score from state matrix.
        
        Stability is measured by:
        1. Condition number (lower is more stable)
        2. Spectral gap (larger gap indicates stability)
        3. Matrix norm behavior
        
        Args:
            state_matrix: Internal state representation matrix
            
        Returns:
            Stability score in [0, 1], where 1 is most stable
        """
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(state_matrix)
        eigenvalues = np.abs(eigenvalues)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
        
        # Measure 1: Condition number (inverse of stability)
        max_eig = eigenvalues[0] if len(eigenvalues) > 0 else 1.0
        min_eig = eigenvalues[-1] if len(eigenvalues) > 0 else 1.0
        condition_number = max_eig / (min_eig + 1e-10)
        condition_stability = 1.0 / (1.0 + np.log(condition_number + 1.0))
        
        # Measure 2: Spectral gap (larger gap = more stability)
        if len(eigenvalues) > 1:
            spectral_gap = eigenvalues[0] - eigenvalues[1]
            gap_stability = np.tanh(spectral_gap)  # Normalize to [0, 1]
        else:
            gap_stability = 0.5
        
        # Measure 3: Eigenvalue concentration (higher concentration = more stability)
        if len(eigenvalues) > 0:
            eigenvalue_concentration = eigenvalues[0] / (np.sum(eigenvalues) + 1e-10)
        else:
            eigenvalue_concentration = 0.0
        
        # Combine measures
        stability = (0.4 * condition_stability + 
                    0.3 * gap_stability + 
                    0.3 * eigenvalue_concentration)
        
        return float(np.clip(stability, 0.0, 1.0))
    
    def compute_maturity_index(self, 
                              state_matrix: np.ndarray, 
                              confidence: float) -> float:
        """
        Compute decision maturity index.
        
        Maturity indicates how well-formed and decisive the decision state is.
        High maturity means the decision is clear and well-supported by structure.
        
        Args:
            state_matrix: Internal state representation
            confidence: Model's confidence in prediction
            
        Returns:
            Maturity index in [0, 1]
        """
        # Factor 1: Confidence level
        confidence_factor = confidence
        
        # Factor 2: State concentration (trace vs Frobenius norm)
        trace = np.abs(np.trace(state_matrix))
        frobenius = np.linalg.norm(state_matrix, 'fro')
        concentration = trace / (frobenius + 1e-10)
        concentration_factor = np.clip(concentration, 0.0, 1.0)
        
        # Factor 3: Eigenvalue dominance
        eigenvalues = np.abs(np.linalg.eigvals(state_matrix))
        if len(eigenvalues) > 0:
            eigenvalues_sorted = np.sort(eigenvalues)[::-1]
            if len(eigenvalues_sorted) > 1:
                dominance = eigenvalues_sorted[0] / (eigenvalues_sorted[1] + 1e-10)
                dominance_factor = np.tanh(dominance / 5.0)  # Normalize
            else:
                dominance_factor = 1.0
        else:
            dominance_factor = 0.0
        
        # Combine factors
        maturity = (0.3 * confidence_factor + 
                   0.4 * concentration_factor + 
                   0.3 * dominance_factor)
        
        return float(np.clip(maturity, 0.0, 1.0))
    
    def compute_coherence(self, state_matrix: np.ndarray) -> float:
        """
        Compute internal coherence of the decision state.
        
        Coherence measures how well the internal state aligns with itself.
        
        Args:
            state_matrix: Internal state representation
            
        Returns:
            Coherence measure in [0, 1]
        """
        # Symmetry measure
        symmetry = np.linalg.norm(state_matrix - state_matrix.T, 'fro')
        symmetry_score = np.exp(-symmetry)
        
        # Positive definiteness measure (via eigenvalues)
        eigenvalues = np.linalg.eigvals(state_matrix)
        real_eigenvalues = np.real(eigenvalues)
        positive_ratio = np.sum(real_eigenvalues > 0) / len(real_eigenvalues)
        
        # Combine measures
        coherence = 0.6 * symmetry_score + 0.4 * positive_ratio
        
        return float(np.clip(coherence, 0.0, 1.0))
    
    def compute_confidence_alignment(self, 
                                    state_matrix: np.ndarray,
                                    confidence: float) -> float:
        """
        Compute alignment between model confidence and structural stability.
        
        High alignment means the model's confidence is well-supported by structure.
        
        Args:
            state_matrix: Internal state representation
            confidence: Model confidence
            
        Returns:
            Alignment score in [0, 1]
        """
        stability = self.compute_stability_score(state_matrix)
        
        # Alignment is high when both are high or both are low
        # Use inverse of absolute difference
        alignment = 1.0 - abs(confidence - stability)
        
        return float(np.clip(alignment, 0.0, 1.0))
    
    def analyze_decision_state(self,
                              state_matrix: np.ndarray,
                              confidence: float,
                              prediction: Any) -> StructuralMetrics:
        """
        Perform comprehensive UEST analysis of a decision state.
        
        Args:
            state_matrix: Internal state representation matrix
            confidence: Model's prediction confidence
            prediction: The model's prediction
            
        Returns:
            StructuralMetrics object with all computed metrics
        """
        # Compute all metrics
        eigenvalues = np.linalg.eigvals(state_matrix)
        
        stability_score = self.compute_stability_score(state_matrix)
        maturity_index = self.compute_maturity_index(state_matrix, confidence)
        eigenvalue_entropy = self.compute_eigenvalue_entropy(eigenvalues)
        coherence_measure = self.compute_coherence(state_matrix)
        confidence_alignment = self.compute_confidence_alignment(state_matrix, confidence)
        
        # Compute overall interpretability score
        interpretability_score = (
            0.25 * stability_score +
            0.25 * maturity_index +
            0.20 * (1.0 - eigenvalue_entropy) +  # Lower entropy = more interpretable
            0.15 * coherence_measure +
            0.15 * confidence_alignment
        )
        
        return StructuralMetrics(
            stability_score=stability_score,
            maturity_index=maturity_index,
            eigenvalue_entropy=eigenvalue_entropy,
            coherence_measure=coherence_measure,
            confidence_alignment=confidence_alignment,
            interpretability_score=interpretability_score
        )
    
    def generate_interpretation(self, 
                               metrics: StructuralMetrics,
                               prediction: Any,
                               confidence: float) -> str:
        """
        Generate human-readable interpretation of the decision.
        
        Args:
            metrics: Computed structural metrics
            prediction: Model prediction
            confidence: Model confidence
            
        Returns:
            Human-readable interpretation string
        """
        interpretation_parts = []
        
        # Overall assessment
        if metrics.interpretability_score > 0.8:
            interpretation_parts.append(
                "✓ HIGHLY INTERPRETABLE: This decision is well-structured and trustworthy."
            )
        elif metrics.interpretability_score > 0.6:
            interpretation_parts.append(
                "○ MODERATELY INTERPRETABLE: This decision has reasonable structural support."
            )
        else:
            interpretation_parts.append(
                "⚠ LOW INTERPRETABILITY: This decision lacks clear structural foundation."
            )
        
        # Stability assessment
        if metrics.stability_score > self.stability_threshold:
            interpretation_parts.append(
                f"  • The decision state is STABLE (score: {metrics.stability_score:.3f})"
            )
        else:
            interpretation_parts.append(
                f"  • The decision state is UNSTABLE (score: {metrics.stability_score:.3f})"
            )
        
        # Maturity assessment
        if metrics.maturity_index > 0.7:
            interpretation_parts.append(
                f"  • The decision is MATURE and well-formed (index: {metrics.maturity_index:.3f})"
            )
        elif metrics.maturity_index > 0.4:
            interpretation_parts.append(
                f"  • The decision shows moderate maturity (index: {metrics.maturity_index:.3f})"
            )
        else:
            interpretation_parts.append(
                f"  • The decision is IMMATURE or uncertain (index: {metrics.maturity_index:.3f})"
            )
        
        # Confidence alignment
        if metrics.confidence_alignment > 0.8:
            interpretation_parts.append(
                f"  • Model confidence ({confidence:.3f}) is WELL-ALIGNED with structural stability"
            )
        elif metrics.confidence_alignment < 0.5:
            interpretation_parts.append(
                f"  • WARNING: Model confidence ({confidence:.3f}) misaligned with structure"
            )
        
        return "\n".join(interpretation_parts)
    
    def transform_to_whitebox(self,
                             state_matrix: np.ndarray,
                             confidence: float,
                             prediction: Any) -> WhiteBoxOutput:
        """
        Transform black box neural network output to white box interpretable form.
        
        This is the main entry point for the UEST framework transformation.
        
        Args:
            state_matrix: Internal state representation from NN
            confidence: Model's confidence in prediction
            prediction: The model's prediction
            
        Returns:
            WhiteBoxOutput with full structural analysis and interpretation
        """
        # Perform structural analysis
        metrics = self.analyze_decision_state(state_matrix, confidence, prediction)
        
        # Compute eigenspace representation
        eigenvalues, eigenvectors = np.linalg.eig(state_matrix)
        
        # Sort by eigenvalue magnitude
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Generate detailed stability indicators
        stability_indicators = {
            'dominant_eigenvalue': float(np.abs(eigenvalues[0])) if len(eigenvalues) > 0 else 0.0,
            'eigenvalue_range': float(np.abs(eigenvalues[0] - eigenvalues[-1])) if len(eigenvalues) > 0 else 0.0,
            'spectral_radius': float(np.max(np.abs(eigenvalues))) if len(eigenvalues) > 0 else 0.0,
            'trace': float(np.trace(state_matrix)),
            'determinant': float(np.linalg.det(state_matrix)),
            'condition_number': float(np.linalg.cond(state_matrix))
        }
        
        # Generate interpretation
        interpretation = self.generate_interpretation(metrics, prediction, confidence)
        
        return WhiteBoxOutput(
            prediction=prediction,
            confidence=confidence,
            structural_metrics=metrics,
            eigenspace_representation=eigenvalues,
            stability_indicators=stability_indicators,
            interpretation=interpretation
        )
    
    def extract_state_matrix(self, 
                            hidden_states: np.ndarray,
                            method: str = 'covariance') -> np.ndarray:
        """
        Extract state matrix from neural network hidden states.
        
        Args:
            hidden_states: Hidden layer activations from NN
            method: Extraction method ('covariance', 'correlation', 'gram')
            
        Returns:
            State matrix suitable for UEST analysis
        """
        if hidden_states.ndim == 1:
            hidden_states = hidden_states.reshape(-1, 1)
        
        if method == 'covariance':
            # Compute covariance matrix
            state_matrix = np.cov(hidden_states.T)
        elif method == 'correlation':
            # Compute correlation matrix
            state_matrix = np.corrcoef(hidden_states.T)
        elif method == 'gram':
            # Compute Gram matrix
            state_matrix = hidden_states.T @ hidden_states
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Ensure it's 2D
        if state_matrix.ndim == 0:
            state_matrix = np.array([[state_matrix]])
        elif state_matrix.ndim == 1:
            state_matrix = np.diag(state_matrix)
        
        return state_matrix
