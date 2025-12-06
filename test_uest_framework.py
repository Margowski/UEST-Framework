"""
Unit tests for the UEST Framework

Tests core functionality of structural stability quantification,
maturity analysis, and white box transformation.
"""

import numpy as np
import unittest
from uest_framework import (
    UESTFramework, 
    StructuralMetrics, 
    WhiteBoxOutput
)


class TestUESTFramework(unittest.TestCase):
    """Test suite for UEST Framework core functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.uest = UESTFramework(dimensionality=5, stability_threshold=0.7)
        np.random.seed(42)
    
    def test_initialization(self):
        """Test framework initialization"""
        self.assertEqual(self.uest.dimensionality, 5)
        self.assertEqual(self.uest.stability_threshold, 0.7)
    
    def test_eigenvalue_entropy_zero(self):
        """Test entropy computation with zero eigenvalues"""
        eigenvalues = np.zeros(5)
        entropy = self.uest.compute_eigenvalue_entropy(eigenvalues)
        self.assertEqual(entropy, 0.0)
    
    def test_eigenvalue_entropy_uniform(self):
        """Test entropy with uniform eigenvalue distribution"""
        eigenvalues = np.ones(5)
        entropy = self.uest.compute_eigenvalue_entropy(eigenvalues)
        # Uniform distribution should have maximum normalized entropy
        self.assertAlmostEqual(entropy, 1.0, places=5)
    
    def test_eigenvalue_entropy_concentrated(self):
        """Test entropy with concentrated eigenvalues"""
        eigenvalues = np.array([10.0, 0.1, 0.1, 0.1, 0.1])
        entropy = self.uest.compute_eigenvalue_entropy(eigenvalues)
        # Concentrated distribution should have low entropy
        self.assertLess(entropy, 0.5)
    
    def test_stability_score_identity(self):
        """Test stability score with identity matrix"""
        state_matrix = np.eye(5)
        stability = self.uest.compute_stability_score(state_matrix)
        # Identity matrix has uniform eigenvalues, so moderate stability score
        self.assertGreaterEqual(stability, 0.0)
        self.assertLessEqual(stability, 1.0)
    
    def test_stability_score_random(self):
        """Test stability score with random matrix"""
        state_matrix = np.random.randn(5, 5)
        stability = self.uest.compute_stability_score(state_matrix)
        # Should be in valid range
        self.assertGreaterEqual(stability, 0.0)
        self.assertLessEqual(stability, 1.0)
    
    def test_maturity_index_high_confidence(self):
        """Test maturity with high confidence and structured state"""
        # Create well-structured state matrix
        v = np.random.randn(5)
        v = v / np.linalg.norm(v)
        state_matrix = np.outer(v, v) * 10.0
        
        maturity = self.uest.compute_maturity_index(state_matrix, confidence=0.95)
        # Should have high maturity
        self.assertGreater(maturity, 0.5)
    
    def test_maturity_index_low_confidence(self):
        """Test maturity with low confidence"""
        state_matrix = np.random.randn(5, 5)
        maturity = self.uest.compute_maturity_index(state_matrix, confidence=0.3)
        # Should have lower maturity
        self.assertLess(maturity, 0.8)
    
    def test_coherence_symmetric(self):
        """Test coherence with symmetric matrix"""
        state_matrix = np.random.randn(5, 5)
        state_matrix = (state_matrix + state_matrix.T) / 2  # Make symmetric
        coherence = self.uest.compute_coherence(state_matrix)
        # Symmetric matrix should have high coherence
        self.assertGreater(coherence, 0.5)
    
    def test_coherence_asymmetric(self):
        """Test coherence with highly asymmetric matrix"""
        state_matrix = np.random.randn(5, 5) * 10
        coherence = self.uest.compute_coherence(state_matrix)
        # Should still be in valid range
        self.assertGreaterEqual(coherence, 0.0)
        self.assertLessEqual(coherence, 1.0)
    
    def test_confidence_alignment_matched(self):
        """Test alignment when confidence matches stability"""
        # Create stable state matrix with dominant eigenvalue
        v = np.random.randn(5)
        v = v / np.linalg.norm(v)
        state_matrix = np.outer(v, v) * 10.0 + np.eye(5) * 0.1
        confidence = 0.8
        
        alignment = self.uest.compute_confidence_alignment(state_matrix, confidence)
        # Should have reasonable alignment
        self.assertGreater(alignment, 0.3)
    
    def test_confidence_alignment_mismatched(self):
        """Test alignment when confidence doesn't match stability"""
        # Create unstable state matrix
        state_matrix = np.random.randn(5, 5) * 0.1
        confidence = 0.95  # High confidence despite instability
        
        alignment = self.uest.compute_confidence_alignment(state_matrix, confidence)
        # Should be in valid range
        self.assertGreaterEqual(alignment, 0.0)
        self.assertLessEqual(alignment, 1.0)
    
    def test_analyze_decision_state(self):
        """Test full decision state analysis"""
        state_matrix = np.eye(5) * 3
        confidence = 0.85
        prediction = 1
        
        metrics = self.uest.analyze_decision_state(state_matrix, confidence, prediction)
        
        # Check that all metrics are computed
        self.assertIsInstance(metrics, StructuralMetrics)
        self.assertIsNotNone(metrics.stability_score)
        self.assertIsNotNone(metrics.maturity_index)
        self.assertIsNotNone(metrics.eigenvalue_entropy)
        self.assertIsNotNone(metrics.coherence_measure)
        self.assertIsNotNone(metrics.confidence_alignment)
        self.assertIsNotNone(metrics.interpretability_score)
        
        # All metrics should be in [0, 1]
        self.assertGreaterEqual(metrics.stability_score, 0.0)
        self.assertLessEqual(metrics.stability_score, 1.0)
        self.assertGreaterEqual(metrics.maturity_index, 0.0)
        self.assertLessEqual(metrics.maturity_index, 1.0)
        self.assertGreaterEqual(metrics.interpretability_score, 0.0)
        self.assertLessEqual(metrics.interpretability_score, 1.0)
    
    def test_generate_interpretation(self):
        """Test interpretation generation"""
        metrics = StructuralMetrics(
            stability_score=0.85,
            maturity_index=0.90,
            eigenvalue_entropy=0.2,
            coherence_measure=0.95,
            confidence_alignment=0.88,
            interpretability_score=0.87
        )
        
        interpretation = self.uest.generate_interpretation(metrics, prediction=1, confidence=0.9)
        
        # Should contain key information
        self.assertIsInstance(interpretation, str)
        self.assertGreater(len(interpretation), 50)
        self.assertIn('STABLE', interpretation.upper())
    
    def test_transform_to_whitebox(self):
        """Test complete white box transformation"""
        state_matrix = np.eye(5) * 2
        confidence = 0.88
        prediction = 'Class A'
        
        whitebox = self.uest.transform_to_whitebox(state_matrix, confidence, prediction)
        
        # Check output structure
        self.assertIsInstance(whitebox, WhiteBoxOutput)
        self.assertEqual(whitebox.prediction, 'Class A')
        self.assertEqual(whitebox.confidence, 0.88)
        self.assertIsInstance(whitebox.structural_metrics, StructuralMetrics)
        self.assertIsInstance(whitebox.eigenspace_representation, np.ndarray)
        self.assertIsInstance(whitebox.stability_indicators, dict)
        self.assertIsInstance(whitebox.interpretation, str)
        
        # Check stability indicators
        self.assertIn('dominant_eigenvalue', whitebox.stability_indicators)
        self.assertIn('trace', whitebox.stability_indicators)
        self.assertIn('determinant', whitebox.stability_indicators)
    
    def test_extract_state_matrix_covariance(self):
        """Test state matrix extraction using covariance"""
        hidden_states = np.random.randn(10, 5)
        state_matrix = self.uest.extract_state_matrix(hidden_states, method='covariance')
        
        # Should be square
        self.assertEqual(state_matrix.shape[0], state_matrix.shape[1])
        # Should be 5x5 for 5 features
        self.assertEqual(state_matrix.shape[0], 5)
    
    def test_extract_state_matrix_correlation(self):
        """Test state matrix extraction using correlation"""
        hidden_states = np.random.randn(10, 5)
        state_matrix = self.uest.extract_state_matrix(hidden_states, method='correlation')
        
        # Should be square
        self.assertEqual(state_matrix.shape[0], state_matrix.shape[1])
        self.assertEqual(state_matrix.shape[0], 5)
    
    def test_extract_state_matrix_gram(self):
        """Test state matrix extraction using Gram matrix"""
        hidden_states = np.random.randn(10, 5)
        state_matrix = self.uest.extract_state_matrix(hidden_states, method='gram')
        
        # Should be square
        self.assertEqual(state_matrix.shape[0], state_matrix.shape[1])
        self.assertEqual(state_matrix.shape[0], 5)
    
    def test_extract_state_matrix_1d(self):
        """Test state matrix extraction with 1D input"""
        hidden_states = np.random.randn(10)
        state_matrix = self.uest.extract_state_matrix(hidden_states, method='covariance')
        
        # Should create 2D matrix
        self.assertEqual(len(state_matrix.shape), 2)
    
    def test_extract_state_matrix_invalid_method(self):
        """Test that invalid method raises error"""
        hidden_states = np.random.randn(10, 5)
        
        with self.assertRaises(ValueError):
            self.uest.extract_state_matrix(hidden_states, method='invalid')
    
    def test_structural_metrics_string_representation(self):
        """Test StructuralMetrics string representation"""
        metrics = StructuralMetrics(
            stability_score=0.85,
            maturity_index=0.90,
            eigenvalue_entropy=0.2,
            coherence_measure=0.95,
            confidence_alignment=0.88,
            interpretability_score=0.87
        )
        
        str_repr = str(metrics)
        self.assertIn('Stability Score', str_repr)
        self.assertIn('0.8500', str_repr)
    
    def test_whitebox_output_string_representation(self):
        """Test WhiteBoxOutput string representation"""
        metrics = StructuralMetrics(
            stability_score=0.85,
            maturity_index=0.90,
            eigenvalue_entropy=0.2,
            coherence_measure=0.95,
            confidence_alignment=0.88,
            interpretability_score=0.87
        )
        
        whitebox = WhiteBoxOutput(
            prediction='Test',
            confidence=0.9,
            structural_metrics=metrics,
            eigenspace_representation=np.array([1.0, 0.5]),
            stability_indicators={'test': 1.0},
            interpretation='Test interpretation'
        )
        
        str_repr = str(whitebox)
        self.assertIn('White Box', str_repr)
        self.assertIn('Test', str_repr)
        self.assertIn('0.9000', str_repr)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.uest = UESTFramework(dimensionality=10)
        np.random.seed(42)
    
    def test_complete_workflow_stable(self):
        """Test complete workflow with stable decision"""
        # Create stable hidden states
        v = np.random.randn(10)
        v = v / np.linalg.norm(v)
        hidden_states = np.outer(v, v) * 10.0
        
        # Extract state matrix
        state_matrix = self.uest.extract_state_matrix(hidden_states, method='gram')
        
        # Transform to white box
        whitebox = self.uest.transform_to_whitebox(
            state_matrix=state_matrix,
            confidence=0.95,
            prediction=1
        )
        
        # Verify high interpretability for stable state
        self.assertGreater(whitebox.structural_metrics.interpretability_score, 0.5)
    
    def test_complete_workflow_unstable(self):
        """Test complete workflow with unstable decision"""
        # Create unstable hidden states
        hidden_states = np.random.randn(10, 10) * 0.1
        
        # Extract state matrix
        state_matrix = self.uest.extract_state_matrix(hidden_states, method='covariance')
        
        # Transform to white box
        whitebox = self.uest.transform_to_whitebox(
            state_matrix=state_matrix,
            confidence=0.55,
            prediction=0
        )
        
        # Verify reasonable output
        self.assertGreaterEqual(whitebox.structural_metrics.interpretability_score, 0.0)
        self.assertLessEqual(whitebox.structural_metrics.interpretability_score, 1.0)


if __name__ == '__main__':
    unittest.main()
