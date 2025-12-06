"""
Example usage of the UEST Framework for Explainable AI

This script demonstrates how to transform black box neural network outputs
into white box interpretable outputs using the UEST framework.
"""

import numpy as np
from uest_framework import UESTFramework, WhiteBoxOutput


def simulate_nn_hidden_states(size: int = 10, 
                              stability: str = 'stable') -> tuple[np.ndarray, float, int]:
    """
    Simulate neural network hidden states for demonstration.
    
    Args:
        size: Dimension of hidden state
        stability: 'stable', 'unstable', or 'moderate'
        
    Returns:
        Tuple of (hidden_states, confidence, prediction)
    """
    np.random.seed(42)
    
    if stability == 'stable':
        # Create well-structured, stable hidden states
        # Dominant direction with small perturbations
        dominant_direction = np.random.randn(size)
        dominant_direction /= np.linalg.norm(dominant_direction)
        
        hidden_states = np.outer(dominant_direction, dominant_direction) * 10.0
        hidden_states += np.random.randn(size, size) * 0.1  # Small noise
        
        confidence = 0.95
        prediction = 1  # Class 1
        
    elif stability == 'unstable':
        # Create poorly structured, unstable hidden states
        # Random, no clear structure
        hidden_states = np.random.randn(size, size) * 5.0
        
        confidence = 0.55
        prediction = 0  # Class 0
        
    else:  # moderate
        # Create moderately structured hidden states
        # Mix of structure and noise
        dominant_direction = np.random.randn(size)
        dominant_direction /= np.linalg.norm(dominant_direction)
        
        hidden_states = np.outer(dominant_direction, dominant_direction) * 5.0
        hidden_states += np.random.randn(size, size) * 1.0  # Moderate noise
        
        confidence = 0.75
        prediction = 1  # Class 1
    
    return hidden_states, confidence, prediction


def demonstrate_blackbox_to_whitebox():
    """
    Main demonstration: Transform black box NN outputs to white box interpretable form.
    """
    print("=" * 80)
    print("UEST Framework Demonstration: Black Box → White Box Transformation")
    print("=" * 80)
    print()
    
    # Initialize UEST Framework
    uest = UESTFramework(dimensionality=10, stability_threshold=0.7)
    
    # Test Case 1: Stable, high-confidence decision
    print("Test Case 1: STABLE Neural Network Decision")
    print("-" * 80)
    
    hidden_states_stable, confidence_stable, prediction_stable = simulate_nn_hidden_states(
        size=10, stability='stable'
    )
    
    print(f"BLACK BOX OUTPUT:")
    print(f"  Prediction: {prediction_stable}")
    print(f"  Confidence: {confidence_stable:.4f}")
    print(f"  (No structural interpretation available)")
    print()
    
    # Extract state matrix from hidden states
    state_matrix_stable = uest.extract_state_matrix(hidden_states_stable, method='gram')
    
    # Transform to white box
    whitebox_output_stable = uest.transform_to_whitebox(
        state_matrix=state_matrix_stable,
        confidence=confidence_stable,
        prediction=prediction_stable
    )
    
    print(whitebox_output_stable)
    print()
    print()
    
    # Test Case 2: Unstable, low-confidence decision
    print("Test Case 2: UNSTABLE Neural Network Decision")
    print("-" * 80)
    
    hidden_states_unstable, confidence_unstable, prediction_unstable = simulate_nn_hidden_states(
        size=10, stability='unstable'
    )
    
    print(f"BLACK BOX OUTPUT:")
    print(f"  Prediction: {prediction_unstable}")
    print(f"  Confidence: {confidence_unstable:.4f}")
    print(f"  (No structural interpretation available)")
    print()
    
    # Extract state matrix from hidden states
    state_matrix_unstable = uest.extract_state_matrix(hidden_states_unstable, method='gram')
    
    # Transform to white box
    whitebox_output_unstable = uest.transform_to_whitebox(
        state_matrix=state_matrix_unstable,
        confidence=confidence_unstable,
        prediction=prediction_unstable
    )
    
    print(whitebox_output_unstable)
    print()
    print()
    
    # Test Case 3: Moderate stability decision
    print("Test Case 3: MODERATELY STABLE Neural Network Decision")
    print("-" * 80)
    
    hidden_states_moderate, confidence_moderate, prediction_moderate = simulate_nn_hidden_states(
        size=10, stability='moderate'
    )
    
    print(f"BLACK BOX OUTPUT:")
    print(f"  Prediction: {prediction_moderate}")
    print(f"  Confidence: {confidence_moderate:.4f}")
    print(f"  (No structural interpretation available)")
    print()
    
    # Extract state matrix from hidden states
    state_matrix_moderate = uest.extract_state_matrix(hidden_states_moderate, method='gram')
    
    # Transform to white box
    whitebox_output_moderate = uest.transform_to_whitebox(
        state_matrix=state_matrix_moderate,
        confidence=confidence_moderate,
        prediction=prediction_moderate
    )
    
    print(whitebox_output_moderate)
    print()
    print()
    
    # Summary comparison
    print("=" * 80)
    print("SUMMARY: Structural Comparison Across Decision States")
    print("=" * 80)
    print()
    
    print(f"{'Metric':<30} {'Stable':<15} {'Unstable':<15} {'Moderate':<15}")
    print("-" * 80)
    
    metrics = [
        ('Stability Score', 
         whitebox_output_stable.structural_metrics.stability_score,
         whitebox_output_unstable.structural_metrics.stability_score,
         whitebox_output_moderate.structural_metrics.stability_score),
        ('Maturity Index',
         whitebox_output_stable.structural_metrics.maturity_index,
         whitebox_output_unstable.structural_metrics.maturity_index,
         whitebox_output_moderate.structural_metrics.maturity_index),
        ('Interpretability Score',
         whitebox_output_stable.structural_metrics.interpretability_score,
         whitebox_output_unstable.structural_metrics.interpretability_score,
         whitebox_output_moderate.structural_metrics.interpretability_score),
        ('Coherence Measure',
         whitebox_output_stable.structural_metrics.coherence_measure,
         whitebox_output_unstable.structural_metrics.coherence_measure,
         whitebox_output_moderate.structural_metrics.coherence_measure),
    ]
    
    for metric_name, stable_val, unstable_val, moderate_val in metrics:
        print(f"{metric_name:<30} {stable_val:<15.4f} {unstable_val:<15.4f} {moderate_val:<15.4f}")
    
    print()
    print("=" * 80)
    print("Key Insight: The UEST Framework reveals the internal structural quality")
    print("of neural network decisions, providing transparency beyond simple confidence scores.")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_blackbox_to_whitebox()
