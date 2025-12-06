"""
Neural Network Integration Example

This example demonstrates how to integrate the UEST Framework with
actual neural network implementations to extract hidden states and
perform white box transformation.
"""

import numpy as np
from uest_framework import UESTFramework


class SimpleNeuralNetwork:
    """
    A simple feedforward neural network for demonstration purposes.
    
    This demonstrates how to extract hidden states for UEST analysis.
    """
    
    def __init__(self, input_size: int = 10, hidden_size: int = 8, output_size: int = 3):
        """Initialize the neural network with random weights"""
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros(output_size)
        
        # Store hidden states for UEST analysis
        self.hidden_states = None
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def softmax(self, x):
        """Softmax activation for output layer"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def forward(self, X):
        """
        Forward pass through the network.
        
        Args:
            X: Input data
            
        Returns:
            Tuple of (prediction, confidence, hidden_states)
        """
        # Hidden layer
        z1 = X @ self.W1 + self.b1
        h1 = self.sigmoid(z1)
        
        # Store hidden states for UEST analysis
        self.hidden_states = h1
        
        # Output layer
        z2 = h1 @ self.W2 + self.b2
        output = self.softmax(z2)
        
        # Get prediction and confidence
        prediction = np.argmax(output)
        confidence = np.max(output)
        
        return prediction, confidence, self.hidden_states


def demonstrate_nn_integration():
    """
    Demonstrate UEST Framework integration with a neural network.
    """
    print("=" * 80)
    print("Neural Network Integration with UEST Framework")
    print("=" * 80)
    print()
    
    # Initialize neural network
    nn = SimpleNeuralNetwork(input_size=10, hidden_size=8, output_size=3)
    
    # Initialize UEST Framework
    uest = UESTFramework(dimensionality=8, stability_threshold=0.65)
    
    # Test with different input patterns
    test_cases = [
        ("Strong Signal", np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])),
        ("Weak Signal", np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])),
        ("Mixed Signal", np.array([0.8, 0.2, 0.7, 0.1, 0.6, 0.3, 0.5, 0.2, 0.4, 0.1]))
    ]
    
    for case_name, input_data in test_cases:
        print(f"\nTest Case: {case_name}")
        print("-" * 80)
        
        # Forward pass through neural network
        prediction, confidence, hidden_states = nn.forward(input_data)
        
        print(f"\nNeural Network Output (Black Box):")
        print(f"  Input: {input_data[:3]}... (showing first 3 features)")
        print(f"  Prediction Class: {prediction}")
        print(f"  Confidence: {confidence:.4f}")
        print(f"  Hidden Layer Shape: {hidden_states.shape}")
        print()
        
        # Extract state matrix from hidden states
        # Use the hidden states as a batch of 1 sample
        hidden_batch = hidden_states.reshape(1, -1)
        state_matrix = uest.extract_state_matrix(hidden_batch, method='gram')
        
        print(f"UEST Analysis (White Box):")
        print(f"  State Matrix Shape: {state_matrix.shape}")
        
        # Transform to white box
        whitebox = uest.transform_to_whitebox(
            state_matrix=state_matrix,
            confidence=confidence,
            prediction=prediction
        )
        
        # Display key metrics
        print(f"\n  Structural Metrics:")
        print(f"    Stability Score: {whitebox.structural_metrics.stability_score:.4f}")
        print(f"    Maturity Index: {whitebox.structural_metrics.maturity_index:.4f}")
        print(f"    Interpretability: {whitebox.structural_metrics.interpretability_score:.4f}")
        print(f"    Coherence: {whitebox.structural_metrics.coherence_measure:.4f}")
        print(f"    Confidence Alignment: {whitebox.structural_metrics.confidence_alignment:.4f}")
        
        print(f"\n  Eigenspace Analysis:")
        print(f"    Dominant Eigenvalue: {whitebox.stability_indicators['dominant_eigenvalue']:.4f}")
        print(f"    Spectral Radius: {whitebox.stability_indicators['spectral_radius']:.4f}")
        print(f"    Condition Number: {whitebox.stability_indicators['condition_number']:.4f}")
        
        print(f"\n  Interpretation:")
        for line in whitebox.interpretation.split('\n'):
            print(f"    {line}")
        
        print()
    
    print("=" * 80)
    print("Key Insight: UEST Framework provides structural validation of NN decisions")
    print("=" * 80)


def demonstrate_batch_analysis():
    """
    Demonstrate batch analysis of multiple predictions.
    """
    print("\n\n")
    print("=" * 80)
    print("Batch Analysis: Multiple Predictions")
    print("=" * 80)
    print()
    
    # Initialize
    nn = SimpleNeuralNetwork(input_size=10, hidden_size=8, output_size=3)
    uest = UESTFramework(dimensionality=8, stability_threshold=0.65)
    
    # Generate random test inputs
    np.random.seed(123)
    num_samples = 5
    test_inputs = np.random.randn(num_samples, 10)
    
    print(f"Analyzing {num_samples} predictions...\n")
    
    results = []
    for i, input_data in enumerate(test_inputs):
        # Get NN prediction
        prediction, confidence, hidden_states = nn.forward(input_data)
        
        # UEST analysis
        hidden_batch = hidden_states.reshape(1, -1)
        state_matrix = uest.extract_state_matrix(hidden_batch, method='gram')
        whitebox = uest.transform_to_whitebox(state_matrix, confidence, prediction)
        
        results.append({
            'sample': i + 1,
            'prediction': prediction,
            'confidence': confidence,
            'stability': whitebox.structural_metrics.stability_score,
            'maturity': whitebox.structural_metrics.maturity_index,
            'interpretability': whitebox.structural_metrics.interpretability_score
        })
    
    # Display summary table
    print(f"{'Sample':<8} {'Pred':<6} {'Conf':<8} {'Stability':<12} {'Maturity':<10} {'Interp.':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['sample']:<8} {r['prediction']:<6} {r['confidence']:<8.4f} "
              f"{r['stability']:<12.4f} {r['maturity']:<10.4f} {r['interpretability']:<10.4f}")
    
    print()
    print("Analysis:")
    high_interp = [r for r in results if r['interpretability'] > 0.7]
    low_interp = [r for r in results if r['interpretability'] < 0.5]
    
    print(f"  • {len(high_interp)}/{num_samples} predictions are highly interpretable")
    print(f"  • {len(low_interp)}/{num_samples} predictions have low interpretability")
    
    if high_interp:
        avg_conf_high = np.mean([r['confidence'] for r in high_interp])
        print(f"  • High interpretability predictions: avg confidence = {avg_conf_high:.3f}")
    
    if low_interp:
        avg_conf_low = np.mean([r['confidence'] for r in low_interp])
        print(f"  • Low interpretability predictions: avg confidence = {avg_conf_low:.3f}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_nn_integration()
    demonstrate_batch_analysis()
