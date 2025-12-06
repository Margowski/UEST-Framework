"""
Model Comparison using UEST Framework

This example demonstrates how to use UEST metrics to compare
and select between different neural network models or configurations.
"""

import numpy as np
from uest_framework import UESTFramework


def simulate_model_predictions(model_type: str, num_samples: int = 10):
    """
    Simulate predictions from different model types.
    
    Args:
        model_type: 'well_trained', 'undertrained', or 'overfitted'
        num_samples: Number of predictions to simulate
        
    Returns:
        List of (hidden_states, confidence, prediction) tuples
    """
    np.random.seed(42)
    results = []
    
    for i in range(num_samples):
        if model_type == 'well_trained':
            # Well-trained model: high confidence, stable structure
            dominant_direction = np.random.randn(10)
            dominant_direction /= np.linalg.norm(dominant_direction)
            hidden_states = dominant_direction * 8.0 + np.random.randn(10) * 0.5
            confidence = 0.85 + np.random.rand() * 0.1
            prediction = 1
            
        elif model_type == 'undertrained':
            # Undertrained model: low confidence, weak structure
            hidden_states = np.random.randn(10) * 2.0
            confidence = 0.45 + np.random.rand() * 0.15
            prediction = 0
            
        else:  # overfitted
            # Overfitted model: high confidence but unstable structure
            hidden_states = np.random.randn(10) * 3.0
            hidden_states[0] *= 5.0  # Artificial spike
            confidence = 0.90 + np.random.rand() * 0.08
            prediction = 1
        
        results.append((hidden_states, confidence, prediction))
    
    return results


def analyze_model(model_name: str, predictions: list, uest: UESTFramework):
    """
    Analyze a model using UEST metrics.
    
    Args:
        model_name: Name of the model
        predictions: List of (hidden_states, confidence, prediction) tuples
        uest: UEST Framework instance
        
    Returns:
        Dictionary of aggregated metrics
    """
    metrics_list = []
    
    for hidden_states, confidence, prediction in predictions:
        # Extract state matrix
        hidden_batch = hidden_states.reshape(1, -1)
        state_matrix = uest.extract_state_matrix(hidden_batch, method='covariance')
        
        # Transform to white box
        whitebox = uest.transform_to_whitebox(state_matrix, confidence, prediction)
        
        metrics_list.append({
            'stability': whitebox.structural_metrics.stability_score,
            'maturity': whitebox.structural_metrics.maturity_index,
            'interpretability': whitebox.structural_metrics.interpretability_score,
            'coherence': whitebox.structural_metrics.coherence_measure,
            'confidence_alignment': whitebox.structural_metrics.confidence_alignment,
            'confidence': confidence
        })
    
    # Aggregate metrics
    aggregated = {
        'model_name': model_name,
        'num_predictions': len(predictions),
        'avg_stability': np.mean([m['stability'] for m in metrics_list]),
        'std_stability': np.std([m['stability'] for m in metrics_list]),
        'avg_maturity': np.mean([m['maturity'] for m in metrics_list]),
        'avg_interpretability': np.mean([m['interpretability'] for m in metrics_list]),
        'std_interpretability': np.std([m['interpretability'] for m in metrics_list]),
        'avg_coherence': np.mean([m['coherence'] for m in metrics_list]),
        'avg_confidence_alignment': np.mean([m['confidence_alignment'] for m in metrics_list]),
        'avg_confidence': np.mean([m['confidence'] for m in metrics_list]),
        'high_interp_ratio': sum(1 for m in metrics_list if m['interpretability'] > 0.7) / len(metrics_list)
    }
    
    return aggregated


def demonstrate_model_comparison():
    """
    Demonstrate model comparison using UEST Framework.
    """
    print("=" * 80)
    print("Model Comparison using UEST Framework")
    print("=" * 80)
    print()
    print("Comparing three different model configurations:")
    print("  1. Well-Trained Model: Good generalization, stable structure")
    print("  2. Undertrained Model: Insufficient learning, weak structure")
    print("  3. Overfitted Model: High confidence but unstable structure")
    print()
    
    # Initialize UEST Framework
    uest = UESTFramework(dimensionality=10, stability_threshold=0.7)
    
    # Simulate predictions from different models
    num_samples = 20
    models = {
        'Well-Trained': simulate_model_predictions('well_trained', num_samples),
        'Undertrained': simulate_model_predictions('undertrained', num_samples),
        'Overfitted': simulate_model_predictions('overfitted', num_samples)
    }
    
    # Analyze each model
    results = []
    for model_name, predictions in models.items():
        print(f"Analyzing {model_name} Model...")
        analysis = analyze_model(model_name, predictions, uest)
        results.append(analysis)
    
    print("\n" + "=" * 80)
    print("Model Comparison Results")
    print("=" * 80)
    print()
    
    # Display comparison table
    print(f"{'Metric':<30} {'Well-Trained':<15} {'Undertrained':<15} {'Overfitted':<15}")
    print("-" * 80)
    
    metrics_to_compare = [
        ('Average Confidence', 'avg_confidence'),
        ('Average Stability', 'avg_stability'),
        ('Stability Std Dev', 'std_stability'),
        ('Average Maturity', 'avg_maturity'),
        ('Average Interpretability', 'avg_interpretability'),
        ('Interpretability Std Dev', 'std_interpretability'),
        ('Average Coherence', 'avg_coherence'),
        ('Confidence Alignment', 'avg_confidence_alignment'),
        ('High Interpretability %', 'high_interp_ratio')
    ]
    
    for metric_name, metric_key in metrics_to_compare:
        well_trained_val = results[0][metric_key]
        undertrained_val = results[1][metric_key]
        overfitted_val = results[2][metric_key]
        
        if metric_key == 'high_interp_ratio':
            print(f"{metric_name:<30} {well_trained_val*100:<14.1f}% "
                  f"{undertrained_val*100:<14.1f}% {overfitted_val*100:<14.1f}%")
        else:
            print(f"{metric_name:<30} {well_trained_val:<15.4f} "
                  f"{undertrained_val:<15.4f} {overfitted_val:<15.4f}")
    
    print()
    print("=" * 80)
    print("UEST Framework Analysis & Recommendations")
    print("=" * 80)
    print()
    
    # Determine best model based on UEST metrics
    best_idx = np.argmax([r['avg_interpretability'] for r in results])
    best_model = results[best_idx]['model_name']
    
    print(f"Recommended Model: {best_model}")
    print()
    print("Reasoning:")
    print()
    
    # Well-Trained analysis
    print("1. Well-Trained Model:")
    if results[0]['avg_interpretability'] > 0.7:
        print("   ✓ HIGH interpretability - decisions are structurally sound")
    else:
        print("   ○ MODERATE interpretability")
    
    if results[0]['avg_stability'] > 0.6:
        print("   ✓ GOOD stability - predictions are robust")
    else:
        print("   ⚠ LOW stability")
    
    if results[0]['avg_confidence_alignment'] > 0.75:
        print("   ✓ STRONG confidence-structure alignment")
    else:
        print("   ○ Moderate confidence-structure alignment")
    print()
    
    # Undertrained analysis
    print("2. Undertrained Model:")
    if results[1]['avg_interpretability'] < 0.5:
        print("   ⚠ LOW interpretability - model hasn't learned clear patterns")
    else:
        print("   ○ Moderate interpretability")
    
    if results[1]['avg_maturity'] < 0.6:
        print("   ⚠ LOW maturity - decisions are not well-formed")
    else:
        print("   ○ Moderate maturity")
    
    if results[1]['avg_confidence'] < 0.6:
        print("   ⚠ LOW confidence - model is uncertain")
    print()
    
    # Overfitted analysis
    print("3. Overfitted Model:")
    if results[2]['avg_confidence'] > 0.85:
        print("   ○ HIGH confidence - but may be overconfident")
    
    if results[2]['avg_confidence_alignment'] < 0.7:
        print("   ⚠ POOR confidence-structure alignment")
        print("     → High confidence NOT backed by stable structure")
        print("     → Classic sign of overfitting")
    
    if results[2]['std_interpretability'] > results[0]['std_interpretability']:
        print("   ⚠ HIGH interpretability variance")
        print("     → Inconsistent structural quality across predictions")
    print()
    
    print("=" * 80)
    print("Key Insight: UEST metrics reveal structural issues beyond accuracy")
    print("=" * 80)
    print()
    print("Traditional metrics (accuracy, confidence) might favor the overfitted model,")
    print("but UEST reveals its structural instability and misalignment.")
    print("The well-trained model shows the best balance of confidence and structure.")
    print()


def demonstrate_early_stopping():
    """
    Demonstrate using UEST for early stopping during training.
    """
    print("\n\n")
    print("=" * 80)
    print("Using UEST for Early Stopping / Training Monitoring")
    print("=" * 80)
    print()
    
    uest = UESTFramework(dimensionality=10, stability_threshold=0.7)
    
    # Simulate training epochs
    epochs = [1, 5, 10, 15, 20, 25, 30]
    
    print("Simulating model training across epochs...")
    print()
    print(f"{'Epoch':<8} {'Confidence':<12} {'Stability':<12} {'Interp.':<12} {'Status':<20}")
    print("-" * 80)
    
    for epoch in epochs:
        # Simulate improving then degrading performance (overfitting)
        if epoch <= 20:
            # Improving phase
            improvement = epoch / 20.0
            v = np.random.randn(10)
            v = v / np.linalg.norm(v)
            hidden = v * (5.0 + improvement * 3.0) + np.random.randn(10) * (1.0 - improvement * 0.5)
            conf = 0.6 + improvement * 0.25
        else:
            # Overfitting phase
            overfit_factor = (epoch - 20) / 10.0
            hidden = np.random.randn(10) * (2.0 + overfit_factor * 2.0)
            hidden[0] *= (1.0 + overfit_factor * 3.0)
            conf = 0.85 + overfit_factor * 0.1
        
        # Analyze
        hidden_batch = hidden.reshape(1, -1)
        state_matrix = uest.extract_state_matrix(hidden_batch, method='covariance')
        whitebox = uest.transform_to_whitebox(state_matrix, conf, 1)
        
        stability = whitebox.structural_metrics.stability_score
        interp = whitebox.structural_metrics.interpretability_score
        
        # Determine status
        if interp > 0.75 and stability > 0.65:
            status = "✓ Good"
        elif interp > 0.6:
            status = "○ Acceptable"
        else:
            status = "⚠ Warning"
        
        if epoch == 20:
            status += " (STOP HERE)"
        
        print(f"{epoch:<8} {conf:<12.4f} {stability:<12.4f} {interp:<12.4f} {status:<20}")
    
    print()
    print("Recommendation: Stop training at epoch ~20")
    print("  • Interpretability and stability peak around epoch 20")
    print("  • After epoch 20, confidence increases but structural quality degrades")
    print("  • This is a clear sign of overfitting that traditional metrics might miss")
    print()
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_model_comparison()
    demonstrate_early_stopping()
