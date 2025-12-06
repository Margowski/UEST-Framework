# UEST Framework API Reference

## Table of Contents
- [Core Classes](#core-classes)
- [Main Methods](#main-methods)
- [Metrics Explained](#metrics-explained)
- [Usage Patterns](#usage-patterns)

---

## Core Classes

### `UESTFramework`

The main class for performing UEST analysis.

**Constructor:**
```python
UESTFramework(dimensionality: int = 10, stability_threshold: float = 0.7)
```

**Parameters:**
- `dimensionality`: Expected dimensionality of decision state (default: 10)
- `stability_threshold`: Threshold for considering a state stable (default: 0.7)

---

### `StructuralMetrics`

Data container for structural analysis results.

**Attributes:**
- `stability_score` (float): Overall stability measure [0, 1]
- `maturity_index` (float): Decision maturity level [0, 1]
- `eigenvalue_entropy` (float): Entropy of eigenvalue distribution [0, 1]
- `coherence_measure` (float): Internal state coherence [0, 1]
- `confidence_alignment` (float): Alignment between confidence and structure [0, 1]
- `interpretability_score` (float): Overall interpretability measure [0, 1]

---

### `WhiteBoxOutput`

Complete white box representation of a neural network decision.

**Attributes:**
- `prediction`: Original prediction (any type)
- `confidence` (float): Prediction confidence
- `structural_metrics` (StructuralMetrics): UEST structural analysis
- `eigenspace_representation` (np.ndarray): Eigenspace decomposition
- `stability_indicators` (Dict[str, float]): Detailed stability measures
- `interpretation` (str): Human-readable interpretation

---

## Main Methods

### `transform_to_whitebox()`

Transform black box NN output to white box interpretable form.

**Signature:**
```python
def transform_to_whitebox(
    state_matrix: np.ndarray,
    confidence: float,
    prediction: Any
) -> WhiteBoxOutput
```

**Parameters:**
- `state_matrix`: Internal state representation from NN (NxN matrix)
- `confidence`: Model's confidence in prediction [0, 1]
- `prediction`: The model's prediction (any type)

**Returns:** `WhiteBoxOutput` with complete structural analysis

**Example:**
```python
uest = UESTFramework()
whitebox = uest.transform_to_whitebox(
    state_matrix=state_matrix,
    confidence=0.85,
    prediction="Class A"
)
print(whitebox)
```

---

### `extract_state_matrix()`

Extract state matrix from neural network hidden states.

**Signature:**
```python
def extract_state_matrix(
    hidden_states: np.ndarray,
    method: str = 'covariance'
) -> np.ndarray
```

**Parameters:**
- `hidden_states`: Hidden layer activations from NN
- `method`: Extraction method - one of:
  - `'covariance'`: Covariance matrix (default)
  - `'correlation'`: Correlation matrix
  - `'gram'`: Gram matrix (inner product)

**Returns:** State matrix (NxN) suitable for UEST analysis

**Example:**
```python
uest = UESTFramework()
hidden_states = np.random.randn(10, 8)  # 10 samples, 8 features
state_matrix = uest.extract_state_matrix(hidden_states, method='gram')
```

---

### `analyze_decision_state()`

Perform comprehensive UEST analysis of a decision state.

**Signature:**
```python
def analyze_decision_state(
    state_matrix: np.ndarray,
    confidence: float,
    prediction: Any
) -> StructuralMetrics
```

**Parameters:**
- `state_matrix`: Internal state representation matrix
- `confidence`: Model's prediction confidence
- `prediction`: The model's prediction

**Returns:** `StructuralMetrics` object with all computed metrics

---

## Metrics Explained

### Stability Score (0-1)

**What it measures:** Structural stability of the decision state

**High score means:**
- Decision is robust to small perturbations
- Eigenvalues are well-concentrated
- Good spectral gap between dominant and secondary modes

**Low score means:**
- Decision may be fragile
- Multiple competing directions in decision space
- Potential instability

**Use case:** Determine if you can trust the decision under slight input variations

---

### Maturity Index (0-1)

**What it measures:** How well-formed and decisive the decision is

**High score means:**
- Decision is clear and unambiguous
- Internal state strongly supports the prediction
- Model has "made up its mind"

**Low score means:**
- Decision is uncertain or ambiguous
- Model may benefit from more training or better features

**Use case:** Identify when model needs more training or when inputs are ambiguous

---

### Eigenvalue Entropy (0-1)

**What it measures:** Distribution of eigenvalues

**High entropy (near 1):**
- Eigenvalues are uniformly distributed
- No single dominant direction
- May indicate uncertainty

**Low entropy (near 0):**
- Eigenvalues are concentrated
- Clear dominant direction(s)
- More interpretable structure

**Use case:** Understand the complexity of the decision structure

---

### Coherence Measure (0-1)

**What it measures:** Internal consistency of the decision state

**High score means:**
- State matrix is symmetric
- Positive eigenvalues dominate
- Internally consistent

**Low score means:**
- Asymmetric or inconsistent structure
- Potential numerical issues

**Use case:** Validate that the internal representation makes mathematical sense

---

### Confidence Alignment (0-1)

**What it measures:** Agreement between model confidence and structural stability

**High score means:**
- Model's stated confidence matches structural quality
- Trustworthy confidence estimates

**Low score means:**
- Misalignment between confidence and structure
- Model may be overconfident or underconfident
- **Warning sign for overfitting or calibration issues**

**Use case:** Detect when model confidence should not be trusted

---

### Interpretability Score (0-1)

**What it measures:** Overall interpretability (weighted combination of all metrics)

**Formula:**
```
0.25 * stability +
0.25 * maturity +
0.20 * (1 - entropy) +
0.15 * coherence +
0.15 * confidence_alignment
```

**High score (>0.8):**
- Decision is highly interpretable
- Safe to trust and explain to stakeholders

**Medium score (0.6-0.8):**
- Reasonable interpretability
- Use with appropriate caution

**Low score (<0.6):**
- Poor interpretability
- Requires further investigation

**Use case:** Single metric for overall decision quality

---

## Usage Patterns

### Pattern 1: Basic Analysis

```python
from uest_framework import UESTFramework
import numpy as np

# Initialize
uest = UESTFramework()

# Your neural network produces hidden states
hidden_states = your_nn.get_hidden_states(input_data)
confidence = your_nn.predict_proba(input_data).max()
prediction = your_nn.predict(input_data)

# Extract and analyze
state_matrix = uest.extract_state_matrix(hidden_states)
whitebox = uest.transform_to_whitebox(state_matrix, confidence, prediction)

# Check interpretability
if whitebox.structural_metrics.interpretability_score > 0.8:
    print("High quality decision!")
else:
    print("Warning: Low interpretability")
```

---

### Pattern 2: Model Comparison

```python
uest = UESTFramework()

models = [model_a, model_b, model_c]
results = []

for model in models:
    # Get predictions and hidden states
    hidden = model.get_hidden_states(X_test)
    conf = model.predict_proba(X_test).max(axis=1).mean()
    
    # Analyze
    state_matrix = uest.extract_state_matrix(hidden)
    wb = uest.transform_to_whitebox(state_matrix, conf, None)
    
    results.append({
        'model': model,
        'interpretability': wb.structural_metrics.interpretability_score,
        'stability': wb.structural_metrics.stability_score
    })

# Select best model by interpretability
best = max(results, key=lambda x: x['interpretability'])
print(f"Best model: {best['model']}")
```

---

### Pattern 3: Early Stopping

```python
uest = UESTFramework()

for epoch in range(max_epochs):
    # Train
    model.train_epoch()
    
    # Evaluate on validation set
    hidden = model.get_hidden_states(X_val)
    conf = model.predict_proba(X_val).max(axis=1).mean()
    
    # UEST analysis
    state_matrix = uest.extract_state_matrix(hidden)
    wb = uest.transform_to_whitebox(state_matrix, conf, None)
    
    interp = wb.structural_metrics.interpretability_score
    alignment = wb.structural_metrics.confidence_alignment
    
    # Stop if overfitting detected
    if conf > 0.9 and alignment < 0.6:
        print(f"Overfitting detected at epoch {epoch}")
        break
```

---

### Pattern 4: Confidence Calibration Check

```python
uest = UESTFramework()

# Analyze predictions
for sample in test_samples:
    hidden = model.get_hidden_states(sample)
    conf = model.predict_proba(sample).max()
    pred = model.predict(sample)
    
    state_matrix = uest.extract_state_matrix(hidden.reshape(1, -1))
    wb = uest.transform_to_whitebox(state_matrix, conf, pred)
    
    # Check calibration
    alignment = wb.structural_metrics.confidence_alignment
    
    if conf > 0.9 and alignment < 0.5:
        print(f"WARNING: High confidence ({conf:.2f}) not supported by structure")
        print("Recommendation: Don't trust this prediction")
```

---

## Best Practices

1. **Always check confidence alignment** for high-stakes decisions
2. **Use multiple extraction methods** and compare results
3. **Set appropriate thresholds** based on your domain requirements
4. **Monitor trends over time** rather than single predictions
5. **Combine UEST metrics with traditional metrics** for comprehensive evaluation

---

## Troubleshooting

**Problem:** Low interpretability scores across all predictions

**Solutions:**
- Model may need more training
- Check data quality and feature engineering
- Consider model architecture changes

---

**Problem:** High confidence but low confidence alignment

**Solutions:**
- Model is likely overfitted
- Apply regularization
- Collect more diverse training data
- Consider early stopping

---

**Problem:** NaN or Inf in state matrices

**Solutions:**
- Check for numerical overflow in hidden states
- Normalize hidden states before extraction
- Use correlation method instead of covariance
- Add small regularization to state matrix

---

## Advanced Topics

### Custom Metrics

You can extend the framework by computing additional metrics:

```python
uest = UESTFramework()
state_matrix = uest.extract_state_matrix(hidden_states)

# Compute custom metrics
eigenvalues = np.linalg.eigvals(state_matrix)
your_custom_metric = your_metric_function(eigenvalues)
```

### Batch Processing

For efficient batch analysis:

```python
uest = UESTFramework()
metrics_batch = []

for hidden_states, conf, pred in batch:
    state_matrix = uest.extract_state_matrix(hidden_states)
    wb = uest.transform_to_whitebox(state_matrix, conf, pred)
    metrics_batch.append(wb.structural_metrics)

# Aggregate statistics
avg_interp = np.mean([m.interpretability_score for m in metrics_batch])
```

---

## References

- README.md: Overview and installation
- example_usage.py: Basic demonstration
- nn_integration_example.py: Neural network integration
- model_comparison_example.py: Model comparison and early stopping
