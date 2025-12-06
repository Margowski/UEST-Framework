# UEST-Framework

**Unified Eigenspace Structural Theory for Explainable AI (XAI 2.0)**

This UEST-Framework demonstrates the core principle of UEST-inspired Explainable AI in a **FINITE-DIMENSIONAL, DISCRETE** setting. It quantifies the internal structural stability and maturity of Neural Network decision states, transforming **"Black Box"** results into **"Structurally Interpretable White Box"** outputs.

## Overview

The UEST Framework provides a mathematically rigorous approach to making neural network decisions interpretable by analyzing their internal structure through eigenspace decomposition. Rather than treating neural networks as opaque black boxes, UEST reveals the underlying structural properties that determine decision quality.

## Key Features

- **Structural Stability Quantification**: Measures how stable and robust a neural network's decision state is
- **Decision Maturity Analysis**: Quantifies how well-formed and decisive the decision is
- **Eigenspace Representation**: Decomposes decision states into interpretable eigenspace components
- **White Box Transformation**: Converts opaque NN outputs into transparent, interpretable results
- **Confidence Alignment**: Validates whether model confidence aligns with structural stability

## Core Metrics

The framework computes six key structural metrics:

1. **Stability Score** (0-1): Overall structural stability based on eigenvalue distribution
2. **Maturity Index** (0-1): How well-formed and decisive the decision state is
3. **Eigenvalue Entropy**: Measures concentration vs. distribution of eigenvalues
4. **Coherence Measure**: Internal consistency of the decision state
5. **Confidence Alignment**: Agreement between model confidence and structural stability
6. **Interpretability Score**: Overall measure of decision interpretability

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from uest_framework import UESTFramework
import numpy as np

# Initialize the framework
uest = UESTFramework(dimensionality=10, stability_threshold=0.7)

# Simulate neural network hidden states
hidden_states = np.random.randn(10, 10)
confidence = 0.85
prediction = 1

# Extract state matrix
state_matrix = uest.extract_state_matrix(hidden_states, method='gram')

# Transform to white box output
whitebox = uest.transform_to_whitebox(
    state_matrix=state_matrix,
    confidence=confidence,
    prediction=prediction
)

# View interpretable results
print(whitebox)
```

## Example Usage

Run the demonstration script to see the framework in action:

```bash
python example_usage.py
```

This will show how UEST transforms black box neural network outputs into interpretable white box outputs with detailed structural analysis.

## How It Works

### 1. State Matrix Extraction
The framework extracts a state matrix from neural network hidden states using methods like:
- **Covariance**: Captures state correlations
- **Gram Matrix**: Measures self-similarity
- **Correlation**: Normalized relationships

### 2. Eigenspace Analysis
The state matrix is decomposed into eigenvalues and eigenvectors, revealing:
- Dominant directions in decision space
- Structural concentration vs. distribution
- Stability characteristics

### 3. Structural Metrics Computation
Multiple metrics are computed to quantify:
- **Stability**: Condition number, spectral gap, eigenvalue concentration
- **Maturity**: Confidence alignment, state concentration, eigenvalue dominance
- **Coherence**: Symmetry and positive-definiteness

### 4. White Box Output Generation
All metrics are combined with human-readable interpretations to provide:
- Transparent decision quality assessment
- Structural validation of model confidence
- Actionable insights about decision reliability

## Architecture

```
UEST Framework
├── uest_framework.py          # Core framework implementation
│   ├── UESTFramework          # Main analysis class
│   ├── StructuralMetrics      # Metrics data container
│   └── WhiteBoxOutput         # Interpretable output container
├── example_usage.py           # Demonstration script
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Use Cases

- **Model Validation**: Verify that high-confidence predictions have structural support
- **Decision Trust**: Determine when to trust a model's predictions
- **Model Debugging**: Identify when models are uncertain despite high confidence
- **Explainable AI**: Provide stakeholders with interpretable decision justification
- **Robustness Analysis**: Assess structural stability of decisions

## Mathematical Foundation

The UEST Framework is based on spectral analysis of decision state matrices. Key mathematical concepts include:

- **Eigenvalue Decomposition**: Revealing principal components of decision states
- **Spectral Gap**: Measuring separation between dominant and secondary modes
- **Condition Number**: Quantifying numerical stability
- **Shannon Entropy**: Measuring eigenvalue distribution concentration

## Contributing

Contributions are welcome! Areas for enhancement:
- Additional state matrix extraction methods
- Extended metrics for specific domains
- Integration with popular ML frameworks (PyTorch, TensorFlow)
- Visualization tools for eigenspace analysis

## License

This framework is provided for research and educational purposes.

## Citation

If you use this framework in your research, please cite:

```
UEST-Framework: Unified Eigenspace Structural Theory for Explainable AI
A finite-dimensional, discrete approach to neural network interpretability
```

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.
