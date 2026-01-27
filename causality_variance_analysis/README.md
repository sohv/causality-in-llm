# Causal Discovery Variance Analysis

## Overview

This codebase addresses a critical methodological flaw in the original paper: **treating single algorithm runs as "ground truth" when causal discovery algorithms are inherently stochastic.**

### Key Improvements

1. **100+ runs per algorithm** with different random seeds and hyperparameters
2. **95% confidence intervals** using bootstrap methods
3. **Proper comparison framework** - algorithmic variance vs LLM prediction ranges
4. **Overlap analysis** - quantifying when LLMs are "correct" vs "incorrect"
5. **Baseline comparisons** - random guessing, size-based heuristics

## Installation

```bash
# Clone repository
cd causality_variance_analysis

# Install dependencies
pip install -r requirements.txt --break-system-packages

# Verify installation
python -c "import causal_learn; print('causal-learn installed')"
```

## Quick Start

### Run All Experiments (3-4 hours on standard CPU)

```bash
python run_experiments.py --runs 100 --output results
```

### Run Specific Experiments

```bash
# Titanic only
python run_experiments.py --experiments titanic --runs 100

# Benchmarks only
python run_experiments.py --experiments benchmarks --runs 50

# Synthetic DAGs only
python run_experiments.py --experiments synthetic --runs 100
```

### Generate Visualizations

```bash
python visualize_results.py --results results --output plots
```

This creates:
- `precision_comparison.png` - Algorithmic CI vs LLM ranges
- `recall_comparison.png`
- `f1_comparison.png`
- `shd_comparison.png`
- `overlap_heatmap.png` - Overlap percentage across all experiments
- `shd_vs_complexity.png` - Performance degradation with graph size
- `results_summary.csv` - LaTeX table for paper

## Results Structure

```
results/
├── titanic_lingam_variance.json           # Algorithm results (100 runs)
├── titanic_lingam_llm_comparison.json     # Overlap with LLM estimates
├── asia_pc_variance.json
├── asia_lingam_variance.json
├── synthetic_12_pc_variance.json
└── ...
```

### JSON Format

```json
{
  "dataset": "titanic",
  "algorithm": "lingam",
  "n_runs": 100,
  "results": {
    "precision": {
      "mean": 0.8461,
      "std": 0.0234,
      "ci_95_lower": 0.8105,
      "ci_95_upper": 0.8817,
      "median": 0.8450,
      "min": 0.7800,
      "max": 0.9200,
      "n_runs": 100
    },
    ...
  }
}
```

## What This Fixes

### Original Paper Flaw

```python
# WRONG - single run treated as truth
true_shd = run_algorithm_once(data)  # e.g., 46
llm_estimate = (3, 7)
conclusion = "LLM underestimated by 7x"  # MISLEADING
```

### Corrected Approach

```python
# CORRECT - 100 runs with variance
shd_values = [run_algorithm(data, seed=i) for i in range(100)]
mean_shd = 46.3
ci_95 = (38.1, 54.7)
llm_estimate = (3, 7)

# Proper comparison
overlaps = llm_estimate[1] >= ci_95[0]  # False
conclusion = "LLM range does not overlap with algorithmic 95% CI"
```

## Key Findings (Expected)

Based on proper variance analysis, we expect to find:

1. **High algorithmic variance** - PC and LiNGAM show SHD std ≈ 20-40% of mean
2. **Some LLM estimates are accurate** - GPT-5 ranges overlap with CIs for simple datasets
3. **LLMs fail on assumption violations** - LiNGAM on discrete data → complete failure
4. **Systematic optimism bias** - LLMs underestimate SHD by 2-3x on average

## Experimental Design

### Datasets

| Dataset | Nodes | Edges | Type | Purpose |
|---------|-------|-------|------|---------|
| Titanic | 7 | 5 | Real | Social science data |
| ASIA | 8 | 8 | Benchmark | Medical diagnosis |
| CANCER | 5 | 4 | Benchmark | Simple probabilistic |
| EARTHQUAKE | 5 | 4 | Benchmark | Alarm system |
| SACHS | 11 | 17 | Benchmark | Protein signaling |
| SURVEY | 6 | 6 | Benchmark | Survey responses |
| CHILD | 20 | 25 | Benchmark | Medical diagnosis |
| Synthetic-12 | 12 | ~14 | Synthetic | Medium complexity |
| Synthetic-30 | 30 | ~45 | Synthetic | High complexity |

### Algorithms

- **LiNGAM**: Linear non-Gaussian acyclic model
  - Assumes: Linearity, non-Gaussian errors, no latent confounders
  - Variance source: Random initialization, numerical precision
  
- **PC**: Constraint-based causal discovery
  - Assumes: Faithfulness, conditional independence tests
  - Variance source: Significance level (α ∈ [0.01, 0.10]), test statistics
  
- **FCI**: Fast Causal Inference (proxy for PsiFCI)
  - Assumes: Faithfulness, allows latent confounders
  - Variance source: Significance level, orientation rules

### Metrics

All metrics computed with 95% confidence intervals:

- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1-score** = 2 × (Precision × Recall) / (Precision + Recall)
- **SHD** = Missing edges + Extra edges + Reversed edges

## Comparison with LLM Estimates

### Overlap Metrics

1. **Binary overlap**: Do ranges intersect at all?
2. **Overlap percentage**: What % of the smaller interval overlaps?
3. **Containment**: Does one interval fully contain the other?

### Example Analysis

```python
# Algorithmic result (from 100 runs)
precision_ci = (0.81, 0.88)

# LLM estimates (from paper)
gpt5_range = (0.62, 0.76)
deepseek_range = (0.60, 0.70)

# Analysis
gpt5_overlap = max(0.62, 0.81) <= min(0.76, 0.88)  # False - no overlap
deepseek_overlap = max(0.60, 0.81) <= min(0.70, 0.88)  # False - no overlap

# Conclusion: Both LLMs systematically underestimated precision
```

## Citation

If you use this code, please cite:

```bibtex
@article{causality_variance_2026,
  title={Do LLMs Understand Causal Discovery Algorithms? A Variance-Aware Evaluation},
  author={Anonymous},
  journal={LREC 2026 (under review)},
  year={2026}
}
```

## License

MIT License - See LICENSE file

## Contributing

This is research code. For bugs or improvements, please open an issue.

## Acknowledgments

- causal-learn library for algorithm implementations
- bnlearn for benchmark datasets
- pgmpy for Bayesian network utilities
