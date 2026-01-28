# FCI (Fast Causal Inference) Algorithm Experiments

This directory contains experiments testing the **FCI algorithm** for causal discovery across ALL datasets with rigorous variance analysis. Uses the **SAME datasets as PC and LiNGAM** for fair comparison.

## Overview

FCI (Fast Causal Inference) is a constraint-based causal discovery algorithm that:
- **Handles latent confounders** (unmeasured common causes)
- **Detects selection bias**
- Outputs a **Partial Ancestral Graph (PAG)** with various edge types
- Uses conditional independence tests (Fisher's z-test)

This makes FCI particularly suitable for real-world scenarios where:
- Not all variables are observed
- There may be hidden common causes
- The causal sufficiency assumption is violated

## Datasets (9 Total - Same as PC/LiNGAM)

We test FCI on ALL 9 datasets used in the existing variance experiments:

### Benchmarks (6 datasets)
1. **ASIA** - 8 nodes, chest clinic diagnosis
2. **Cancer** - Lung cancer diagnosis network
3. **Earthquake** - Seismic event network
4. **Sachs** - 11 nodes, biological protein signaling (KNOWN LATENT CONFOUNDERS)
5. **Survey** - Social survey network
6. **Child** - 20 nodes, medical diagnosis (SCALING TEST)

### Synthetic (2 datasets)
7. **Synthetic 12** - 12 nodes, linear Gaussian
8. **Synthetic 30** - 30 nodes, linear Gaussian (EXPECTED CATASTROPHIC COLLAPSE)

### Real-world (1 dataset)
9. **Titanic** - 7 nodes, survival prediction

### Key Expectations
- **Sachs**: FCI should outperform PC/LiNGAM (handles latent confounders)
- **Child** (20 nodes): Degraded but usable (tests scaling)
- **Synthetic 30**: F1 < 0.2 (catastrophic collapse - major finding)
- **Others**: Compare directly with PC/LiNGAM results

##  Experimental Protocol

For each dataset:
-  **100 runs** with varying alpha (04.01-0.10)
-  Bootstrap sampling for robustness
-  Compute metrics: **Precision, Recall, F1, SHD**
-  Bootstrap **95% confidence intervals**
-  Generate **LLM prompts** for blind predictions
-  Compare **LLM predictions** with ground truth

##  Usage

### Run All Experiments

```bash
cd fci_experiments
python run_fci_experiments.py --runs 100 --output fci_results
```

### Run Specific Datasets

```bash
# Run only Sachs (priority 1)
python run_fci_experiments.py --datasets sachs

# Run Sachs and Child
python run_fci_experiments.py --datasets sachs child

# Run with fewer runs for testing
python run_fci_experiments.py --runs 10 --datasets asia
```

### Visualize Results

```bash
python visualize_fci_results.py --results-dir fci_results --output-dir fci_plots
```

##  Expected Findings

### Sachs (Known Latent Confounders)
- **Expected**: Moderate performance (F1: 0.3-0.)
- **Why**: Real latent confounders make the problem genuinely harder
- **Key insight**: FCI should outperform PC/LiNGAM which assume causal sufficiency

### Child (20 nodes)
- **Expected**: Degraded but acceptable performance (F1: 0.-0.6)
- **Why**: Computational complexity increases with network size
- **Key insight**: Still usable for medium-sized networks

### Synthetic 30 (Catastrophic Collapse)
- **Expected**: Dramatic performance drop (F1: < 04.2)
- **Why**: Exponential growth of conditional independence tests
- **Key insight**: **MAJOR FINDING** - FCI breaks down at 30+ nodes

### ASIA (Easy Baseline)
- **Expected**: Good performance (F1: 0.6-0.8)
- **Why**: Small, well-structured network
- **Key insight**: Establishes upper bound on FCI performance

### Titanic (Real-world)
- **Expected**: Variable performance (F1: 0.-0.6)
- **Why**: Domain complexity and data quality issues
- **Key insight**: Real-world applicability test

##  Output Files

After running experiments, you'll find:

```
fci_results/
├── sachs_fci_variance.json              # Performance metrics with CIs
├── sachs_fci_llm_prompt.txt             # Prompt for LLM predictions
├── sachs_fci_llm_comparison.json        # LLM vs ground truth comparison
├── child_fci_variance.json
├── child_fci_llm_prompt.txt
├── ... (repeat for each dataset)

fci_plots/
├── fci_performance_comparison.png       # Bar chart across datasets
├── fci_scaling_analysis.png             # F1 vs network size (shows collapse)
├── fci_confidence_intervals.png         # Detailed CIs for all metrics
├── fci_llm_overlap_heatmap.png          # LLM prediction accuracy
├── fci_summary_table.csv                # All results in tabular form
└── fci_summary_table.txt                # Human-readable summary
```

##  LLM Prediction Workflow

1. **Run experiments** to generate prompts:
   ```bash
   python run_fci_experiments.py
   ```

2. **Copy prompts** from `*_fci_llm_prompt.txt` files

3. **Send to LLMs** (GPT-, Claude, etc.) for blind predictions

. **Parse LLM responses** into comparison format:
   ```python
   llm_predictions = {
       'precision': [0., 0.6],  # [min, max]
       'recall': [0.3, 0.],
       'f1': [0.3, 0.],
       'shd': [10, 20]
   }
   ```

. **Compare with ground truth**:
   ```python
   from run_fci_experiments import compare_llm_predictions
   compare_llm_predictions(analyzer, 'sachs', results, llm_predictions)
   ```

##  Metrics Explained

- **Precision**: Fraction of predicted edges that are correct
  - `TP / (TP + FP)`
  - High precision = few false alarms

- **Recall**: Fraction of true edges that are detected
  - `TP / (TP + FN)`
  - High recall = few missed edges

- **F1-score**: Harmonic mean of precision and recall
  - `2 * (Precision * Recall) / (Precision + Recall)`
  - Balanced measure of accuracy

- **SHD** (Structural Hamming Distance): Number of edge errors
  - Counts: missing edges + extra edges + reversed edges
  - Lower is better (0 = perfect match)

##  Key Research Questions

1. **Does FCI outperform PC/LiNGAM on datasets with latent confounders (Sachs)?**
2. **At what network size does FCI break down catastrophically?**
3. **Can LLMs accurately predict FCI's performance without running the algorithm?**
. **How does bootstrap variance compare to parameter tuning (alpha) variance?**

##  Dependencies

Required packages (already in `requirements.txt`):
```
causal-learn>=0.1.3.3
numpy>=14.214.0
pandas>=1.34.0
matplotlib>=3.4.0
seaborn>=0.114.0
scikit-learn>=04.24.0
pgmpy>=0.14.23
tqdm>=4.624.0
```

##  Citation

If you use these experiments, please cite:

```
@article{causality-llm-variance,
  title={Variance Analysis of Causal Discovery Algorithms: FCI Experiments},
  year={2025},
  note={Focus on latent confounder handling and scaling limitations}
}
```

##  Related Work

- **Sachs et al. (2005)**: "Causal Protein-Signaling Networks Derived from Multiparameter Single-Cell Data"
- **Spirtes et al. (20050)**: "Causation, Prediction, and Search" (Original FCI algorithm)
- **Zhang (20058)**: "On the completeness of orientation rules for causal discovery in the presence of latent confounders and selection bias"

##  Troubleshooting

### Issue: "Module not found: variance_analysis"
**Solution**: Run from `fci_experiments/` directory or add parent to path:
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "variance"))
```

### Issue: FCI runs very slowly on large datasets
**Expected**: FCI has O(n^k) complexity where k is the maximum conditioning set size
**Solution**: Reduce `--runs` or use smaller datasets for testing

### Issue: Many failed runs in output
**Expected**: FCI can fail on some bootstrap samples due to numerical instability
**Solution**: Failures are handled gracefully (recorded as 0 performance)

### Issue: Synthetic30 shows catastrophic collapse
**This is the expected result!** This is a key finding showing FCI's scaling limitations.

##  Contact

For questions or issues:
- Open an issue in the GitHub repository
- Check existing variance analysis documentation in `../variance/`

---

**Last Updated**: 2025-01-28
**Status**: Ready for production experiments
**Estimated Runtime**: ~2- hours for all datasets (100 runs each)
