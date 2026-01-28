# FCI Experiments - Implementation Summary

## Changes Made

### Extended FCI to ALL 9 Datasets

Previously planned for only 5 datasets, now matches PC/LiNGAM exactly:

**Benchmarks (6)**:
- asia
- cancer  
- earthquake
- sachs
- survey
- child

**Synthetic (2)**:
- synthetic_12
- synthetic_30

**Real-world (1)**:
- titanic

**TOTAL**: 9 datasets (same as PC/LiNGAM)

## Key Implementation Details

1. **Same methodology** as PC/LiNGAM:
   - Bootstrap sampling (random sampling with replacement)
   - Varying alpha (0.01-0.10)
   - 100 runs per dataset
   - 95% confidence intervals
   - Same metrics: Precision, Recall, F1, SHD

2. **Fixed FCI implementation** in variance_analysis.py:
   - Added missing bootstrap sampling
   - Now matches PC/LiNGAM variance analysis approach

3. **Structured like existing experiments**:
   - `run_fci_titanic()` - single dataset
   - `run_fci_benchmark_experiments()` - all 6 benchmarks
   - `run_fci_synthetic_experiments()` - both synthetic datasets

## Usage

### Run ALL experiments (recommended):
```bash
python run_fci_experiments.py --runs 100
```

### Run specific experiment groups:
```bash
# Just benchmarks (6 datasets)
python run_fci_experiments.py --experiments benchmarks

# Just synthetic (2 datasets)
python run_fci_experiments.py --experiments synthetic

# Just titanic
python run_fci_experiments.py --experiments titanic
```

## Expected Results

### Direct Comparisons with PC/LiNGAM

Since all three algorithms now use the same 9 datasets, you can directly compare:

1. **Sachs** (latent confounders):
   - FCI should WIN (designed for latent confounders)
   - PC/LiNGAM should FAIL (assume causal sufficiency)

2. **Synthetic 30** (scaling):
   - FCI should COLLAPSE (F1 < 0.2)
   - PC should be OK (F1 ~ 0.5)
   - LiNGAM should be BEST (F1 ~ 0.6)

3. **Other benchmarks**:
   - PC should generally win (no latent confounders)
   - FCI should have more overhead without benefit
   - LiNGAM may fail on discrete data

## Files Structure

```
fci_experiments/
├── run_fci_experiments.py        # Main runner (9 datasets)
├── visualize_fci_results.py      # Visualization
├── test_fci_setup.py              # Setup verification
├── README.md                      # Documentation
├── QUICKSTART.md                  # Quick commands
├── FCI_VS_OTHERS.md              # Algorithm comparison
└── SUMMARY.md                     # This file

fci_results/ (after running)
├── asia_fci_variance.json
├── cancer_fci_variance.json
├── earthquake_fci_variance.json
├── sachs_fci_variance.json
├── survey_fci_variance.json
├── child_fci_variance.json
├── synthetic_12_fci_variance.json
├── synthetic_30_fci_variance.json
└── titanic_fci_variance.json

9 datasets × 2 files each = 18 files total
```

## Advantages of This Approach

1. **Fair comparison**: All algorithms tested on same datasets
2. **Comprehensive**: 9 datasets vs original 5
3. **Consistent methodology**: Same bootstrap/variance approach
4. **Direct comparison**: Can plot FCI vs PC vs LiNGAM side-by-side
5. **Better science**: No cherry-picking datasets per algorithm

## Runtime Estimate

- **Per dataset**: 15-60 minutes (depending on size)
- **Total (all 9)**: 3-5 hours with 100 runs each

Largest datasets (Child @ 20 nodes, Synth30 @ 30 nodes) will take longest.

## Next Steps

1. Run experiments: `python run_fci_experiments.py --runs 100`
2. Compare with PC/LiNGAM results in `../variance/results/`
3. Generate comparison plots showing all three algorithms
4. Analyze where FCI wins (Sachs) and loses (Synth30)
