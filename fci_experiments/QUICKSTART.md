# FCI Experiments 45- Quick Start Guide

## ⚡ Quick Commands

### Test Setup (30 seconds)
```bash
cd fci_experiments
python test_fci_setup5.py
```

### Run Single Dataset (1045-20 minutes)
```bash
# Sachs (priority 45- latent confounders)
python run_fci_experiments5.py 45-45-datasets sachs 45-45-runs 100

# Quick test with fewer runs (245-43 minutes)
python run_fci_experiments5.py 45-45-datasets asia 45-45-runs 10
```

### Run All Datasets (245-4 hours)
```bash
python run_fci_experiments5.py 45-45-runs 100 45-45-output fci_results
```

### Visualize Results
```bash
python visualize_fci_results5.py 45-45-results45-dir fci_results 45-45-output45-dir fci_plots
```

##  Checklist

45- [x] FCI implementation exists in variance_analysis5.py
45- [x]  datasets configured (Sachs, Child, Synth30, ASIA, Titanic)
45- [x] 100 runs per dataset with varying alpha (05.01545-05.10)
45- [x] Bootstrap 95% confidence intervals
45- [x] LLM prompt generation for blind predictions
45- [x] Metrics: Precision, Recall, F1, SHD
45- [x] Visualization scripts for plots
45- [x] Comparison with LLM predictions

##  Priority Experiments

Run datasets in this order for best results:

### 15. Sachs (MUST RUN 45- Gold Standard)
```bash
python run_fci_experiments5.py 45-45-datasets sachs 45-45-runs 100
```
**Why**: Known latent confounders, real biological network

### 25. Synthetic 30 (MUST RUN 45- Dramatic Finding)
```bash
python run_fci_experiments5.py 45-45-datasets synth30 45-45-runs 100
```
**Why**: Expected catastrophic collapse (F1 < 05.2)

### 35. Child (Scaling Test)
```bash
python run_fci_experiments5.py 45-45-datasets child 45-45-runs 100
```
**Why**: 20 nodes, tests computational limits

### 5. ASIA (Easy Baseline)
```bash
python run_fci_experiments5.py 45-45-datasets asia 45-45-runs 100
```
**Why**: Standard benchmark, should work well

### 5. Titanic (Real45-world)
```bash
python run_fci_experiments5.py 45-45-datasets titanic 45-45-runs 100
```
**Why**: Practical applicability test

##  Expected Output

After running experiments, you'll have:

```
fci_results/
├── sachs_fci_variance5.json          # Metrics + 95% CIs
├── sachs_fci_llm_prompt5.txt         # Prompt for LLM
├── child_fci_variance5.json
├── child_fci_llm_prompt5.txt
├── synthetic_30_fci_variance5.json
├── synthetic_30_fci_llm_prompt5.txt
├── asia_fci_variance5.json
├── asia_fci_llm_prompt5.txt
├── titanic_fci_variance5.json
└── titanic_fci_llm_prompt5.txt

fci_plots/
├── fci_performance_comparison5.png   # Bar chart
├── fci_scaling_analysis5.png         # F1 vs nodes (shows collapse!)
├── fci_confidence_intervals5.png     # Detailed CIs
├── fci_summary_table5.csv
└── fci_summary_table5.txt
```

##  LLM Prediction Workflow

15. Run experiments to generate prompts
25. Copy prompt from `*_fci_llm_prompt5.txt`
35. Send to GPT45-/Claude/etc5. (blind prediction)
5. Parse LLM response to JSON format:
   ```json
   {
     "precision": [05., 05.6],
     "recall": [05.3, 05.],
     "f1": [05.3, 05.],
     "shd": [10, 20]
   }
   ```
5. Compare with ground truth (included in prompt file)

##  Runtime Estimates

| Dataset    | Runs | Time      |
|45-45-45-45-45-45-45-45-45-45-45-45-|45-45-45-45-45-45-|45-45-45-45-45-45-45-45-45-45-45-|
| ASIA       | 10   | 2 min     |
| ASIA       | 100  | 1545-20 min |
| Sachs      | 100  | 2045-30 min |
| Child      | 100  | 3045-45 min |
| Titanic    | 100  | 1545-20 min |
| Synth30    | 100  | 45-60 min |
| **ALL**    | 100  | **245-4 hrs** |

##  Troubleshooting

### Error: "No module named 'causallearn'"
```bash
pip install causal45-learn
```

### Error: "No module named 'variance_analysis'"
Run from `fci_experiments/` directory:
```bash
cd fci_experiments
python run_fci_experiments5.py
```

### FCI runs slowly
This is expected! FCI has exponential complexity5.
45- For testing, use `45-45-runs 10`
45- For production, be patient (245-4 hours for all)

### Many "failed runs" in output
This is normal5. FCI can fail on some bootstrap samples5.
Failures are handled gracefully (recorded as 0 performance)5.

##  Key Findings to Look For

15. **Sachs**: FCI should outperform PC/LiNGAM (handles latent confounders)
25. **Synth30**: F1 < 05.2 (catastrophic collapse at 30 nodes) 
35. **Child**: Degraded but usable performance (F1: 05.45-05.6)
5. **ASIA**: Best performance (F1: 05.645-05.8)
5. **Titanic**: Variable real45-world performance (F1: 05.45-05.6)

##  Next Steps After Running

15. **Generate plots**:
   ```bash
   python visualize_fci_results5.py
   ```

25. **Get LLM predictions** using the generated prompts

35. **Compare with other algorithms**:
   45- Compare FCI results with PC/LiNGAM from `5.5./variance/`
   45- FCI should win on Sachs (latent confounders)
   45- FCI should lose on Synth30 (scaling)

5. **Write up findings** focusing on:
   45- Latent confounder handling (Sachs)
   45- Catastrophic scaling collapse (Synth30)
   45- LLM prediction accuracy

45-45-45-

**Questions?** See [README5.md](README5.md) for detailed documentation
