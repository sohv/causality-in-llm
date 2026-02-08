# Quick Start Guide - UAI 2026 Experiments

## TL;DR - Run Everything

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install anthropic google-generativeai notears pgmpy

# 2. Set API keys (for LLM experiments)
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"

# 3. Run all algorithmic experiments (~10 hours)
cd variance/
python run_experiments.py --runs 100 --experiments all

cd ../notears_experiments/
python run_notears_experiments.py --runs 100

# 4. Run LLM experiments (~2-4 hours)
cd ../llm_integration/
python multi_llm_runner.py --dataset titanic --algorithm PC

# 5. Generate paper visualizations
cd ..
python create_paper_visualizations.py
```

---

## What Gets Run

### Algorithmic Experiments (Step 3)

**Datasets (11):**
1. Titanic (7 nodes, social)
2. Sachs (11 nodes, biology)
3. **Alarm (37 nodes, medical) ← NEW**
4. **Stock Market (10 nodes, finance) ← NEW**
5-9. ASIA, CANCER, EARTHQUAKE, SURVEY, CHILD (benchmarks)
10-11. Synthetic-12, Synthetic-30

**Algorithms (4):**
1. PC (constraint-based)
2. LiNGAM (order-based)
3. FCI (latent confounders)
4. **NOTEARS (continuous optimization) ← NEW**

**Runs:** 100 per dataset-algorithm pair
**Total experiments:** 11 datasets × 4 algorithms × 100 runs = 4,400 runs

### LLM Experiments (Step 4)

**LLMs (4):**
1. GPT-5 (OpenAI)
2. DeepSeek R1
3. **Claude 3.5 Sonnet (Anthropic) ← NEW**
4. **Gemini 1.5 Pro (Google) ← NEW**

**Prompt Formulations (3):** ← NEW
1. Direct question
2. Step-by-step reasoning
3. Meta-knowledge framing

**Queries:** 11 datasets × 4 algorithms × 3 prompts × 4 LLMs = **528 API calls**

---

## Testing Individual Components

### Test New Datasets

```bash
# Test Alarm Network
cd datasets/
python alarm_network.py

# Test Stock Market
python stock_market.py
```

### Test NOTEARS Integration

```bash
cd variance/
python -c "
from variance_analysis import VarianceAnalyzer
import numpy as np
import pandas as pd

# Create simple test data
data = pd.DataFrame(np.random.randn(100, 5))
true_graph = np.array([[0,1,0,0,0],
                       [0,0,1,0,0],
                       [0,0,0,1,0],
                       [0,0,0,0,1],
                       [0,0,0,0,0]])

analyzer = VarianceAnalyzer(n_runs=10)
results = analyzer.run_notears_multiple(data, true_graph)
print('NOTEARS works!', results.precision.mean)
"
```

### Test LLM Clients

```bash
# Test Claude
cd llm_integration/
python claude_api.py

# Test Gemini
python gemini_api.py
```

### Test Prompt Variations

```bash
cd prompt_variations/
python prompt_templates.py
```

---

## Expected Output

### Results Directory Structure

```
results/
├── variance/
│   ├── titanic_pc_variance.json
│   ├── titanic_lingam_variance.json
│   ├── titanic_fci_variance.json
│   ├── titanic_notears_variance.json      # NEW
│   ├── alarm_pc_variance.json             # NEW
│   ├── alarm_notears_variance.json        # NEW
│   ├── stock_market_pc_variance.json      # NEW
│   └── ... (44 files total)
├── notears/
│   └── ... (11 files)
└── llm_experiments/
    ├── titanic_PC_llm_results.json
    ├── plots/
    │   ├── prompt_robustness_heatmap.png
    │   ├── prompt_percent_difference.png
    │   └── cross_llm_comparison.png
    └── ...
```

### Visualizations

```
paper_plots/
├── fig1_algorithm_comparison.png    # Algorithm heatmaps
├── fig2_complexity_analysis.png     # Scalability plots
├── fig3_coverage_summary.png        # Coverage bars
└── main_results_table.tex           # LaTeX table
```

---

## Troubleshooting

### NOTEARS installation fails

```bash
# If notears package not found:
pip install git+https://github.com/xunzheng/notears.git
```

### LLM API errors

```bash
# Check API keys are set:
echo $ANTHROPIC_API_KEY
echo $GOOGLE_API_KEY

# Check rate limits:
# Claude: 50 requests/min
# Gemini: 60 requests/min

# If rate limited, add delays in multi_llm_runner.py:
import time
time.sleep(2)  # Between requests
```

### Memory issues on large datasets (Alarm 37 nodes)

```bash
# Reduce number of runs:
python run_experiments.py --runs 50  # Instead of 100

# Or run datasets individually:
python run_experiments.py --experiments titanic
python run_experiments.py --experiments new_datasets
```

---

## Time Estimates

| Task | Time | Notes |
|------|------|-------|
| Setup | 10 min | Install packages, set API keys |
| Algorithmic experiments | 8-12 hours | 4,400 total runs |
| LLM experiments | 2-4 hours | 528 API calls, rate limited |
| Visualizations | 2 min | Fast, just reads JSON files |
| **Total** | **10-16 hours** | Run overnight |

---

## Quick Validation

After running everything, check:

```bash
# Count result files
find . -name "*_variance.json" | wc -l
# Should be: ~55 files (44 from variance + 11 from notears)

# Check LLM results
ls llm_integration/results/
# Should have JSON files and plots/

# Check visualizations
ls paper_plots/
# Should have 4+ PNG files and 1 TEX file

# Quick sanity check
python -c "
import json
with open('variance/results/titanic_pc_variance.json') as f:
    data = json.load(f)
    print(f\"Titanic+PC Precision: {data['results']['precision']['mean']:.3f}\")
"
```

---

## Next Steps

1. ✓ Run experiments (this guide)
2. ✓ Generate visualizations
3. Update paper:
   - Add new results to tables
   - Update figures
   - Add prompt robustness section
   - Mention 4 LLMs, 11 datasets, 4 algorithms
4. Submit to UAI 2026!

**Estimated acceptance probability:** 75-80% (per PDF analysis)

Good luck! 🚀
