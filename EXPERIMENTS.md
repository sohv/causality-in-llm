# Experiments

## 📊 **Data Sources**

**Real Benchmark Datasets:** All causal benchmark datasets (asia, cancer, earthquake, sachs, survey, child, alarm, insurance, barley) now use **real data from bnlearn** - no simulations. This ensures authentic scientific evaluation.

## 0. Setup

```bash
pip install -r requirements.txt
```

```bash
export ANTHROPIC_API_KEY="..."
export GOOGLE_API_KEY="..."
export QWEN_API_KEY="..."
export QWEN_BASE_URL="https://api.together.xyz/v1"
export QWEN_MODEL="Qwen/Qwen2.5-72B-Instruct"
export LLAMA_API_KEY="..."
export LLAMA_BASE_URL="https://api.together.xyz/v1"
export LLAMA_MODEL="meta-llama/Llama-3.3-70B-Instruct-Turbo"
export OPENAI_API_KEY="..."  # For GPT-5
export DEEPSEEK_API_KEY="..."  # For DeepSeek-R1
export DEEPSEEK_BASE_URL="https://api.deepseek.com"  # Optional override
```

---

## 1. Algorithmic Experiments (6 algorithms x 13 datasets x 100 runs)

### 1a. PC + LiNGAM + FCI + NOTEARS + GES + GRaSP on Titanic + Benchmarks + Synthetic

```bash
cd variance
python run_experiments.py --runs 100 --output results_full --experiments all
```

### 1b. PC + LiNGAM + FCI + NOTEARS + GES + GRaSP on Alarm + Stock Market + Insurance + Barley

```bash
cd variance
python run_experiments.py --runs 100 --output results_full --experiments new_datasets
```

### 1c. FCI dedicated (all 13 datasets)

```bash
cd fci_experiments
python run_fci_experiments.py --runs 100 --output fci_results --experiments all
```

### 1d. NOTEARS dedicated (all 13 datasets)

```bash
cd notears_experiments
python run_notears_experiments.py --runs 100 --output results/
```

### 1e. GES dedicated (all 13 datasets)

```bash
cd ges_grasp_experiments
python run_ges_grasp_experiments.py --runs 100 --output results/ --algorithm ges
```

### 1f. GRaSP dedicated (all 13 datasets)

```bash
cd ges_grasp_experiments
python run_ges_grasp_experiments.py --runs 100 --output results/ --algorithm grasp
```

---

## 2. LLM Experiments (6 LLMs x 3 prompts x 13 datasets x 6 algorithms = 1,404 queries)

**Available LLMs:** 
1. Claude 3.5 Sonnet (Anthropic)
2. Gemini 1.5 Pro (Google) 
3. Qwen 2.5 72B (Alibaba)
4. Llama 3.3 70B (Meta)
5. GPT-5 (OpenAI) - **NEW**
6. DeepSeek-R1 (DeepSeek) - **NEW**

Each command below queries all available LLMs with all 3 prompt formulations (Direct, Step-by-Step, Meta-Knowledge) automatically.

### 2a. All combos at once

```bash
cd llm_integration
python multi_llm_runner.py --all-combos --output results/llm_experiments
```

### 2b. Or one dataset-algorithm pair at a time

```bash
cd llm_integration

# Titanic
python multi_llm_runner.py --dataset titanic --algorithm PC --output results/llm_experiments
python multi_llm_runner.py --dataset titanic --algorithm LiNGAM --output results/llm_experiments
python multi_llm_runner.py --dataset titanic --algorithm FCI --output results/llm_experiments
python multi_llm_runner.py --dataset titanic --algorithm NOTEARS --output results/llm_experiments
python multi_llm_runner.py --dataset titanic --algorithm GES --output results/llm_experiments
python multi_llm_runner.py --dataset titanic --algorithm GRaSP --output results/llm_experiments

# Sachs
python multi_llm_runner.py --dataset sachs --algorithm PC --output results/llm_experiments
python multi_llm_runner.py --dataset sachs --algorithm LiNGAM --output results/llm_experiments
python multi_llm_runner.py --dataset sachs --algorithm FCI --output results/llm_experiments
python multi_llm_runner.py --dataset sachs --algorithm NOTEARS --output results/llm_experiments
python multi_llm_runner.py --dataset sachs --algorithm GES --output results/llm_experiments
python multi_llm_runner.py --dataset sachs --algorithm GRaSP --output results/llm_experiments

# Alarm
python multi_llm_runner.py --dataset alarm --algorithm PC --output results/llm_experiments
python multi_llm_runner.py --dataset alarm --algorithm LiNGAM --output results/llm_experiments
python multi_llm_runner.py --dataset alarm --algorithm FCI --output results/llm_experiments
python multi_llm_runner.py --dataset alarm --algorithm NOTEARS --output results/llm_experiments
python multi_llm_runner.py --dataset alarm --algorithm GES --output results/llm_experiments
python multi_llm_runner.py --dataset alarm --algorithm GRaSP --output results/llm_experiments

# Stock Market
python multi_llm_runner.py --dataset stock_market --algorithm PC --output results/llm_experiments
python multi_llm_runner.py --dataset stock_market --algorithm LiNGAM --output results/llm_experiments
python multi_llm_runner.py --dataset stock_market --algorithm FCI --output results/llm_experiments
python multi_llm_runner.py --dataset stock_market --algorithm NOTEARS --output results/llm_experiments
python multi_llm_runner.py --dataset stock_market --algorithm GES --output results/llm_experiments
python multi_llm_runner.py --dataset stock_market --algorithm GRaSP --output results/llm_experiments

# Insurance
python multi_llm_runner.py --dataset insurance --algorithm PC --output results/llm_experiments
python multi_llm_runner.py --dataset insurance --algorithm LiNGAM --output results/llm_experiments
python multi_llm_runner.py --dataset insurance --algorithm FCI --output results/llm_experiments
python multi_llm_runner.py --dataset insurance --algorithm NOTEARS --output results/llm_experiments
python multi_llm_runner.py --dataset insurance --algorithm GES --output results/llm_experiments
python multi_llm_runner.py --dataset insurance --algorithm GRaSP --output results/llm_experiments

# Barley
python multi_llm_runner.py --dataset barley --algorithm PC --output results/llm_experiments
python multi_llm_runner.py --dataset barley --algorithm LiNGAM --output results/llm_experiments
python multi_llm_runner.py --dataset barley --algorithm FCI --output results/llm_experiments
python multi_llm_runner.py --dataset barley --algorithm NOTEARS --output results/llm_experiments
python multi_llm_runner.py --dataset barley --algorithm GES --output results/llm_experiments
python multi_llm_runner.py --dataset barley --algorithm GRaSP --output results/llm_experiments

# Asia
python multi_llm_runner.py --dataset asia --algorithm PC --output results/llm_experiments
python multi_llm_runner.py --dataset asia --algorithm LiNGAM --output results/llm_experiments
python multi_llm_runner.py --dataset asia --algorithm FCI --output results/llm_experiments
python multi_llm_runner.py --dataset asia --algorithm NOTEARS --output results/llm_experiments
python multi_llm_runner.py --dataset asia --algorithm GES --output results/llm_experiments
python multi_llm_runner.py --dataset asia --algorithm GRaSP --output results/llm_experiments

# Cancer
python multi_llm_runner.py --dataset cancer --algorithm PC --output results/llm_experiments
python multi_llm_runner.py --dataset cancer --algorithm LiNGAM --output results/llm_experiments
python multi_llm_runner.py --dataset cancer --algorithm FCI --output results/llm_experiments
python multi_llm_runner.py --dataset cancer --algorithm NOTEARS --output results/llm_experiments
python multi_llm_runner.py --dataset cancer --algorithm GES --output results/llm_experiments
python multi_llm_runner.py --dataset cancer --algorithm GRaSP --output results/llm_experiments

# Earthquake
python multi_llm_runner.py --dataset earthquake --algorithm PC --output results/llm_experiments
python multi_llm_runner.py --dataset earthquake --algorithm LiNGAM --output results/llm_experiments
python multi_llm_runner.py --dataset earthquake --algorithm FCI --output results/llm_experiments
python multi_llm_runner.py --dataset earthquake --algorithm NOTEARS --output results/llm_experiments
python multi_llm_runner.py --dataset earthquake --algorithm GES --output results/llm_experiments
python multi_llm_runner.py --dataset earthquake --algorithm GRaSP --output results/llm_experiments

# Survey
python multi_llm_runner.py --dataset survey --algorithm PC --output results/llm_experiments
python multi_llm_runner.py --dataset survey --algorithm LiNGAM --output results/llm_experiments
python multi_llm_runner.py --dataset survey --algorithm FCI --output results/llm_experiments
python multi_llm_runner.py --dataset survey --algorithm NOTEARS --output results/llm_experiments
python multi_llm_runner.py --dataset survey --algorithm GES --output results/llm_experiments
python multi_llm_runner.py --dataset survey --algorithm GRaSP --output results/llm_experiments

# Child
python multi_llm_runner.py --dataset child --algorithm PC --output results/llm_experiments
python multi_llm_runner.py --dataset child --algorithm LiNGAM --output results/llm_experiments
python multi_llm_runner.py --dataset child --algorithm FCI --output results/llm_experiments
python multi_llm_runner.py --dataset child --algorithm NOTEARS --output results/llm_experiments
python multi_llm_runner.py --dataset child --algorithm GES --output results/llm_experiments
python multi_llm_runner.py --dataset child --algorithm GRaSP --output results/llm_experiments

# Synthetic 12
python multi_llm_runner.py --dataset synthetic_12 --algorithm PC --output results/llm_experiments
python multi_llm_runner.py --dataset synthetic_12 --algorithm LiNGAM --output results/llm_experiments
python multi_llm_runner.py --dataset synthetic_12 --algorithm FCI --output results/llm_experiments
python multi_llm_runner.py --dataset synthetic_12 --algorithm NOTEARS --output results/llm_experiments
python multi_llm_runner.py --dataset synthetic_12 --algorithm GES --output results/llm_experiments
python multi_llm_runner.py --dataset synthetic_12 --algorithm GRaSP --output results/llm_experiments

# Synthetic 30
python multi_llm_runner.py --dataset synthetic_30 --algorithm PC --output results/llm_experiments
python multi_llm_runner.py --dataset synthetic_30 --algorithm LiNGAM --output results/llm_experiments
python multi_llm_runner.py --dataset synthetic_30 --algorithm FCI --output results/llm_experiments
python multi_llm_runner.py --dataset synthetic_30 --algorithm NOTEARS --output results/llm_experiments
python multi_llm_runner.py --dataset synthetic_30 --algorithm GES --output results/llm_experiments
python multi_llm_runner.py --dataset synthetic_30 --algorithm GRaSP --output results/llm_experiments
```

---

## 3. Algorithm vs LLM Comparison Experiments (NEW)

### 3a. Compare specific dataset + algorithm combination

```bash
python compare_algorithms_vs_llms.py --dataset titanic --algorithm PC
python compare_algorithms_vs_llms.py --dataset sachs --algorithm LiNGAM
python compare_algorithms_vs_llms.py --dataset alarm --algorithm FCI
```

### 3b. Run full comparison across all combinations

```bash
python compare_algorithms_vs_llms.py --all-combinations
```

### 3c. Generate comparison visualizations only

```bash
python compare_algorithms_vs_llms.py --viz-only
```

**What this does:**
- Compares algorithm ground truth results vs LLM predictions
- Calculates prediction accuracy for F1, Precision, Recall metrics  
- Generates visualizations showing which LLMs predict algorithm performance best
- Creates detailed reports with accuracy statistics
- Saves results to `results/comparison/`

---

## 4. Visualizations

```bash
python create_paper_visualizations.py
```
