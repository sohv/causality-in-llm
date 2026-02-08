# Experiments

## Setup

```bash
pip install -r requirements.txt
```

```bash
export ANTHROPIC_API_KEY="..."
export GOOGLE_API_KEY="..."
export QWEN_API_KEY="..."
export LLAMA_API_KEY="..."
export OPENAI_API_KEY="..."
export DEEPSEEK_API_KEY="..."
```

---

## Execution Order

### 1. Algorithmic Experiments (Ground Truth)

```bash
cd variance
python run_experiments.py --runs 100 --output results_full --experiments all
```

---

### 2. Main Experiment: LLM Predictions (468 queries)

Single optimized prompt per algorithm-dataset combination (6 LLMs × 78 combinations).

```bash
cd llm_integration
python multi_llm_runner.py --all-combos --output results/llm_experiments
```

**Output:** Per-combination results in `results/llm_experiments/`

---

### 3. Supplementary Validation: Prompt Robustness (216 queries)

Subset robustness test (PC, LiNGAM, FCI × titanic, asia, sachs with 4 prompt variations).

```bash
python run_supplementary_validation.py --output results/supplementary_validation
```

**Output:** `results/supplementary_validation/` with robustness analysis

---

### 4. Comparison: LLM vs Algorithm Performance

```bash
python compare_algorithms_vs_llms.py --all-combinations --output results/comparison
```

**Output:** Comparison tables, accuracy metrics, visualizations

---

### 5. Visualizations

```bash
python create_paper_visualizations.py --output results/paper_figs
```

---

## Runtime

- Algorithm experiments: ~3-4 hours
- Main LLM experiment: ~1.5-2 hours  
- Robustness validation: ~30-45 minutes
- Comparison: ~20-30 minutes
- **Total: ~6-7 hours**

---

## Paper Usage

**Main Results:** Stage 2 (single prompt, 78 combinations)  
**Supplementary Materials:** Stage 3 (prompt robustness subset)  
**Comparison Figures:** Stage 4 (LLM accuracy)
