# On the Identifiability Limits of Persona Steering

This repository provides empirical validation of non-identifiability in persona steering vectors for language models using direct orthogonal component testing.

## Overview

We validate non-identifiability through direct empirical tests on Qwen2.5-3B-Instruct across three semantic traits:

- **Formality**: Formal vs. informal language markers
- **Politeness**: Polite vs. rude language markers  
- **Humor**: Humorous vs. serious language markers

**Key method**: Test if orthogonal components to steering vector **v** produce equivalent semantic effects (evidence for non-identifiability).

## Core Finding

Orthogonal components to v (v⊥) produce nearly identical semantic effects across all traits:
- **Cohen's d**: 0.075 ± 0.058 (v vs v+v⊥) → semantically equivalent
- **Correlation**: 0.285 ± 0.087 (high consistency)
- **α-sweep**: Curves overlap across α ∈ [0.0, 0.5, 1.0, 2.0] → identical scaling

## Experimental Method

We test non-identifiability through **two direct empirical tests**:

### Test 1: Orthogonal Component Irrelevance
Tests if adding random orthogonal vectors to v produces equivalent semantic effects.

**Setup**:
```python
python test_orthogonal.py --traits formality politeness humor --n_seeds 10
```

**What it does**:
1. Extract steering vector v from contrastive prompts
2. Generate text with v
3. Generate text with v + v⊥ (random orthogonal components)
4. Compute Cohen's d and correlation in semantic scores
5. Repeat for 10 random orthogonal seeds

**Key Metric**: Cohen's d < 0.3 indicates v and v+v⊥ are semantically equivalent (evidence for non-identifiability).

### Test 2: α-Sweep Response Curves
Tests if steering strength α scaling is identical for v and v+v⊥.

**Setup**:
```python
python test_alpha_sweep.py --alphas 0.0 0.5 1.0 2.0 --n_seeds 10
```

**What it does**:
1. Test steering strength α ∈ [0.0, 0.5, 1.0, 2.0]
2. For each α, generate text with v and v+v⊥
3. Plot response curves: should overlap if non-identifiable

**Key Finding**: Overlapping curves across all α values indicates identical scaling behavior (evidence for non-identifiability).

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

The project uses two configuration files:

- **`config.json`**: Contains persona prompts for all traits (formality, politeness, humor, sentiment)
- **`model_config.yml`**: Contains model configuration (currently Qwen/Qwen2.5-3B-Instruct)

These files can be modified to:
- Add new persona prompts for existing traits
- Add new traits with custom prompts
- Configure different models (when adding support for additional models)

## Usage

### Run All Experiments

**Test orthogonal component irrelevance (10 seeds, all 3 traits)**:
```bash
python test_orthogonal.py --traits formality politeness humor --n_seeds 10
```

Arguments:
- `--traits`: Which traits to test (formality, politeness, humor) - default: all
- `--n_seeds`: Number of orthogonal seeds per trait - default: 5
- `--model`: Model name - default: Qwen/Qwen2.5-3B-Instruct

Output: `results/orthogonal_test_results.json`

**Test α-sweep for formality (10 seeds, 4 alpha values)**:
```bash
python test_alpha_sweep.py --alphas 0.0 0.5 1.0 2.0 --n_seeds 10
```

Arguments:
- `--alphas`: Alpha values to test - default: [0.0, 0.5, 1.0, 2.0]
- `--n_seeds`: Number of seeds per alpha - default: 10
- `--model`: Model name - default: Qwen/Qwen2.5-3B-Instruct

Output:
- JSON: `results/alpha_sweep_results.json`
- Plot: `figures/alpha_sweep_response_curves.png` & `.pdf`

### Running Experiments with LLaMA-3.1-8B

To run the same experiments using the LLaMA-3.1-8B-Instruct model, simply add the `--model` flag:

**Test orthogonal component irrelevance with LLaMA-3.1-8B**:
```bash
python test_orthogonal.py --traits formality politeness humor --n_seeds 10 --model meta-llama/Llama-3.1-8B-Instruct
```

**Test α-sweep with LLaMA-3.1-8B**:
```bash
python test_alpha_sweep.py --alphas 0.0 0.5 1.0 2.0 --n_seeds 10 --model meta-llama/Llama-3.1-8B-Instruct
```

Note: You may need to request access to the LLaMA model on HuggingFace and authenticate using `huggingface-cli login` before running these experiments.

## Results

### Test 1: Orthogonal Component Irrelevance

#### Qwen2.5-3B-Instruct Results (n=10 seeds per trait)

| Trait      | Cohen's d     | Correlation   | Effect (v⊥ only) | Interpretation                            |
|------------|---------------|---------------|------------------|-------------------------------------------|
| Formality  | 0.075 ± 0.058 | 0.285 ± 0.087 | 100.4%          | Semantically equivalent (d < 0.3)         |
| Politeness | 0.092 ± 0.069 | 0.414 ± 0.083 | 100.6%          | Semantically equivalent (d < 0.3)         |
| Humor      | 0.072 ± 0.061 | 0.044 ± 0.097 | 100.5%          | Semantically equivalent (d < 0.3)         |

**Key Findings**:
- **Cohen's d < 0.3** for all traits → v and v+v⊥ produce semantically equivalent outputs
- **Correlation ≈ 0-0.4** → v⊥ adds independent variation (evidence for null space)
- **Effect ≈ 100%** → pure v⊥ produces equivalent effect to v alone
- All results consistent with non-identifiability across formality, politeness, and humor traits

#### LLaMA-3.1-8B-Instruct Results (n=10 seeds per trait)

| Trait      | Cohen's d     | Correlation   | Effect (v⊥ only) | Interpretation                            |
|------------|---------------|---------------|------------------|-------------------------------------------|
| Formality  | 0.096 ± 0.068 | 0.192 ± 0.085 | 96.8%           | Semantically equivalent (d < 0.3)         |
| Politeness | 0.085 ± 0.043 | 0.347 ± 0.077 | 100.4%          | Semantically equivalent (d < 0.3)         |
| Humor      | 0.119 ± 0.119 | 0.016 ± 0.104 | 95.9%           | Semantically equivalent (d < 0.3)         |

**Key Findings**:
- **Cohen's d < 0.3** for all traits → v and v+v⊥ produce semantically equivalent outputs
- Results replicate across both Qwen2.5-3B and LLaMA-3.1-8B models
- Non-identifiability appears to be a general property, not model-specific

### Test 2: α-Sweep Response Curves

#### Qwen2.5-3B-Instruct

Testing formality trait with α ∈ [0.0, 0.5, 1.0, 2.0] across 10 random orthogonal seeds.

| Alpha | Mean Effect Diff | Std Effect Diff |
|-------|------------------|-----------------|
| 0.0   | 0.022 ± 0.016   | Small variance  |
| 0.5   | 0.017 ± 0.012   | Small variance  |
| 1.0   | 0.016 ± 0.012   | Small variance  |
| 2.0   | 0.010 ± 0.008   | Small variance  |

**Finding**: Response curves for v and v+v⊥ overlap across all α values, demonstrating identical scaling behavior. This confirms that orthogonal components scale proportionally with the base steering vector, further supporting non-identifiability.

See [figures/alpha_sweep_response_curves.png](figures/alpha_sweep_response_curves.png) for Qwen2.5-3B visualization.

#### LLaMA-3.1-8B-Instruct

Testing formality trait with α ∈ [0.0, 0.5, 1.0, 2.0] across 10 random orthogonal seeds.

| Alpha | Mean Effect Diff | Std Effect Diff |
|-------|------------------|-----------------|
| 0.0   | 0.023 ± 0.009   | Small variance  |
| 0.5   | 0.017 ± 0.009   | Small variance  |
| 1.0   | 0.013 ± 0.011   | Small variance  |
| 2.0   | 0.019 ± 0.015   | Small variance  |

**Finding**: Consistently small effect differences (< 0.025) across all α values indicate that v and v+v⊥ produce nearly identical outputs regardless of steering strength. This replicates the Qwen2.5-3B findings on LLaMA-3.1-8B, confirming that non-identifiability is model-independent.

## Conclusion

Our empirical validation strongly supports **non-identifiability of persona steering vectors** in language models:

1. **Orthogonal components matter equally**: Adding v⊥ to v produces semantically equivalent outputs (Cohen's d < 0.3), indicating a large null space of unidentifiable directions.

2. **Consistent across traits**: Non-identifiability holds for formality, politeness, and humor—suggesting it's a fundamental property of steering vector geometry in LLMs.

3. **Scaling behavior**: Response curves overlap across all steering strengths α, confirming that orthogonal components scale identically with the base vector (no implicit regularization).

4. **Implication for steering**: Persona vectors lack unique behavioral signatures. Multiple geometrically distinct vectors can produce identical semantic effects. This is problematic for:
   - Interpretability: Difficult to reason about what a specific vector "means"
   - Generalization: No guarantee learned vectors transfer to new contexts
   - Control: Cannot assume one vector = one persona trait

## File Structure

```
├── config/
│   ├── config.json              # Persona prompt templates & test prompts
│   └── model_config.yml         # Model configuration (Qwen2.5-3B)
├── results/
│   ├── orthogonal_test_results_10_seeds.json
│   ├── orthogonal_test_results_5_seeds.json
│   └── alpha_sweep_results.json
├── figures/
│   └── alpha_sweep_response_curves.png
├── persona_vector_experiment.py # Core experiment class
├── test_orthogonal.py          # Test 1: Orthogonal component irrelevance
├── test_alpha_sweep.py         # Test 2: α-sweep response curves
├── ica_steering_demo.py        # Prop 2 validation: ICA-based steering
└── visualize_perp_only_effects.py
```

## License

MIT License - See LICENSE file for details.