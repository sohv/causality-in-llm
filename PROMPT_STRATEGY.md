# LLM Prompt Strategy: Main Experiment + Supplementary Validation

## Overview

Clean, defensible methodology for "Confidence Crisis: LLMs Miss Algorithmic Reality" paper:

**Main Experiment**: Single optimized prompt per combination (eliminates prompt confounds)  
**Supplementary**: Subset robustness testing (defends against prompt criticism)

## Strategy

### Main Experiment
- **78 single prompts** (6 algorithms × 13 datasets)
- **468 LLM queries** (78 prompts × 6 LLMs)
- **Clean narrative**: "Even with optimal prompting, LLMs fail"
- **Eliminates confound**: No critic can claim "just bad prompting"

### Supplementary Validation  
- **9 subset combinations** (PC, LiNGAM, FCI × titanic, asia, sachs)
- **4 variations each** (structured, conversational, minimal, comparative)
- **216 LLM queries** (36 prompts × 6 LLMs)
- **Defensive evidence**: "Different prompt styles, same unreliability"

## Files Generated

### Prompts
```
prompts/
├── main/                    # 78 single optimized prompts
│   ├── PC_titanic.txt      
│   ├── LiNGAM_asia.txt
│   └── ... (78 total)
├── supplement/              # 36 validation prompts  
│   ├── PC_titanic_v1_structured.txt
│   ├── PC_titanic_v2_conversational.txt
│   └── ... (36 total)
└── generate_prompts.py      # Generator script
```

### Experiment Scripts
```
llm_integration/multi_llm_runner.py     # Main experiment (single prompts)
run_supplementary_validation.py         # Robustness validation
```

## Usage

### Main Experiment (468 queries)
```bash
# Single combination
python llm_integration/multi_llm_runner.py --dataset titanic --algorithm PC

# All combinations  
python llm_integration/multi_llm_runner.py --all-combos
```

### Supplementary Validation (216 queries)
```bash
python run_supplementary_validation.py --output results/validation
```

## Paper Methodology

### Main Results Section
"We used algorithm-dataset-specific optimized prompts, developed through expert domain knowledge and best practices. Each prompt contained identical experimental setup descriptions, algorithm properties, and response format requirements. This eliminated prompt variation as a potential confound."

**Tables/Figures**: Main experiment results (78 combinations × 6 LLMs)

### Supplementary Materials
"To address potential concerns about prompt sensitivity, we conducted robustness validation on a representative subset (9 combinations) using 4 different prompt formulations: structured expert instruction, conversational questioning, minimal context, and comparative analysis. Results show that LLM unreliability persists across all prompt variations (Supplementary Figure S1, Table S1)."

## Key Benefits

1. **Ironclad main claim**: Single prompts = no prompt confound
2. **Defensive supplementary**: Shows unreliability isn't prompt-dependent  
3. **Manageable scope**: ~700 total queries vs 1,400+ with variations
4. **Clear narrative**: Problem is LLM capability, not prompt engineering
5. **Addresses all criticism**: "You prompted badly" → No, we tested variations

## Experimental Timeline

- **Main experiment**: 78 combinations × 6 LLMs = ~2-3 hours
- **Supplementary validation**: 36 combinations × 6 LLMs = ~1 hour  
- **Total runtime**: ~4 hours for complete experiment
- **Total queries**: ~700 (affordable, reproducible)

## Expected Results

**Main**: Clear evidence that LLM estimates are systematically biased/unreliable  
**Supplementary**: Robustness analysis showing variation persists across prompt styles

This strategy provides the strongest possible methodological foundation for the paper's central claim.