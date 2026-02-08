#!/usr/bin/env python3
"""
Generate Optimized LLM Prompts for Causal Discovery Research

Creates:
1. Main experiment: 78 single prompts (6 algorithms × 13 datasets)
2. Supplementary validation: 36 variation prompts (subset testing)

Usage: python generate_prompts.py
"""

from pathlib import Path
from typing import Dict

# Algorithm specifications
ALGORITHMS = {
    'PC': {
        'name': 'PC (Peter-Clark)',
        'type': 'Constraint-based',
        'properties': [
            'Tests conditional independence to discover edges',
            'Assumes causal sufficiency (no hidden confounders)', 
            'Assumes faithfulness condition',
            'Sensitive to significance level (alpha parameter)',
            'May produce undirected edges when orientation uncertain'
        ],
        'strengths': [
            'Theoretically well-founded',
            'Works with discrete and continuous data',
            'Produces interpretable partial DAGs'
        ],
        'weaknesses': [
            'Poor performance with small samples',
            'Sensitive to independence test choice',
            'Many undirected edges reduce identifiability'
        ]
    },
    'LiNGAM': {
        'name': 'LiNGAM (Linear Non-Gaussian Acyclic Model)',
        'type': 'Functional causal model',
        'properties': [
            'Exploits non-Gaussianity to identify causal direction',
            'Assumes linear relationships between variables',
            'Requires non-Gaussian error terms',
            'Produces fully oriented DAG',
            'No hidden confounders allowed'
        ],
        'strengths': [
            'Uniquely identifies causal direction',
            'Strong performance on linear data',
            'No undirected edges (complete orientation)'
        ],
        'weaknesses': [
            'Fails when data is Gaussian',
            'Assumes strict linearity',
            'Sensitive to nonlinear relationships'
        ]
    },
    'FCI': {
        'name': 'FCI (Fast Causal Inference)',
        'type': 'Constraint-based with latent confounders',
        'properties': [
            'Extension of PC allowing hidden confounders',
            'Uses special edge types (o->, <->)',
            'Tests conditional independence like PC',
            'Weaker identifiability assumptions',
            'Returns partial ancestral graph (PAG)'
        ],
        'strengths': [
            'Handles latent confounders',
            'More realistic assumptions',
            'Theoretically sound'
        ],
        'weaknesses': [
            'Complex output interpretation',
            'Many ambiguous edges',
            'Lower precision with confounders'
        ]
    },
    'NOTEARS': {
        'name': 'NOTEARS (No Tears)',
        'type': 'Score-based continuous optimization',
        'properties': [
            'Uses acyclicity as continuous constraint',
            'Differentiable optimization approach',
            'Assumes linear relationships',
            'L1 regularization for sparsity',
            'Modern neural-inspired method'
        ],
        'strengths': [
            'Scalable to larger graphs',
            'Modern optimization framework',
            'Theoretically justified acyclicity constraint'
        ],
        'weaknesses': [
            'Strict linearity assumption',
            'May get trapped in local minima',
            'Requires careful regularization tuning'
        ]
    },
    'GES': {
        'name': 'GES (Greedy Equivalence Search)',
        'type': 'Score-based greedy search',
        'properties': [
            'Searches over equivalence classes of DAGs',
            'Forward phase: add edges to increase score',
            'Backward phase: remove edges',
            'Uses BIC or BDeu scoring functions',
            'Returns best-scoring DAG'
        ],
        'strengths': [
            'No linearity assumptions',
            'Works with mixed data types',
            'Generally robust performance'
        ],
        'weaknesses': [
            'Greedy search can miss global optimum', 
            'Computationally expensive for large graphs',
            'Performance depends on scoring function choice'
        ]
    },
    'GRaSP': {
        'name': 'GRaSP (Greedy Relaxation of Sparsest Permutation)',
        'type': 'Permutation-based search',
        'properties': [
            'Searches over variable orderings',
            'Finds sparsest graph for each ordering',
            'Greedy relaxation strategy',
            'Relatively recent algorithm',
            'Handles various data types'
        ],
        'strengths': [
            'Strong theoretical guarantees',
            'Flexible data handling',
            'Often computationally efficient'
        ],
        'weaknesses': [
            'Less empirical validation than PC/LiNGAM',
            'Assumes sparse graph structure',
            'Limited comparative studies available'
        ]
    }
}

# Dataset specifications
DATASETS = {
    'titanic': {
        'name': 'Titanic',
        'domain': 'Social Science (Survival Analysis)',
        'nodes': 7, 'edges': 5, 'samples': 891,
        'data_type': 'Mixed (categorical + continuous)',
        'complexity': 'Low',
        'context': 'Real-world passenger survival data'
    },
    'asia': {
        'name': 'Asia',
        'domain': 'Medical (Diagnostic Network)',
        'nodes': 8, 'edges': 8, 'samples': 10000,
        'data_type': 'Discrete (Binary)',
        'complexity': 'Low',
        'context': 'Benchmark medical diagnosis network'
    },
    'cancer': {
        'name': 'Cancer',
        'domain': 'Medical (Lung Cancer)',
        'nodes': 5, 'edges': 4, 'samples': 10000,
        'data_type': 'Discrete (Binary)',
        'complexity': 'Very Low',
        'context': 'Simple lung cancer risk factors'
    },
    'earthquake': {
        'name': 'Earthquake', 
        'domain': 'Seismic (Earthquake Detection)',
        'nodes': 7, 'edges': 7, 'samples': 10000,
        'data_type': 'Discrete (Binary)',
        'complexity': 'Low',
        'context': 'Seismic event detection network'
    },
    'sachs': {
        'name': 'Sachs',
        'domain': 'Biology (Protein Signaling)',
        'nodes': 11, 'edges': 17, 'samples': 7466,
        'data_type': 'Continuous (Flow Cytometry)',
        'complexity': 'Medium',
        'context': 'Real protein signaling pathway data'
    },
    'survey': {
        'name': 'Survey',
        'domain': 'Social Science (Survey Research)',
        'nodes': 6, 'edges': 6, 'samples': 10000,
        'data_type': 'Mixed (Ordinal + Continuous)',
        'complexity': 'Low',
        'context': 'Survey response patterns'
    },
    'child': {
        'name': 'Child',
        'domain': 'Medical (Pediatric Diagnosis)',
        'nodes': 20, 'edges': 25, 'samples': 10000,
        'data_type': 'Discrete (Categorical)',
        'complexity': 'Medium-High',
        'context': 'Pediatric disease diagnosis network'
    },
    'alarm': {
        'name': 'Alarm',
        'domain': 'Medical (ICU Monitoring)',
        'nodes': 37, 'edges': 46, 'samples': 20000,
        'data_type': 'Discrete (Multi-state)',
        'complexity': 'High',
        'context': 'ICU patient monitoring system'
    },
    'barley': {
        'name': 'Barley',
        'domain': 'Agricultural (Crop Genetics)',
        'nodes': 48, 'edges': 84, 'samples': 1000,
        'data_type': 'Mixed (Genetic + Environmental)',
        'complexity': 'Very High',
        'context': 'Barley variety and environment interactions'
    },
    'stock_market': {
        'name': 'Stock Market',
        'domain': 'Finance (Market Dynamics)',
        'nodes': 10, 'edges': 18, 'samples': 5000,
        'data_type': 'Continuous (Time Series)',
        'complexity': 'Medium',
        'context': 'Financial market interdependencies'
    },
    'insurance': {
        'name': 'Insurance',
        'domain': 'Insurance (Risk Assessment)',
        'nodes': 27, 'edges': 52, 'samples': 10000,
        'data_type': 'Mixed (Claims + Demographics)',
        'complexity': 'High',
        'context': 'Insurance risk evaluation network'
    },
    'synthetic_12': {
        'name': 'Synthetic 12-Node',
        'domain': 'Simulation (Controlled)',
        'nodes': 12, 'edges': 15, 'samples': 5000,
        'data_type': 'Continuous (Generated)',
        'complexity': 'Medium',
        'context': 'Controlled synthetic DAG, known ground truth'
    },
    'synthetic_30': {
        'name': 'Synthetic 30-Node',
        'domain': 'Simulation (Controlled)',
        'nodes': 30, 'edges': 45, 'samples': 5000,
        'data_type': 'Continuous (Generated)',
        'complexity': 'High', 
        'context': 'Large synthetic DAG, challenging complexity'
    }
}


def generate_main_prompt(algorithm: str, dataset: str) -> str:
    """Generate single optimized prompt for algorithm-dataset combination."""
    
    algo_spec = ALGORITHMS[algorithm]
    data_spec = DATASETS[dataset]
    
    return f"""You are an expert in causal discovery algorithms with deep knowledge of their empirical performance characteristics.

========================================
EXPERIMENTAL SETUP
========================================
Algorithm: {algo_spec['name']}
Type: {algo_spec['type']}
Dataset: {data_spec['name']}
Domain: {data_spec['domain']}

Dataset Characteristics:
- Variables: {data_spec['nodes']} nodes
- True causal edges: {data_spec['edges']} 
- Sample size: {data_spec['samples']} observations
- Data type: {data_spec['data_type']}
- Structural complexity: {data_spec['complexity']}
- Context: {data_spec['context']}

Algorithm Properties:
""" + '\n'.join(f"• {prop}" for prop in algo_spec['properties']) + f"""

Strengths:
""" + '\n'.join(f"• {strength}" for strength in algo_spec['strengths']) + f"""

Weaknesses:
""" + '\n'.join(f"• {weakness}" for weakness in algo_spec['weaknesses']) + f"""

========================================
PERFORMANCE PREDICTION TASK
========================================
The algorithm will be run 100 times with:
- Different random seeds and bootstrap samples
- Hyperparameter variations (alpha, regularization, etc.)
- Performance measured against known ground truth

Metrics:
- Precision: correctly identified edges / total identified edges (0-1)
- Recall: correctly identified edges / total true edges (0-1) 
- F1-score: harmonic mean of precision and recall (0-1)
- SHD: Structural Hamming Distance (integer, total edge errors)

Based on algorithm properties, dataset characteristics, and typical empirical performance, predict REALISTIC RANGES:

REQUIRED FORMAT:
Precision: [lower, upper]
Recall: [lower, upper]  
F1: [lower, upper]
SHD: [lower, upper]

Then explain in 3-4 sentences:
1. Why you chose these ranges
2. Main performance-limiting factors
3. Your confidence level (High/Medium/Low)

Example:
Precision: [0.60, 0.75]
Recall: [0.55, 0.70]
F1: [0.58, 0.72] 
SHD: [8, 15]

Reasoning: This algorithm matches the data type well and assumptions should hold. Precision will likely exceed recall due to conservative edge identification. The moderate complexity suggests stable performance. Medium confidence due to hyperparameter sensitivity."""


def generate_variations(base_prompt: str, algorithm: str, dataset: str) -> list:
    """Generate 4 principled variations for supplementary validation."""
    
    algo_spec = ALGORITHMS[algorithm]
    data_spec = DATASETS[dataset]
    
    # V1: Original structured format (keep base)
    v1 = base_prompt
    
    # V2: Direct conversational style
    v2 = f"""I need your expert prediction on algorithm performance.

I'm testing {algo_spec['name']} (a {algo_spec['type']} method) on {data_spec['name']} data:
- {data_spec['nodes']} variables, {data_spec['edges']} true edges
- {data_spec['samples']} samples, {data_spec['data_type'].lower()} 
- Domain: {data_spec['domain']}

Based on your experience, what performance ranges would you expect after 100 test runs?

For each metric, give me your best estimate:

Precision (correct/identified): [?, ?] 
Recall (correct/true): [?, ?]
F1-score: [?, ?] 
SHD (total errors): [?, ?]

Explain your reasoning briefly."""

    # V3: Minimal context
    v3 = f"""Algorithm: {algo_spec['name']}
Dataset: {data_spec['name']} ({data_spec['nodes']} vars, {data_spec['samples']} samples)

Predict performance after 100 runs:
Precision: [?, ?]
Recall: [?, ?]
F1: [?, ?] 
SHD: [?, ?]

Reasoning:"""

    # V4: Comparative framing  
    v4 = f"""Performance comparison question:

For {algo_spec['name']} on {data_spec['name']} data ({data_spec['complexity'].lower()} complexity):

What would represent:
- GOOD performance ranges?
- POOR performance ranges?  
- MOST LIKELY performance ranges?

Your realistic estimates:
Precision: [?, ?]
Recall: [?, ?]
F1: [?, ?]
SHD: [?, ?]

Why this performance level?"""

    return [v1, v2, v3, v4]


def main():
    """Generate all prompts."""
    
    prompts_dir = Path(__file__).parent
    main_dir = prompts_dir / 'main'
    supplement_dir = prompts_dir / 'supplement'
    
    main_dir.mkdir(exist_ok=True)
    supplement_dir.mkdir(exist_ok=True)
    
    print("Generating LLM prompts for causal discovery research...\n")
    
    # 1. Generate 78 main prompts (one per combination)
    print("MAIN EXPERIMENT PROMPTS:")
    print("="*50)
    main_count = 0
    for algorithm in ALGORITHMS:
        for dataset in DATASETS:
            prompt = generate_main_prompt(algorithm, dataset)
            
            filename = main_dir / f"{algorithm}_{dataset}.txt"
            with open(filename, 'w') as f:
                f.write(prompt)
            
            main_count += 1
            print(f"✓ {algorithm:8s} × {dataset:15s}")
    
    # 2. Generate supplementary validation (subset with variations)
    print(f"\nSUPPLEMENTARY VALIDATION:")
    print("="*50)
    
    # Subset: diverse algorithms and datasets
    subset_algos = ['PC', 'LiNGAM', 'FCI']
    subset_datasets = ['titanic', 'asia', 'sachs']
    
    supplement_count = 0
    for algorithm in subset_algos:
        for dataset in subset_datasets:
            base_prompt = generate_main_prompt(algorithm, dataset)
            variations = generate_variations(base_prompt, algorithm, dataset)
            
            variation_names = ['structured', 'conversational', 'minimal', 'comparative']
            for i, (variation, name) in enumerate(zip(variations, variation_names), 1):
                filename = supplement_dir / f"{algorithm}_{dataset}_v{i}_{name}.txt"
                with open(filename, 'w') as f:
                    f.write(variation)
                supplement_count += 1
                print(f"✓ {algorithm:7s} × {dataset:8s} V{i} ({name})")
    
    print(f"\n{'='*60}")
    print(f"PROMPT GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Main experiment: {main_count} prompts ({len(ALGORITHMS)} algs × {len(DATASETS)} datasets)")
    print(f"Supplementary:   {supplement_count} prompts ({len(subset_algos)} algs × {len(subset_datasets)} datasets × 4 variations)")
    print(f"Total queries:   ~{main_count * 6 + supplement_count * 6} (assuming 6 LLMs)")
    print(f"\nFiles saved to:")
    print(f"  Main: {main_dir}/")
    print(f"  Supplement: {supplement_dir}/")


if __name__ == "__main__":
    main()