#!/usr/bin/env python3
"""
FCI (Fast Causal Inference) Algorithm Experiments
-------------------------------------------------
Run FCI algorithm experiments across ALL datasets with variance analysis.
Uses the SAME datasets as PC and LiNGAM for fair comparison.

Datasets (9 total):
- Benchmarks (6): asia, cancer, earthquake, sachs, survey, child
- Synthetic (2): 12 nodes, 30 nodes
- Real-world (1): Titanic

For each dataset:
- 100 runs with varying alpha (0.01-0.10)
- Bootstrap sampling (same as PC/LiNGAM)
- Compute: Precision, Recall, F1, SHD
- Bootstrap 95% CIs
- Generate LLM prompts
- Compare with PC/LiNGAM results
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "variance"))
sys.path.append(str(Path(__file__).parent.parent / "datasets"))

import numpy as np
import pandas as pd
import json
from variance_analysis import VarianceAnalyzer
from run_experiments import (
    load_titanic,
    load_bnlearn_network,
    generate_synthetic_dag
)

# Import new datasets
from alarm_network import load_alarm
from stock_market import load_stock_market
from insurance_network import load_insurance
from barley_network import load_barley

# ============================================================================
# FCI Experiment Runners (matching PC/LiNGAM structure)
# ============================================================================

def run_fci_titanic(analyzer: VarianceAnalyzer):
    """Run Titanic + FCI experiments."""
    print("\n" + "="*80)
    print("TITANIC DATASET - FCI")
    print("="*80)

    data, true_graph = load_titanic()
    print(f"Loaded Titanic: {data.shape[0]} samples, {data.shape[1]} variables")
    print("Variables: pclass, sex, age, sibsp, parch, fare, survived")

    # Run FCI with variance analysis
    results = analyzer.run_fci_multiple(data, true_graph)

    # Print summary
    print("\n--- FCI Algorithm Performance (Mean ± 95% CI) ---")
    print(f"Precision: {results.precision.mean:.4f} "
          f"[{results.precision.ci_lower:.4f}, {results.precision.ci_upper:.4f}]")
    print(f"Recall:    {results.recall.mean:.4f} "
          f"[{results.recall.ci_lower:.4f}, {results.recall.ci_upper:.4f}]")
    print(f"F1-score:  {results.f1.mean:.4f} "
          f"[{results.f1.ci_lower:.4f}, {results.f1.ci_upper:.4f}]")
    print(f"SHD:       {results.shd.mean:.1f} "
          f"[{results.shd.ci_lower:.1f}, {results.shd.ci_upper:.1f}]")

    # Save results
    analyzer.save_results(results, 'titanic', 'fci')

    # Generate LLM prompt
    prompt = generate_llm_prompt('titanic', results, {
        'n_nodes': 7,
        'n_edges': int(np.sum(true_graph)),
        'description': 'Titanic survival prediction with socioeconomic factors',
        'variables': ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'survived']
    })

    save_llm_prompt(analyzer.output_dir, 'titanic', prompt)

    return results


def run_fci_benchmark_experiments(analyzer: VarianceAnalyzer):
    """Run FCI on ALL bnlearn benchmark networks (same as PC/LiNGAM)."""
    benchmarks = ['asia', 'cancer', 'earthquake', 'sachs', 'survey', 'child']

    all_results = {}

    for bench_name in benchmarks:
        print("\n" + "="*80)
        print(f"BENCHMARK: {bench_name.upper()} - FCI")
        print("="*80)

        try:
            data, true_graph, nodes = load_bnlearn_network(bench_name)
            print(f"Loaded {bench_name}: {len(nodes)} nodes, {data.shape[0]} samples")

            # Run FCI algorithm
            print("\n--- Running FCI Algorithm ---")
            results_fci = analyzer.run_fci_multiple(data, true_graph)

            print(f"FCI - Precision: {results_fci.precision.mean:.4f} "
                  f"[{results_fci.precision.ci_lower:.4f}, {results_fci.precision.ci_upper:.4f}]")
            print(f"FCI - Recall:    {results_fci.recall.mean:.4f} "
                  f"[{results_fci.recall.ci_lower:.4f}, {results_fci.recall.ci_upper:.4f}]")
            print(f"FCI - F1:        {results_fci.f1.mean:.4f} "
                  f"[{results_fci.f1.ci_lower:.4f}, {results_fci.f1.ci_upper:.4f}]")
            print(f"FCI - SHD:       {results_fci.shd.mean:.1f} "
                  f"[{results_fci.shd.ci_lower:.1f}, {results_fci.shd.ci_upper:.1f}]")

            # Check for catastrophic collapse on larger networks
            if len(nodes) >= 20 and results_fci.f1.mean < 0.2:
                print(f"\n[WARNING] Catastrophic collapse detected on {bench_name}!")
                print(f"         F1-score: {results_fci.f1.mean:.4f} < 0.2")

            analyzer.save_results(results_fci, bench_name, 'fci')
            all_results[f"{bench_name}_fci"] = results_fci

            # Generate LLM prompt
            prompt = generate_llm_prompt(bench_name, results_fci, {
                'n_nodes': len(nodes),
                'n_edges': int(np.sum(true_graph)),
                'description': f'{bench_name.title()} benchmark network',
                'variables': nodes
            })
            save_llm_prompt(analyzer.output_dir, bench_name, prompt)

        except Exception as e:
            print(f"Error processing {bench_name}: {e}")
            continue

    return all_results


def run_fci_synthetic_experiments(analyzer: VarianceAnalyzer):
    """Run FCI on synthetic DAGs (same as PC/LiNGAM)."""
    node_counts = [12, 30]

    all_results = {}

    for n_nodes in node_counts:
        print("\n" + "="*80)
        print(f"SYNTHETIC DAG - {n_nodes} NODES - FCI")
        print("="*80)

        data, true_graph = generate_synthetic_dag(n_nodes, edge_prob=0.2)
        print(f"Generated DAG: {n_nodes} nodes, {np.sum(true_graph)} edges")

        if n_nodes == 30:
            print("NOTE: Expect catastrophic collapse at 30 nodes!")

        # Run FCI
        print("\n--- Running FCI Algorithm ---")
        results_fci = analyzer.run_fci_multiple(data, true_graph)

        print(f"FCI - Precision: {results_fci.precision.mean:.4f} "
              f"[{results_fci.precision.ci_lower:.4f}, {results_fci.precision.ci_upper:.4f}]")
        print(f"FCI - Recall:    {results_fci.recall.mean:.4f} "
              f"[{results_fci.recall.ci_lower:.4f}, {results_fci.recall.ci_upper:.4f}]")
        print(f"FCI - F1:        {results_fci.f1.mean:.4f} "
              f"[{results_fci.f1.ci_lower:.4f}, {results_fci.f1.ci_upper:.4f}]")
        print(f"FCI - SHD:       {results_fci.shd.mean:.1f} "
              f"[{results_fci.shd.ci_lower:.1f}, {results_fci.shd.ci_upper:.1f}]")

        # Check for catastrophic collapse
        if results_fci.f1.mean < 0.2:
            print(f"\n[CATASTROPHIC COLLAPSE] F1 = {results_fci.f1.mean:.4f} < 0.2")
            print("This demonstrates FCI's scaling limitations!")

        analyzer.save_results(results_fci, f'synthetic_{n_nodes}', 'fci')
        all_results[f"synthetic_{n_nodes}_fci"] = results_fci

        # Generate LLM prompt
        prompt = generate_llm_prompt(f'synthetic_{n_nodes}', results_fci, {
            'n_nodes': n_nodes,
            'n_edges': int(np.sum(true_graph)),
            'description': f'Synthetic DAG with linear Gaussian data ({n_nodes} nodes)',
            'variables': [f"X{i}" for i in range(n_nodes)]
        })
        save_llm_prompt(analyzer.output_dir, f'synthetic_{n_nodes}', prompt)

    return all_results


def run_fci_new_datasets(analyzer: VarianceAnalyzer):
    """Run FCI on new datasets: Alarm, Stock Market, Insurance, Barley."""
    all_results = {}

    new_datasets = [
        ('alarm', lambda: load_alarm(n_samples=5000), 'Medical ICU Monitoring'),
        ('stock_market', lambda: load_stock_market(n_samples=1000), 'Financial Relationships'),
        ('insurance', lambda: load_insurance(n_samples=2000), 'Insurance Risk Assessment'),
        ('barley', lambda: load_barley(n_samples=3000), 'Agricultural Crop Production'),
    ]

    for ds_name, loader, description in new_datasets:
        print("\n" + "="*80)
        print(f"{ds_name.upper()} - FCI")
        print("="*80)

        try:
            data, true_graph, nodes = loader()
            print(f"Loaded {ds_name}: {len(nodes)} nodes, {data.shape[0]} samples")

            results = analyzer.run_fci_multiple(data, true_graph)

            print(f"FCI - Precision: {results.precision.mean:.4f} "
                  f"[{results.precision.ci_lower:.4f}, {results.precision.ci_upper:.4f}]")
            print(f"FCI - Recall:    {results.recall.mean:.4f} "
                  f"[{results.recall.ci_lower:.4f}, {results.recall.ci_upper:.4f}]")
            print(f"FCI - F1:        {results.f1.mean:.4f} "
                  f"[{results.f1.ci_lower:.4f}, {results.f1.ci_upper:.4f}]")
            print(f"FCI - SHD:       {results.shd.mean:.1f} "
                  f"[{results.shd.ci_lower:.1f}, {results.shd.ci_upper:.1f}]")

            analyzer.save_results(results, ds_name, 'fci')
            all_results[f"{ds_name}_fci"] = results

        except Exception as e:
            print(f"Error processing {ds_name}: {e}")
            continue

    return all_results


# ============================================================================
# LLM Prompt Generation
# ============================================================================

def generate_llm_prompt(dataset_name: str, results, dataset_info: dict) -> str:
    """Generate a prompt for LLMs to predict FCI algorithm performance."""

    prompt = f"""You are tasked with predicting the performance of the FCI (Fast Causal Inference) algorithm on a causal discovery task.

Dataset: {dataset_name.upper()}
Description: {dataset_info['description']}
Number of nodes: {dataset_info['n_nodes']}
Number of edges: {dataset_info['n_edges']}
Variables: {', '.join(dataset_info['variables'])}

The FCI algorithm:
- Handles latent confounders and selection bias
- Uses conditional independence tests (Fisher's z-test)
- Outputs a Partial Ancestral Graph (PAG) with various edge types
- Varies alpha parameter from 0.01 to 0.10 across 100 runs
- Uses bootstrap sampling for robustness

Based on your knowledge of:
1. The FCI algorithm's characteristics and typical performance
2. The dataset domain and structure
3. Statistical properties of the data

Please predict the following performance metrics for FCI on this dataset:

**Your Task:**
Provide your predicted ranges (min, max) for:
1. Precision (fraction of predicted edges that are correct): [min, max]
2. Recall (fraction of true edges that are detected): [min, max]
3. F1-score (harmonic mean of precision and recall): [min, max]
. SHD (Structural Hamming Distance - number of edge errors): [min, max]

**Format your response as JSON:**
{{
  "precision": [min, max],
  "recall": [min, max],
  "f1": [min, max],
  "shd": [min, max],
  "reasoning": "Brief explanation of your predictions"
}}

**Ground Truth (Hidden from LLM):**
After you provide your predictions, they will be compared with the actual results:
- Precision: {results.precision.mean:.4f} [{results.precision.ci_lower:.4f}, {results.precision.ci_upper:.4f}]
- Recall: {results.recall.mean:.4f} [{results.recall.ci_lower:.4f}, {results.recall.ci_upper:.4f}]
- F1: {results.f1.mean:.4f} [{results.f1.ci_lower:.4f}, {results.f1.ci_upper:.4f}]
- SHD: {results.shd.mean:.1f} [{results.shd.ci_lower:.1f}, {results.shd.ci_upper:.1f}]
"""

    return prompt


def save_llm_prompt(output_dir: Path, dataset_name: str, prompt: str):
    """Save LLM prompt to file."""
    prompt_file = output_dir / f"{dataset_name}_fci_llm_prompt.txt"
    with open(prompt_file, 'w') as f:
        f.write(prompt)
    print(f"\n LLM prompt saved to: {prompt_file}")


def compare_llm_predictions(analyzer: VarianceAnalyzer, dataset_name: str,
                           results, llm_predictions: dict):
    """Compare LLM predictions with actual FCI results."""
    print("\n" + "="*80)
    print(f"LLM PREDICTION COMPARISON: {dataset_name.upper()}")
    print("="*80)

    comparison = analyzer.compare_with_llm_estimates(results, llm_predictions)

    for metric, comp in comparison.items():
        overlap = "[PASS]" if comp['overlaps'] else "[FAIL]"
        print(f"{metric:10s}: {overlap} Overlap: {comp['overlap_percentage']:.1f}%")
        print(f"             Ground Truth: [{comp['algorithmic_ci'][0]:.4f}, {comp['algorithmic_ci'][1]:.4f}]")
        print(f"             LLM Estimate: [{comp['llm_range'][0]:.4f}, {comp['llm_range'][1]:.4f}]")

    # Save comparison
    comparison_data = {
        'dataset': dataset_name,
        'algorithm': 'fci',
        'comparison': comparison
    }

    comparison_file = analyzer.output_dir / f'{dataset_name}_fci_llm_comparison.json'
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)

    print(f"\nComparison saved to: {comparison_file}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run FCI experiments on ALL datasets (same as PC/LiNGAM)")
    parser.add_argument('--runs', type=int, default=100,
                       help='Number of runs per algorithm (default: 100)')
    parser.add_argument('--output', type=str, default='fci_results',
                       help='Output directory (default: fci_results)')
    parser.add_argument('--experiments', nargs='+',
                       choices=['titanic', 'benchmarks', 'synthetic', 'new_datasets', 'all'],
                       default=['all'],
                       help='Which experiments to run (default: all)')

    args = parser.parse_args()

    # Initialize analyzer
    output_dir = Path(__file__).parent / args.output
    analyzer = VarianceAnalyzer(n_runs=args.runs, output_dir=str(output_dir))

    print("="*80)
    print("FCI ALGORITHM EXPERIMENTS")
    print("="*80)
    print(f"Runs per dataset: {args.runs}")
    print(f"Output directory: {output_dir}")
    print()
    print("Datasets (matching PC/LiNGAM experiments):")
    print("- Benchmarks (6): asia, cancer, earthquake, sachs, survey, child")
    print("- Synthetic (2): 12 nodes, 30 nodes")
    print("- Real-world (1): Titanic")
    print("TOTAL: 9 datasets")
    print()

    results = {}

    # Run experiments (same order as PC/LiNGAM)
    if 'all' in args.experiments or 'titanic' in args.experiments:
        results['titanic'] = run_fci_titanic(analyzer)

    if 'all' in args.experiments or 'benchmarks' in args.experiments:
        results['benchmarks'] = run_fci_benchmark_experiments(analyzer)

    if 'all' in args.experiments or 'synthetic' in args.experiments:
        results['synthetic'] = run_fci_synthetic_experiments(analyzer)

    if 'all' in args.experiments or 'new_datasets' in args.experiments:
        results['new_datasets'] = run_fci_new_datasets(analyzer)

    print("\n" + "="*80)
    print("ALL FCI EXPERIMENTS COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}/")
    print(f"\nProcessed {9 if 'all' in args.experiments else 'selected'} datasets")
    print("\nGenerated files:")
    print("  - *_fci_variance.json       (Performance metrics with CIs)")
    print("  - *_fci_llm_prompt.txt      (Prompts for LLM predictions)")
    print("\nNext steps:")
    print("  1. Compare FCI results with PC/LiNGAM (in ../variance/results)")
    print("  2. Use visualize_fci_results.py to generate comparison plots")
    print("  3. Use LLM prompts for predictions and overlap analysis")

    return results


if __name__ == "__main__":
    main()
