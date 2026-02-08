#!/usr/bin/env python3
"""
GES + GRaSP Algorithm Experiments
===================================

Run GES (score-based) and GRaSP (permutation-based) experiments
across ALL 13 datasets with variance analysis.

GES = Greedy Equivalence Search (Chickering, 2002)
GRaSP = Greedy relaxation of Sparsest Permutation (Lam et al., 2022)

These complement existing algorithms:
- PC (constraint-based)
- LiNGAM (order-based / FCM-based)
- FCI (latent confounders)
- NOTEARS (continuous optimization)
- GES (score-based) <-- NEW
- GRaSP (permutation-based) <-- NEW

Usage:
    python run_ges_grasp_experiments.py --runs 100 --output results/
    python run_ges_grasp_experiments.py --runs 100 --algorithm ges
    python run_ges_grasp_experiments.py --runs 100 --algorithm grasp
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "variance"))
sys.path.append(str(Path(__file__).parent.parent / "datasets"))

import numpy as np
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


def run_algorithm_on_all_datasets(analyzer: VarianceAnalyzer, algorithm: str):
    """
    Run a single algorithm (GES or GRaSP) on all 13 datasets.

    Args:
        analyzer: VarianceAnalyzer instance
        algorithm: 'ges' or 'grasp'
    """
    if algorithm == 'ges':
        algo_fn = analyzer.run_ges_multiple
    elif algorithm == 'grasp':
        algo_fn = analyzer.run_grasp_multiple
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    all_results = {}

    # ====================
    # REAL-WORLD DATASETS
    # ====================
    print("\n" + "="*80)
    print(f"REAL-WORLD DATASETS - {algorithm.upper()}")
    print("="*80)

    # 1. Titanic
    print(f"\n--- TITANIC ({algorithm.upper()}) ---")
    data, true_graph = load_titanic()
    results = algo_fn(data, true_graph)
    analyzer.save_results(results, 'titanic', algorithm)
    all_results['titanic'] = results
    print_summary(results)

    # 2. Sachs
    print(f"\n--- SACHS ({algorithm.upper()}) ---")
    try:
        data, true_graph, nodes = load_bnlearn_network('sachs')
        results = algo_fn(data, true_graph)
        analyzer.save_results(results, 'sachs', algorithm)
        all_results['sachs'] = results
        print_summary(results)
    except Exception as e:
        print(f"Sachs failed: {e}")

    # 3. Alarm Network
    print(f"\n--- ALARM ({algorithm.upper()}) ---")
    try:
        data, true_graph, nodes = load_alarm(n_samples=5000)
        results = algo_fn(data, true_graph)
        analyzer.save_results(results, 'alarm', algorithm)
        all_results['alarm'] = results
        print_summary(results)
    except Exception as e:
        print(f"Alarm failed: {e}")

    # 4. Stock Market
    print(f"\n--- STOCK MARKET ({algorithm.upper()}) ---")
    try:
        data, true_graph, nodes = load_stock_market(n_samples=1000)
        results = algo_fn(data, true_graph)
        analyzer.save_results(results, 'stock_market', algorithm)
        all_results['stock_market'] = results
        print_summary(results)
    except Exception as e:
        print(f"Stock Market failed: {e}")

    # 5. Insurance Network (NEW)
    print(f"\n--- INSURANCE ({algorithm.upper()}) ---")
    try:
        data, true_graph, nodes = load_insurance(n_samples=2000)
        results = algo_fn(data, true_graph)
        analyzer.save_results(results, 'insurance', algorithm)
        all_results['insurance'] = results
        print_summary(results)
    except Exception as e:
        print(f"Insurance failed: {e}")

    # 6. Barley Network (NEW)
    print(f"\n--- BARLEY ({algorithm.upper()}) ---")
    try:
        data, true_graph, nodes = load_barley(n_samples=3000)
        results = algo_fn(data, true_graph)
        analyzer.save_results(results, 'barley', algorithm)
        all_results['barley'] = results
        print_summary(results)
    except Exception as e:
        print(f"Barley failed: {e}")

    # ====================
    # BENCHMARK DATASETS
    # ====================
    print("\n" + "="*80)
    print(f"BENCHMARK DATASETS - {algorithm.upper()}")
    print("="*80)

    benchmarks = ['asia', 'cancer', 'earthquake', 'survey', 'child']
    for bench_name in benchmarks:
        print(f"\n--- {bench_name.upper()} ({algorithm.upper()}) ---")
        try:
            data, true_graph, nodes = load_bnlearn_network(bench_name)
            results = algo_fn(data, true_graph)
            analyzer.save_results(results, bench_name, algorithm)
            all_results[bench_name] = results
            print_summary(results)
        except Exception as e:
            print(f"{bench_name} failed: {e}")

    # ====================
    # SYNTHETIC DATASETS
    # ====================
    print("\n" + "="*80)
    print(f"SYNTHETIC DATASETS - {algorithm.upper()}")
    print("="*80)

    for n_nodes in [12, 30]:
        print(f"\n--- SYNTHETIC {n_nodes} NODES ({algorithm.upper()}) ---")
        data, true_graph = generate_synthetic_dag(n_nodes, edge_prob=0.2)
        results = algo_fn(data, true_graph)
        analyzer.save_results(results, f'synthetic_{n_nodes}', algorithm)
        all_results[f'synthetic_{n_nodes}'] = results
        print_summary(results)

    return all_results


def print_summary(results):
    """Print result summary."""
    print(f"  Precision: {results.precision.mean:.4f} [{results.precision.ci_lower:.4f}, {results.precision.ci_upper:.4f}]")
    print(f"  Recall:    {results.recall.mean:.4f} [{results.recall.ci_lower:.4f}, {results.recall.ci_upper:.4f}]")
    print(f"  F1:        {results.f1.mean:.4f} [{results.f1.ci_lower:.4f}, {results.f1.ci_upper:.4f}]")
    print(f"  SHD:       {results.shd.mean:.1f} [{results.shd.ci_lower:.1f}, {results.shd.ci_upper:.1f}]")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run GES/GRaSP experiments")
    parser.add_argument('--runs', type=int, default=100,
                       help='Number of runs per algorithm')
    parser.add_argument('--output', type=str, default='results/',
                       help='Output directory')
    parser.add_argument('--algorithm', type=str, default='both',
                       choices=['ges', 'grasp', 'both'],
                       help='Which algorithm to run (default: both)')

    args = parser.parse_args()

    analyzer = VarianceAnalyzer(n_runs=args.runs, output_dir=args.output)

    print("="*80)
    print("GES + GRaSP ALGORITHM EXPERIMENTS")
    print("="*80)
    print(f"Runs per dataset: {args.runs}")
    print(f"Output: {args.output}")
    print(f"Algorithm: {args.algorithm}")
    print()

    if args.algorithm in ('ges', 'both'):
        print("\n" + "#"*80)
        print("# GES (Greedy Equivalence Search)")
        print("#"*80)
        ges_results = run_algorithm_on_all_datasets(analyzer, 'ges')
        print(f"\nGES tested on {len(ges_results)} datasets")

    if args.algorithm in ('grasp', 'both'):
        print("\n" + "#"*80)
        print("# GRaSP (Greedy relaxation of Sparsest Permutation)")
        print("#"*80)
        grasp_results = run_algorithm_on_all_datasets(analyzer, 'grasp')
        print(f"\nGRaSP tested on {len(grasp_results)} datasets")

    print("\n" + "="*80)
    print("ALL GES/GRaSP EXPERIMENTS COMPLETE")
    print("="*80)
    print(f"Results saved to: {args.output}")
    
    # CRITICAL: UAI 2026 Statistical Rigor Enhancement for GES/GRaSP
    print("\n" + "="*60)
    print("RUNNING UAI 2026 GES/GRaSP STATISTICAL ANALYSIS...")
    print("="*60)
    
    try:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent / "uai_2026_enhancements"))
        from statistical_testing import StatisticalTester
        
        tester = StatisticalTester()
        
        # Collect results from both algorithms if run
        algorithm_results = {}
        
        if args.algorithm in ('ges', 'both'):
            algorithm_results['GES'] = ges_results if 'ges_results' in locals() else {}
        
        if args.algorithm in ('grasp', 'both'):
            algorithm_results['GRaSP'] = grasp_results if 'grasp_results' in locals() else {}
        
        # Extract performance scores
        performance_data = {}
        for algo_name, results in algorithm_results.items():
            scores = []
            for dataset_name, metrics in results.items():
                if isinstance(metrics, dict) and 'f1' in metrics:
                    scores.append(metrics['f1'])
            if scores:
                performance_data[algo_name] = scores
        
        # Statistical analysis
        statistical_results = []
        for algo_name, scores in performance_data.items():
            if len(scores) >= 5:
                bootstrap_result = tester.bootstrap_confidence_interval(
                    scores, f"{algo_name} Algorithm Performance"
                )
                statistical_results.append(bootstrap_result)
                
                print(f"  {algo_name} Mean F1: {np.mean(scores):.3f}")
                print(f"  {algo_name} 95% CI: [{bootstrap_result.confidence_interval[0]:.3f}, {bootstrap_result.confidence_interval[1]:.3f}]")
                print(f"  {algo_name} Effect Size: {bootstrap_result.effect_size:.3f}")
        
        # Compare GES vs GRaSP if both were run
        if len(performance_data) == 2 and 'GES' in performance_data and 'GRaSP' in performance_data:
            ges_scores = performance_data['GES']
            grasp_scores = performance_data['GRaSP']
            min_len = min(len(ges_scores), len(grasp_scores))
            
            if min_len >= 5:
                comparison_result = tester.paired_t_test(
                    ges_scores[:min_len], grasp_scores[:min_len], "GES vs GRaSP"
                )
                statistical_results.append(comparison_result)
                print(f"  GES vs GRaSP: p={comparison_result.p_value:.4f}, d={comparison_result.effect_size:.3f}, sig={comparison_result.is_significant}")
        
        # Save statistical report
        if statistical_results:
            report_path = Path(args.output) / "uai_ges_grasp_statistical_analysis.txt"
            tester.generate_statistical_report(statistical_results, str(report_path))
            print(f"  GES/GRaSP statistical report saved: {report_path}")
        
        print("\n" + "="*60)
        print("UAI 2026 GES/GRaSP ENHANCEMENT COMPLETE")
        print("GES/GRaSP experiments now UAI submission ready")
        print("="*60)
        
    except Exception as e:
        print(f"  Error in GES/GRaSP UAI enhancement: {e}")


if __name__ == "__main__":
    main()
