#!/usr/bin/env python3
"""
NOTEARS Algorithm Experiments
==============================

Run NOTEARS (gradient-based continuous optimization) experiments
across ALL datasets with variance analysis.

NOTEARS represents the modern continuous optimization paradigm for
causal discovery (Zheng et al., 2018), complementing:
- PC (constraint-based)
- LiNGAM (order-based)
- FCI (latent confounders)

This addresses the "outdated algorithm coverage" reviewer concern.

Usage:
    python run_notears_experiments.py --runs 100 --output results/
"""

import sys
from pathlib import Path
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


def run_notears_on_all_datasets(analyzer: VarianceAnalyzer):
    """
    Run NOTEARS on all 13 datasets.

    Datasets:
    - Real-world (6): Titanic, Sachs, Alarm, Stock Market, Insurance, Barley
    - Benchmarks (5): ASIA, CANCER, EARTHQUAKE, SURVEY, CHILD
    - Synthetic (2): 12-node, 30-node
    """
    all_results = {}

    # ====================
    # REAL-WORLD DATASETS
    # ====================

    print("\n" + "="*80)
    print("REAL-WORLD DATASETS")
    print("="*80)

    # 1. Titanic
    print("\n--- TITANIC ---")
    data, true_graph = load_titanic()
    results_titanic = analyzer.run_notears_multiple(data, true_graph)
    analyzer.save_results(results_titanic, 'titanic', 'notears')
    all_results['titanic'] = results_titanic
    print_summary(results_titanic, 'Titanic')

    # 2. Sachs
    print("\n--- SACHS ---")
    try:
        data, true_graph, nodes = load_bnlearn_network('sachs')
        results_sachs = analyzer.run_notears_multiple(data, true_graph)
        analyzer.save_results(results_sachs, 'sachs', 'notears')
        all_results['sachs'] = results_sachs
        print_summary(results_sachs, 'Sachs')
    except Exception as e:
        print(f"Sachs failed: {e}")

    # 3. Alarm Network
    print("\n--- ALARM NETWORK ---")
    try:
        data, true_graph, nodes = load_alarm(n_samples=5000)
        results_alarm = analyzer.run_notears_multiple(data, true_graph)
        analyzer.save_results(results_alarm, 'alarm', 'notears')
        all_results['alarm'] = results_alarm
        print_summary(results_alarm, 'Alarm')
    except Exception as e:
        print(f"Alarm failed: {e}")

    # 4. Stock Market
    print("\n--- STOCK MARKET ---")
    try:
        data, true_graph, nodes = load_stock_market(n_samples=1000)
        results_stock = analyzer.run_notears_multiple(data, true_graph)
        analyzer.save_results(results_stock, 'stock_market', 'notears')
        all_results['stock_market'] = results_stock
        print_summary(results_stock, 'Stock Market')
    except Exception as e:
        print(f"Stock Market failed: {e}")

    # 5. Insurance Network (NEW)
    print("\n--- INSURANCE ---")
    try:
        data, true_graph, nodes = load_insurance(n_samples=2000)
        results = analyzer.run_notears_multiple(data, true_graph)
        analyzer.save_results(results, 'insurance', 'notears')
        all_results['insurance'] = results
        print_summary(results, 'Insurance')
    except Exception as e:
        print(f"Insurance failed: {e}")

    # 6. Barley Network (NEW)
    print("\n--- BARLEY ---")
    try:
        data, true_graph, nodes = load_barley(n_samples=3000)
        results = analyzer.run_notears_multiple(data, true_graph)
        analyzer.save_results(results, 'barley', 'notears')
        all_results['barley'] = results
        print_summary(results, 'Barley')
    except Exception as e:
        print(f"Barley failed: {e}")

    # ====================
    # BENCHMARK DATASETS
    # ====================

    print("\n" + "="*80)
    print("BENCHMARK DATASETS")
    print("="*80)

    benchmarks = ['asia', 'cancer', 'earthquake', 'survey', 'child']
    for bench_name in benchmarks:
        print(f"\n--- {bench_name.upper()} ---")
        try:
            data, true_graph, nodes = load_bnlearn_network(bench_name)
            results = analyzer.run_notears_multiple(data, true_graph)
            analyzer.save_results(results, bench_name, 'notears')
            all_results[bench_name] = results
            print_summary(results, bench_name.title())
        except Exception as e:
            print(f"{bench_name} failed: {e}")

    # ====================
    # SYNTHETIC DATASETS
    # ====================

    print("\n" + "="*80)
    print("SYNTHETIC DATASETS")
    print("="*80)

    for n_nodes in [12, 30]:
        print(f"\n--- SYNTHETIC {n_nodes} NODES ---")
        data, true_graph = generate_synthetic_dag(n_nodes, edge_prob=0.2)
        results = analyzer.run_notears_multiple(data, true_graph)
        analyzer.save_results(results, f'synthetic_{n_nodes}', 'notears')
        all_results[f'synthetic_{n_nodes}'] = results
        print_summary(results, f'Synthetic-{n_nodes}')

    return all_results


def print_summary(results, dataset_name):
    """Print result summary."""
    print(f"  Precision: {results.precision.mean:.4f} [{results.precision.ci_lower:.4f}, {results.precision.ci_upper:.4f}]")
    print(f"  Recall:    {results.recall.mean:.4f} [{results.recall.ci_lower:.4f}, {results.recall.ci_upper:.4f}]")
    print(f"  F1:        {results.f1.mean:.4f} [{results.f1.ci_lower:.4f}, {results.f1.ci_upper:.4f}]")
    print(f"  SHD:       {results.shd.mean:.1f} [{results.shd.ci_lower:.1f}, {results.shd.ci_upper:.1f}]")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run NOTEARS experiments")
    parser.add_argument('--runs', type=int, default=100,
                       help='Number of runs per algorithm')
    parser.add_argument('--output', type=str, default='results/notears',
                       help='Output directory')

    args = parser.parse_args()

    analyzer = VarianceAnalyzer(n_runs=args.runs, output_dir=args.output)

    print("="*80)
    print("NOTEARS ALGORITHM EXPERIMENTS")
    print("="*80)
    print(f"Runs per dataset: {args.runs}")
    print(f"Output: {args.output}/")
    print()

    all_results = run_notears_on_all_datasets(analyzer)

    print("\n" + "="*80)
    print("ALL NOTEARS EXPERIMENTS COMPLETE")
    print("="*80)
    print(f"Tested on {len(all_results)} datasets")
    print(f"Results saved to: {args.output}/")


if __name__ == "__main__":
    main()
