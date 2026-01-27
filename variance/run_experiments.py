#!/usr/bin/env python3
"""
Run all experiments from the paper with proper variance analysis.

This script reproduces all experiments but with 100 runs per algorithm
to establish proper confidence intervals.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import networkx as nx
from variance_analysis import VarianceAnalyzer
import json

# ============================================================================
# Dataset Loaders
# ============================================================================

def load_titanic():
    """Load and prepare Titanic dataset."""
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import LabelEncoder

    # Load Titanic from OpenML
    titanic = fetch_openml('titanic', version=1, as_frame=True, parser='auto')
    df = titanic.data.copy()
    df['survived'] = titanic.target

    # Clean and prepare
    df = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'survived']].dropna()

    # Encode categorical variables
    le = LabelEncoder()
    df['sex'] = le.fit_transform(df['sex'])

    # Convert survived to int (it's categorical)
    df['survived'] = df['survived'].astype(str).astype(int)

    # Ensure all columns are numeric (convert any remaining categorical)
    for col in df.columns:
        if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows with NaN after conversion
    df = df.dropna()

    # Convert to float64 for numerical stability in LiNGAM
    df = df.astype(np.float64)

    # Define true causal graph (based on domain knowledge)
    # Variables: pclass, sex, age, sibsp, parch, fare, survived
    # Order: 0=pclass, 1=sex, 2=age, 3=sibsp, 4=parch, 5=fare, 6=survived
    n_vars = 7
    true_graph = np.zeros((n_vars, n_vars))

    # Known causal relationships
    true_graph[0, 5] = 1  # pclass -> fare
    true_graph[0, 6] = 1  # pclass -> survived
    true_graph[1, 6] = 1  # sex -> survived
    true_graph[2, 6] = 1  # age -> survived
    true_graph[5, 6] = 1  # fare -> survived

    return df, true_graph


def load_bnlearn_network(name: str):
    """Load a benchmark network from bnlearn."""
    from pgmpy.utils import get_example_model
    from sklearn.preprocessing import LabelEncoder

    # Map network names to pgmpy examples
    network_map = {
        'asia': 'asia',
        'cancer': 'cancer',
        'earthquake': 'earthquake',
        'sachs': 'sachs',
        'survey': 'survey',
        'child': 'child'
    }

    if name.lower() not in network_map:
        raise ValueError(f"Unknown network: {name}")

    # Load model
    model = get_example_model(network_map[name.lower()])

    # Extract true graph as adjacency matrix
    nodes = sorted(model.nodes())
    n = len(nodes)
    true_graph = np.zeros((n, n))

    node_to_idx = {node: i for i, node in enumerate(nodes)}

    for edge in model.edges():
        i = node_to_idx[edge[0]]
        j = node_to_idx[edge[1]]
        true_graph[i, j] = 1

    # Sample data from model
    data = model.simulate(n_samples=1000, seed=42)
    data = data[nodes]  # Reorder columns

    # Encode all categorical/string columns to numeric
    for col in data.columns:
        if data[col].dtype == 'object' or str(data[col].dtype) == 'category':
            # Use a separate LabelEncoder for each column
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))

    # Ensure all data is numeric and convert to float64
    data = data.apply(pd.to_numeric, errors='coerce')

    # Drop any rows that couldn't be converted (if any)
    data = data.dropna()

    # Final conversion to float64 for numerical stability
    data = data.astype(np.float64)

    return data, true_graph, nodes


def generate_synthetic_dag(n_nodes: int, edge_prob: float = 0.2, seed: int = 42):
    """Generate synthetic DAG with linear Gaussian data."""
    np.random.seed(seed)
    
    # Generate random DAG
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    
    # Add edges respecting topological order
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if np.random.rand() < edge_prob:
                G.add_edge(i, j)
    
    # Convert to adjacency matrix
    true_graph = nx.to_numpy_array(G, dtype=int)
    
    # Generate data using structural equation model
    n_samples = 1000
    data = np.zeros((n_samples, n_nodes))
    
    # Topological sort to generate data in causal order
    topo_order = list(nx.topological_sort(G))
    
    for node in topo_order:
        parents = list(G.predecessors(node))
        
        if len(parents) == 0:
            # Exogenous variable
            data[:, node] = np.random.randn(n_samples)
        else:
            # Linear combination of parents + noise
            weights = np.random.randn(len(parents))
            data[:, node] = data[:, parents] @ weights + np.random.randn(n_samples)
    
    df = pd.DataFrame(data, columns=[f"X{i}" for i in range(n_nodes)])
    
    return df, true_graph


# ============================================================================
# Experiment Runners
# ============================================================================

def run_titanic_experiments(analyzer: VarianceAnalyzer):
    """Run Titanic + LiNGAM experiments."""
    print("\n" + "="*80)
    print("TITANIC DATASET - LiNGAM")
    print("="*80)
    
    data, true_graph = load_titanic()
    print(f"Loaded Titanic: {data.shape[0]} samples, {data.shape[1]} variables")
    
    # Run LiNGAM with variance analysis
    results = analyzer.run_lingam_multiple(data, true_graph)
    
    # Print summary
    print("\n--- Algorithm Performance (Mean ± 95% CI) ---")
    print(f"Precision: {results.precision.mean:.4f} "
          f"[{results.precision.ci_lower:.4f}, {results.precision.ci_upper:.4f}]")
    print(f"Recall:    {results.recall.mean:.4f} "
          f"[{results.recall.ci_lower:.4f}, {results.recall.ci_upper:.4f}]")
    print(f"F1-score:  {results.f1.mean:.4f} "
          f"[{results.f1.ci_lower:.4f}, {results.f1.ci_upper:.4f}]")
    print(f"SHD:       {results.shd.mean:.1f} "
          f"[{results.shd.ci_lower:.1f}, {results.shd.ci_upper:.1f}]")
    
    # Compare with LLM estimates from paper
    llm_estimates_gpt5 = {
        'precision': (0.62, 0.76),
        'recall': (0.55, 0.70),
        'f1': (0.58, 0.73),
        'shd': (3, 7)
    }
    
    llm_estimates_deepseek = {
        'precision': (0.60, 0.70),
        'recall': (0.60, 0.70),
        'f1': (0.60, 0.70),
        'shd': (4, 6)
    }
    
    print("\n--- Comparison with GPT-5 Estimates ---")
    comparison_gpt5 = analyzer.compare_with_llm_estimates(results, llm_estimates_gpt5)
    for metric, comp in comparison_gpt5.items():
        overlap = "✓" if comp['overlaps'] else "✗"
        print(f"{metric:10s}: {overlap} Overlap: {comp['overlap_percentage']:.1f}%")
    
    print("\n--- Comparison with DeepSeek R1 Estimates ---")
    comparison_deepseek = analyzer.compare_with_llm_estimates(results, llm_estimates_deepseek)
    for metric, comp in comparison_deepseek.items():
        overlap = "✓" if comp['overlaps'] else "✗"
        print(f"{metric:10s}: {overlap} Overlap: {comp['overlap_percentage']:.1f}%")
    
    # Save results
    analyzer.save_results(results, 'titanic', 'lingam')
    
    # Save comparisons
    comparison_data = {
        'dataset': 'titanic',
        'algorithm': 'lingam',
        'gpt5_comparison': comparison_gpt5,
        'deepseek_comparison': comparison_deepseek
    }
    
    with open(analyzer.output_dir / 'titanic_lingam_llm_comparison.json', 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    return results


def run_benchmark_experiments(analyzer: VarianceAnalyzer):
    """Run experiments on bnlearn benchmark networks."""
    benchmarks = ['asia', 'cancer', 'earthquake', 'sachs', 'survey', 'child']
    
    all_results = {}
    
    for bench_name in benchmarks:
        print("\n" + "="*80)
        print(f"BENCHMARK: {bench_name.upper()}")
        print("="*80)
        
        try:
            data, true_graph, nodes = load_bnlearn_network(bench_name)
            print(f"Loaded {bench_name}: {len(nodes)} nodes, {data.shape[0]} samples")
            
            # Run PC algorithm (constraint-based)
            print("\n--- Running PC Algorithm ---")
            results_pc = analyzer.run_pc_multiple(data, true_graph)
            
            print(f"PC - Precision: {results_pc.precision.mean:.4f} "
                  f"[{results_pc.precision.ci_lower:.4f}, {results_pc.precision.ci_upper:.4f}]")
            print(f"PC - Recall:    {results_pc.recall.mean:.4f} "
                  f"[{results_pc.recall.ci_lower:.4f}, {results_pc.recall.ci_upper:.4f}]")
            print(f"PC - F1:        {results_pc.f1.mean:.4f} "
                  f"[{results_pc.f1.ci_lower:.4f}, {results_pc.f1.ci_upper:.4f}]")
            print(f"PC - SHD:       {results_pc.shd.mean:.1f} "
                  f"[{results_pc.shd.ci_lower:.1f}, {results_pc.shd.ci_upper:.1f}]")
            
            analyzer.save_results(results_pc, bench_name, 'pc')
            all_results[f"{bench_name}_pc"] = results_pc
            
            # Run LiNGAM (will likely fail on discrete data - showing assumption violations)
            print("\n--- Running LiNGAM Algorithm ---")
            results_lingam = analyzer.run_lingam_multiple(data, true_graph)
            
            print(f"LiNGAM - Precision: {results_lingam.precision.mean:.4f} "
                  f"[{results_lingam.precision.ci_lower:.4f}, {results_lingam.precision.ci_upper:.4f}]")
            print(f"LiNGAM - SHD:       {results_lingam.shd.mean:.1f} "
                  f"[{results_lingam.shd.ci_lower:.1f}, {results_lingam.shd.ci_upper:.1f}]")
            
            analyzer.save_results(results_lingam, bench_name, 'lingam')
            all_results[f"{bench_name}_lingam"] = results_lingam
            
        except Exception as e:
            print(f"Error processing {bench_name}: {e}")
            continue
    
    return all_results


def run_synthetic_experiments(analyzer: VarianceAnalyzer):
    """Run experiments on synthetic DAGs."""
    node_counts = [12, 30]
    
    all_results = {}
    
    for n_nodes in node_counts:
        print("\n" + "="*80)
        print(f"SYNTHETIC DAG - {n_nodes} NODES")
        print("="*80)
        
        data, true_graph = generate_synthetic_dag(n_nodes, edge_prob=0.2)
        print(f"Generated DAG: {n_nodes} nodes, {np.sum(true_graph)} edges")
        
        # Run PC
        print("\n--- Running PC Algorithm ---")
        results_pc = analyzer.run_pc_multiple(data, true_graph)
        
        print(f"PC - Precision: {results_pc.precision.mean:.4f} "
              f"[{results_pc.precision.ci_lower:.4f}, {results_pc.precision.ci_upper:.4f}]")
        print(f"PC - SHD:       {results_pc.shd.mean:.1f} "
              f"[{results_pc.shd.ci_lower:.1f}, {results_pc.shd.ci_upper:.1f}]")
        
        analyzer.save_results(results_pc, f'synthetic_{n_nodes}', 'pc')
        all_results[f"synthetic_{n_nodes}_pc"] = results_pc
        
        # Run LiNGAM
        print("\n--- Running LiNGAM Algorithm ---")
        results_lingam = analyzer.run_lingam_multiple(data, true_graph)
        
        print(f"LiNGAM - Precision: {results_lingam.precision.mean:.4f} "
              f"[{results_lingam.precision.ci_lower:.4f}, {results_lingam.precision.ci_upper:.4f}]")
        print(f"LiNGAM - SHD:       {results_lingam.shd.mean:.1f} "
              f"[{results_lingam.shd.ci_lower:.1f}, {results_lingam.shd.ci_upper:.1f}]")
        
        analyzer.save_results(results_lingam, f'synthetic_{n_nodes}', 'lingam')
        all_results[f"synthetic_{n_nodes}_lingam"] = results_lingam
    
    return all_results


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run all causal discovery experiments")
    parser.add_argument('--runs', type=int, default=100,
                       help='Number of runs per algorithm')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--experiments', nargs='+', 
                       choices=['titanic', 'benchmarks', 'synthetic', 'all'],
                       default=['all'],
                       help='Which experiments to run')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = VarianceAnalyzer(n_runs=args.runs, output_dir=args.output)
    
    print("="*80)
    print("CAUSAL DISCOVERY VARIANCE ANALYSIS")
    print("="*80)
    print(f"Runs per algorithm: {args.runs}")
    print(f"Output directory: {args.output}")
    print()
    
    results = {}
    
    if 'all' in args.experiments or 'titanic' in args.experiments:
        results['titanic'] = run_titanic_experiments(analyzer)
    
    if 'all' in args.experiments or 'benchmarks' in args.experiments:
        results['benchmarks'] = run_benchmark_experiments(analyzer)
    
    if 'all' in args.experiments or 'synthetic' in args.experiments:
        results['synthetic'] = run_synthetic_experiments(analyzer)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*80)
    print(f"Results saved to: {args.output}/")
    
    return results


if __name__ == "__main__":
    main()
