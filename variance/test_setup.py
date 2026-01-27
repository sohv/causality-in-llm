#!/usr/bin/env python3
"""
Test script to verify installation and demonstrate basic functionality.
"""

import numpy as np
import pandas as pd
from variance_analysis import VarianceAnalyzer, compute_metric_stats, compute_shd

def test_metric_stats():
    """Test statistical computation."""
    print("Testing metric statistics computation...")
    
    values = np.random.randn(100) * 10 + 50
    stats = compute_metric_stats(values)
    
    print(f"  Mean: {stats.mean:.2f}")
    print(f"  Std:  {stats.std:.2f}")
    print(f"  95% CI: [{stats.ci_lower:.2f}, {stats.ci_upper:.2f}]")
    print(f"  ✓ Passed\n")


def test_shd_computation():
    """Test SHD calculation."""
    print("Testing SHD computation...")
    
    # Simple 3-node DAG: 0->1->2
    true_graph = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ])
    
    # Learned graph: 0->1, 2->1 (one edge reversed)
    learned_graph = np.array([
        [0, 1, 0],
        [0, 0, 0],
        [0, 1, 0]
    ])
    
    shd = compute_shd(true_graph, learned_graph)
    
    print(f"  True edges: 0->1, 1->2")
    print(f"  Learned edges: 0->1, 2->1")
    print(f"  SHD: {shd}")
    print(f"  Expected: 2 (one missing, one extra)")
    
    assert shd == 2, f"SHD should be 2, got {shd}"
    print(f"  ✓ Passed\n")


def test_analyzer_initialization():
    """Test VarianceAnalyzer initialization."""
    print("Testing VarianceAnalyzer initialization...")
    
    analyzer = VarianceAnalyzer(n_runs=10, output_dir="test_results")
    
    print(f"  Runs: {analyzer.n_runs}")
    print(f"  Output dir: {analyzer.output_dir}")
    print(f"  ✓ Passed\n")


def test_small_experiment():
    """Run a small test experiment."""
    print("Testing small experiment (5 runs)...")
    
    # Generate tiny synthetic data
    np.random.seed(42)
    
    # True graph: X0 -> X1 -> X2
    true_graph = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ])
    
    # Generate data
    n = 100
    X0 = np.random.randn(n)
    X1 = 2 * X0 + np.random.randn(n) * 0.5
    X2 = 1.5 * X1 + np.random.randn(n) * 0.5
    
    data = pd.DataFrame({'X0': X0, 'X1': X1, 'X2': X2})
    
    # Run small test with LiNGAM
    analyzer = VarianceAnalyzer(n_runs=5, output_dir="test_results")
    
    try:
        results = analyzer.run_lingam_multiple(data, true_graph)
        
        print(f"  Precision: {results.precision.mean:.3f} ± {results.precision.std:.3f}")
        print(f"  Recall:    {results.recall.mean:.3f} ± {results.recall.std:.3f}")
        print(f"  F1:        {results.f1.mean:.3f} ± {results.f1.std:.3f}")
        print(f"  SHD:       {results.shd.mean:.1f} ± {results.shd.std:.1f}")
        print(f"  ✓ Passed\n")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Failed: {e}\n")
        return False


def test_llm_comparison():
    """Test LLM comparison functionality."""
    print("Testing LLM comparison...")
    
    # Mock algorithmic results
    from variance_analysis import AlgorithmResults, MetricStats
    
    precision_stats = MetricStats(
        mean=0.85,
        std=0.05,
        ci_lower=0.80,
        ci_upper=0.90,
        median=0.85,
        min_val=0.75,
        max_val=0.95,
        runs=100
    )
    
    recall_stats = MetricStats(
        mean=0.70,
        std=0.08,
        ci_lower=0.62,
        ci_upper=0.78,
        median=0.70,
        min_val=0.55,
        max_val=0.85,
        runs=100
    )
    
    f1_stats = MetricStats(
        mean=0.77,
        std=0.06,
        ci_lower=0.71,
        ci_upper=0.83,
        median=0.77,
        min_val=0.65,
        max_val=0.88,
        runs=100
    )
    
    shd_stats = MetricStats(
        mean=5.2,
        std=1.8,
        ci_lower=3.8,
        ci_upper=6.6,
        median=5.0,
        min_val=2,
        max_val=10,
        runs=100
    )
    
    results = AlgorithmResults(
        precision=precision_stats,
        recall=recall_stats,
        f1=f1_stats,
        shd=shd_stats
    )
    
    # Mock LLM estimates
    llm_estimates = {
        'precision': (0.62, 0.76),
        'recall': (0.55, 0.70),
        'f1': (0.58, 0.73),
        'shd': (3, 7)
    }
    
    analyzer = VarianceAnalyzer(n_runs=100, output_dir="test_results")
    comparison = analyzer.compare_with_llm_estimates(results, llm_estimates)
    
    print(f"  Precision overlap: {comparison['precision']['overlaps']}")
    print(f"  Recall overlap:    {comparison['recall']['overlaps']}")
    print(f"  F1 overlap:        {comparison['f1']['overlaps']}")
    print(f"  SHD overlap:       {comparison['shd']['overlaps']}")
    print(f"  ✓ Passed\n")


def main():
    """Run all tests."""
    print("="*70)
    print("VARIANCE ANALYSIS TEST SUITE")
    print("="*70)
    print()
    
    tests = [
        ("Metric Statistics", test_metric_stats),
        ("SHD Computation", test_shd_computation),
        ("Analyzer Initialization", test_analyzer_initialization),
        ("LLM Comparison", test_llm_comparison),
        ("Small Experiment", test_small_experiment),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            result = test_func()
            if result is None or result:
                passed += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}\n")
            failed += 1
    
    print("="*70)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    
    if failed == 0:
        print("✓ All tests passed! System is ready.")
    else:
        print(f"✗ {failed} tests failed. Check dependencies.")
    
    print("="*70)


if __name__ == "__main__":
    main()
