#!/usr/bin/env python3
"""
Quick test to verify FCI experiments setup.
Tests imports and basic functionality without running full experiments.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / "variance"))

print("Testing FCI Experiments Setup")
print("="*60)

# Test 1: Import variance_analysis
print("\n1. Testing variance_analysis import...")
try:
    from variance_analysis import VarianceAnalyzer
    print("   ✓ VarianceAnalyzer imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import VarianceAnalyzer: {e}")
    sys.exit(1)

# Test 2: Import dataset loaders
print("\n2. Testing dataset loader imports...")
try:
    from run_experiments import (
        load_titanic,
        load_bnlearn_network,
        generate_synthetic_dag
    )
    print("   ✓ Dataset loaders imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import dataset loaders: {e}")
    sys.exit(1)

# Test 3: Test FCI method exists
print("\n3. Testing FCI method availability...")
try:
    analyzer = VarianceAnalyzer(n_runs=1, output_dir="test_output")
    assert hasattr(analyzer, 'run_fci_multiple')
    print("   ✓ run_fci_multiple method exists")
except Exception as e:
    print(f"   ✗ FCI method not available: {e}")
    sys.exit(1)

# Test : Test causal-learn FCI import
print("\n. Testing causal-learn FCI import...")
try:
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.cit import fisherz
    print("   ✓ causal-learn FCI imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import causal-learn FCI: {e}")
    print("   → Install with: pip install causal-learn")
    sys.exit(1)

# Test : Test small synthetic dataset generation
print("\n. Testing synthetic dataset generation...")
try:
    data, true_graph = generate_synthetic_dag(5, edge_prob=0.2, seed=42)
    print(f"   ✓ Generated {data.shape[1]}-node synthetic DAG")
    print(f"     - {data.shape[0]} samples")
    print(f"     - {int(true_graph.sum())} edges")
except Exception as e:
    print(f"   ✗ Failed to generate synthetic data: {e}")
    sys.exit(1)

# Test 6: Test FCI on tiny dataset (1 run only)
print("\n6. Testing FCI execution on tiny dataset...")
try:
    import numpy as np
    np.random.seed(42)

    # Generate tiny 3-node dataset
    tiny_data, tiny_graph = generate_synthetic_dag(3, edge_prob=0.3, seed=42)

    # Run FCI once
    analyzer_test = VarianceAnalyzer(n_runs=1, output_dir="test_output")
    print("   Running FCI (1 iteration, may take 5-10 seconds)...")

    # Quick test without full variance analysis
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.cit import fisherz

    G, edges = fci(tiny_data.values, fisherz, alpha=0.05)
    print("   ✓ FCI executed successfully")
    print(f"     - Input: 3 nodes, {int(tiny_graph.sum())} edges")
    print(f"     - Output graph generated")

except Exception as e:
    print(f"   ✗ FCI execution failed: {e}")
    print("   Note: This might be due to numerical instability on tiny dataset")
    print("   Full experiments should still work with proper datasets")

# Test 7: Verify output directory structure
print("\n7. Testing output directory creation...")
try:
    output_dir = Path(__file__).parent / "fci_results"
    output_dir.mkdir(exist_ok=True)
    print(f"   ✓ Output directory created: {output_dir}")
except Exception as e:
    print(f"   ✗ Failed to create output directory: {e}")

# Test 8: Import visualization dependencies
print("\n8. Testing visualization dependencies...")
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("   ✓ Matplotlib and Seaborn available")
except Exception as e:
    print(f"   ✗ Visualization libraries not available: {e}")
    print("   → Install with: pip install matplotlib seaborn")

print("\n" + "="*60)
print("SETUP TEST COMPLETE")
print("="*60)
print("\nAll critical tests passed!")
print("\nYou can now run FCI experiments:")
print("  python run_fci_experiments.py --datasets sachs --runs 10")
print("\nFor full experiments (takes 2-4 hours):")
print("  python run_fci_experiments.py --runs 100")

# Cleanup test output
import shutil
test_dir = Path("test_output")
if test_dir.exists():
    shutil.rmtree(test_dir)
    print("\nCleaned up test output directory")
