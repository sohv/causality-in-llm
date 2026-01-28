#!/usr/bin/env python3
"""
Create side-by-side comparison plots of PC vs LiNGAM performance.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 11

def load_results(results_dir: Path):
    """Load all JSON results."""
    results = {}
    for json_file in results_dir.glob("*_variance.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            key = json_file.stem.replace('_variance', '')
            results[key] = data
    return results


def plot_algorithm_comparison(results, output_dir: Path):
    """Create grouped bar chart comparing PC vs LiNGAM across all datasets."""

    # Prepare data
    datasets = ['synthetic_12', 'synthetic_30', 'asia', 'cancer',
                'earthquake', 'sachs', 'survey', 'child']

    metrics = ['precision', 'recall', 'f1']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]

        pc_means = []
        pc_errs = []
        lingam_means = []
        lingam_errs = []
        labels = []

        for dataset in datasets:
            pc_key = f"{dataset}_pc"
            lingam_key = f"{dataset}_lingam"

            if pc_key in results and lingam_key in results:
                # PC data
                pc_data = results[pc_key]['results'][metric]
                pc_means.append(pc_data['mean'])
                pc_errs.append(pc_data['ci_95_upper'] - pc_data['mean'])

                # LiNGAM data
                lingam_data = results[lingam_key]['results'][metric]
                lingam_means.append(lingam_data['mean'])
                lingam_errs.append(lingam_data['ci_95_upper'] - lingam_data['mean'])

                labels.append(dataset.replace('_', ' ').title())

        x = np.arange(len(labels))
        width = 0.35

        # Plot bars
        ax.bar(x - width/2, pc_means, width, yerr=pc_errs,
               label='PC', capsize=5, alpha=0.8, color='#2E86AB')
        ax.bar(x + width/2, lingam_means, width, yerr=lingam_errs,
               label='LiNGAM', capsize=5, alpha=0.8, color='#A23B72')

        ax.set_xlabel('Dataset', fontweight='bold')
        ax.set_ylabel(metric.upper(), fontweight='bold')
        ax.set_title(f'{metric.upper()} Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_dir / 'pc_vs_lingam_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved PC vs LiNGAM comparison plot")


def plot_performance_by_dataset(results, output_dir: Path):
    """Create individual plots for each dataset showing PC vs LiNGAM."""

    datasets = ['synthetic_12', 'synthetic_30', 'asia', 'cancer',
                'earthquake', 'sachs', 'survey', 'child']

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]

        pc_key = f"{dataset}_pc"
        lingam_key = f"{dataset}_lingam"

        if pc_key not in results or lingam_key not in results:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(dataset.replace('_', ' ').title())
            continue

        # Extract data
        metrics = ['precision', 'recall', 'f1']
        pc_values = []
        pc_errors = []
        lingam_values = []
        lingam_errors = []

        for metric in metrics:
            pc_data = results[pc_key]['results'][metric]
            pc_values.append(pc_data['mean'])
            pc_errors.append(pc_data['ci_95_upper'] - pc_data['mean'])

            lingam_data = results[lingam_key]['results'][metric]
            lingam_values.append(lingam_data['mean'])
            lingam_errors.append(lingam_data['ci_95_upper'] - lingam_data['mean'])

        x = np.arange(len(metrics))
        width = 0.35

        ax.bar(x - width/2, pc_values, width, yerr=pc_errors,
               label='PC', capsize=5, alpha=0.8, color='#2E86AB')
        ax.bar(x + width/2, lingam_values, width, yerr=lingam_errors,
               label='LiNGAM', capsize=5, alpha=0.8, color='#A23B72')

        ax.set_title(dataset.replace('_', ' ').title(), fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_performance_grid.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved dataset performance grid")


def main():
    results_dir = Path('results_full')
    output_dir = Path('results_full')

    print("Loading results...")
    results = load_results(results_dir)
    print(f"Loaded {len(results)} experiments")

    print("\nGenerating comparison visualizations...")
    plot_algorithm_comparison(results, output_dir)
    plot_performance_by_dataset(results, output_dir)

    print(f"\nDone! Saved to {output_dir}/")


if __name__ == "__main__":
    main()
