#!/usr/bin/env python3
"""
Visualization Script for FCI Experiment Results
------------------------------------------------
Creates publication-quality plots showing:
1. Performance metrics across datasets with confidence intervals
2. LLM prediction overlap analysis
3. Scaling behavior comparison
. Catastrophic collapse visualization for Synth30
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List

# Set style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# ============================================================================
# Data Loading
# ============================================================================

def load_fci_results(results_dir: Path) -> Dict:
    """Load all FCI experiment results from JSON files."""
    results = {}

    # Automatically discover all *_fci_variance.json files
    for result_file in results_dir.glob("*_fci_variance.json"):
        dataset = result_file.stem.replace('_fci_variance', '')
        with open(result_file, 'r') as f:
            results[dataset] = json.load(f)

    return results


def load_llm_comparisons(results_dir: Path) -> Dict:
    """Load LLM comparison results."""
    comparisons = {}

    # Automatically discover all *_fci_llm_comparison.json files
    for comp_file in results_dir.glob("*_fci_llm_comparison.json"):
        dataset = comp_file.stem.replace('_fci_llm_comparison', '')
        with open(comp_file, 'r') as f:
            comparisons[dataset] = json.load(f)

    return comparisons


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_performance_comparison(results: Dict, output_dir: Path):
    """Plot FCI performance metrics across all datasets."""

    datasets = list(results.keys())
    metrics = ['precision', 'recall', 'f1', 'shd']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        means = []
        ci_lowers = []
        ci_uppers = []
        labels = []

        for dataset in datasets:
            if metric in results[dataset]['results']:
                metric_data = results[dataset]['results'][metric]
                means.append(metric_data['mean'])
                ci_lowers.append(metric_data['ci_95_lower'])
                ci_uppers.append(metric_data['ci_95_upper'])
                labels.append(dataset.replace('_', ' ').title())

        x = np.arange(len(labels))
        yerr_lower = [means[i] - ci_lowers[i] for i in range(len(means))]
        yerr_upper = [ci_uppers[i] - means[i] for i in range(len(means))]

        # Color catastrophic collapse red
        colors = ['red' if dataset == 'synthetic_30' and metric in ['precision', 'recall', 'f1']
                 else 'steelblue' for dataset in datasets[:len(labels)]]

        ax.bar(x, means, color=colors, alpha=0.7, label='Mean')
        ax.errorbar(x, means, yerr=[yerr_lower, yerr_upper],
                   fmt='none', ecolor='black', capsize=5, capthick=2)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel(metric.upper() if metric != 'shd' else 'SHD')
        ax.set_title(f'FCI {metric.upper()} Performance')
        ax.grid(axis='y', alpha=0.3)

        # Add horizontal line at 0.2 for collapse threshold on F1
        if metric == 'f1':
            ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.5,
                      label='Catastrophic Collapse Threshold')
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'fci_performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f" Saved: {output_dir / 'fci_performance_comparison.png'}")
    plt.close()


def plot_scaling_analysis(results: Dict, output_dir: Path):
    """Plot how FCI performance scales with network size."""

    # Extract node counts and performance
    node_counts = []
    f1_means = []
    f1_cis = []
    dataset_names = []

    node_map = {
        'asia': 8,
        'sachs': 11,
        'earthquake': 5,
        'cancer': 5,
        'survey': 6,
        'synthetic_12': 12,
        'child': 20,
        'titanic': 7,
        'synthetic_30': 30
    }

    for dataset in results.keys():
        if dataset in node_map:
            node_counts.append(node_map[dataset])
            f1_data = results[dataset]['results']['f1']
            f1_means.append(f1_data['mean'])
            f1_cis.append((f1_data['ci_95_lower'], f1_data['ci_95_upper']))
            dataset_names.append(dataset.replace('_', ' ').title())

    # Sort by node count
    sorted_indices = np.argsort(node_counts)
    node_counts = [node_counts[i] for i in sorted_indices]
    f1_means = [f1_means[i] for i in sorted_indices]
    f1_cis = [f1_cis[i] for i in sorted_indices]
    dataset_names = [dataset_names[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot line with error bars
    yerr_lower = [f1_means[i] - f1_cis[i][0] for i in range(len(f1_means))]
    yerr_upper = [f1_cis[i][1] - f1_means[i] for i in range(len(f1_means))]
    ax.errorbar(node_counts, f1_means,
               yerr=[yerr_lower, yerr_upper],
               fmt='o-', capsize=5, capthick=2, markersize=8,
               linewidth=2, color='steelblue', label='FCI F1-score')

    # Annotate points
    for i, (x, y, name) in enumerate(zip(node_counts, f1_means, dataset_names)):
        ax.annotate(name, (x, y), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=9)

    # Highlight catastrophic collapse
    collapse_idx = dataset_names.index('Synthetic 30')
    ax.plot(node_counts[collapse_idx], f1_means[collapse_idx],
           'ro', markersize=12, label='Catastrophic Collapse')

    # Add collapse threshold line
    ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.5,
              label='Collapse Threshold')

    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('FCI Scaling Behavior: Performance vs Network Size', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fci_scaling_analysis.png', dpi=300, bbox_inches='tight')
    print(f" Saved: {output_dir / 'fci_scaling_analysis.png'}")
    plt.close()


def plot_llm_overlap_heatmap(comparisons: Dict, output_dir: Path):
    """Create heatmap showing LLM prediction overlap percentages."""

    datasets = list(comparisons.keys())
    metrics = ['precision', 'recall', 'f1', 'shd']

    # Build overlap matrix
    overlap_matrix = np.zeros((len(datasets), len(metrics)))

    for i, dataset in enumerate(datasets):
        if 'comparison' in comparisons[dataset]:
            for j, metric in enumerate(metrics):
                if metric in comparisons[dataset]['comparison']:
                    overlap_matrix[i, j] = comparisons[dataset]['comparison'][metric]['overlap_percentage']

    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(overlap_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    # Set ticks
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_yticklabels([d.replace('_', ' ').title() for d in datasets])

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(len(datasets)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{overlap_matrix[i, j]:.0f}%',
                         ha="center", va="center", color="black", fontsize=10)

    ax.set_title('LLM Prediction Overlap with FCI Results (%)', fontsize=14)
    fig.colorbar(im, ax=ax, label='Overlap %')

    plt.tight_layout()
    plt.savefig(output_dir / 'fci_llm_overlap_heatmap.png', dpi=300, bbox_inches='tight')
    print(f" Saved: {output_dir / 'fci_llm_overlap_heatmap.png'}")
    plt.close()


def plot_confidence_intervals(results: Dict, output_dir: Path):
    """Plot detailed confidence intervals for all metrics."""

    datasets = list(results.keys())
    metrics = ['precision', 'recall', 'f1']

    fig, ax = plt.subplots(figsize=(12, 8))

    y_positions = []
    y_labels = []
    y_pos = 0

    colors = plt.cm.Set3(np.linspace(0, 1, len(datasets)))

    for dataset_idx, dataset in enumerate(datasets):
        for metric_idx, metric in enumerate(metrics):
            metric_data = results[dataset]['results'][metric]

            # Plot CI as horizontal line
            ci_lower = metric_data['ci_95_lower']
            ci_upper = metric_data['ci_95_upper']
            mean = metric_data['mean']

            ax.plot([ci_lower, ci_upper], [y_pos, y_pos],
                   color=colors[dataset_idx], linewidth=6, alpha=0.6)
            ax.plot(mean, y_pos, 'o', color=colors[dataset_idx],
                   markersize=8, markeredgecolor='black', markeredgewidth=1)

            y_positions.append(y_pos)
            y_labels.append(f"{dataset.replace('_', ' ').title()}\n{metric.upper()}")

            y_pos += 1

        y_pos += 0.5  # Space between datasets

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel('Value', fontsize=12)
    ax.set_title('FCI Performance: 95% Confidence Intervals', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim([0, 1])

    # Add legend
    handles = [plt.Line2D([0], [0], color=colors[i], linewidth=6)
              for i in range(len(datasets))]
    labels = [d.replace('_', ' ').title() for d in datasets]
    ax.legend(handles, labels, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'fci_confidence_intervals.png', dpi=300, bbox_inches='tight')
    print(f" Saved: {output_dir / 'fci_confidence_intervals.png'}")
    plt.close()


def create_summary_table(results: Dict, output_dir: Path):
    """Create a summary table of all FCI results."""

    rows = []

    for dataset in results.keys():
        row = {
            'Dataset': dataset.replace('_', ' ').title(),
            'N_runs': results[dataset]['n_runs']
        }

        for metric in ['precision', 'recall', 'f1', 'shd']:
            metric_data = results[dataset]['results'][metric]
            row[f'{metric}_mean'] = f"{metric_data['mean']:.3f}"
            row[f'{metric}_ci'] = f"[{metric_data['ci_95_lower']:.3f}, {metric_data['ci_95_upper']:.3f}]"

        rows.append(row)

    df = pd.DataFrame(rows)

    # Save as CSV
    csv_file = output_dir / 'fci_summary_table.csv'
    df.to_csv(csv_file, index=False)
    print(f" Saved: {csv_file}")

    # Save as formatted text
    txt_file = output_dir / 'fci_summary_table.txt'
    with open(txt_file, 'w') as f:
        f.write("FCI EXPERIMENT RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(df.to_string(index=False))
    print(f" Saved: {txt_file}")

    return df


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize FCI experiment results")
    parser.add_argument('--results-dir', type=str, default='fci_results',
                       help='Directory containing FCI results')
    parser.add_argument('--output-dir', type=str, default='fci_plots',
                       help='Directory to save plots')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("="*80)
    print("FCI RESULTS VISUALIZATION")
    print("="*80)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Load results
    print("Loading results...")
    results = load_fci_results(results_dir)
    print(f"Loaded {len(results)} datasets: {', '.join(results.keys())}")

    # Load LLM comparisons if available
    comparisons = load_llm_comparisons(results_dir)
    if comparisons:
        print(f"Loaded {len(comparisons)} LLM comparisons")

    # Generate plots
    print("\nGenerating visualizations...")

    plot_performance_comparison(results, output_dir)
    plot_scaling_analysis(results, output_dir)
    plot_confidence_intervals(results, output_dir)

    if comparisons:
        plot_llm_overlap_heatmap(comparisons, output_dir)

    # Create summary table
    print("\nCreating summary table...")
    create_summary_table(results, output_dir)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"All plots saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - fci_performance_comparison.png")
    print("  - fci_scaling_analysis.png")
    print("  - fci_confidence_intervals.png")
    if comparisons:
        print("  - fci_llm_overlap_heatmap.png")
    print("  - fci_summary_table.csv")
    print("  - fci_summary_table.txt")


if __name__ == "__main__":
    main()
