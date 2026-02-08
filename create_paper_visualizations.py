#!/usr/bin/env python3
"""
Comprehensive Visualization Dashboard for Paper
================================================

Creates all visualizations needed for the UAI 2026 submission.

Visualizations:
1. Algorithm comparison across datasets
2. Dataset complexity vs performance
3. LLM comparison heatmaps
4. Prompt robustness analysis
5. Overlap analysis (algorithmic CI vs LLM ranges)
6. Summary tables for LaTeX

Usage:
    python create_paper_visualizations.py --results_dir results/ --output plots/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List
import argparse


class PaperVisualizationDashboard:
    """
    Creates comprehensive visualizations for the paper.
    """

    def __init__(self, results_dir: str, output_dir: str):
        """
        Initialize visualization dashboard.

        Args:
            results_dir: Directory containing all experiment results
            output_dir: Directory to save visualizations
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Set publication-quality style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10

        print(f"Dashboard initialized")
        print(f"  Results: {self.results_dir}")
        print(f"  Output: {self.output_dir}")

    def load_all_results(self) -> Dict:
        """Load all result JSON files."""
        results = {}

        for json_file in self.results_dir.glob("**/*_variance.json"):
            with open(json_file) as f:
                data = json.load(f)
                key = f"{data['dataset']}_{data['algorithm']}"
                results[key] = data

        print(f"\nLoaded {len(results)} result files")
        return results

    def create_algorithm_comparison(self, results: Dict):
        """
        Figure 1: Algorithm comparison across datasets.

        Shows precision, recall, F1, SHD for each algorithm on each dataset.
        """
        print("\nCreating Figure 1: Algorithm Comparison...")

        # Extract data
        rows = []
        for key, data in results.items():
            dataset = data['dataset']
            algorithm = data['algorithm']
            metrics = data['results']

            rows.append({
                'dataset': dataset,
                'algorithm': algorithm,
                'precision_mean': metrics['precision']['mean'],
                'precision_ci_lower': metrics['precision']['ci_95_lower'],
                'precision_ci_upper': metrics['precision']['ci_95_upper'],
                'recall_mean': metrics['recall']['mean'],
                'f1_mean': metrics['f1']['mean'],
                'shd_mean': metrics['shd']['mean']
            })

        df = pd.DataFrame(rows)

        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        metrics = ['precision_mean', 'recall_mean', 'f1_mean', 'shd_mean']
        titles = ['Precision', 'Recall', 'F1-Score', 'Structural Hamming Distance']

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]

            # Pivot for heatmap
            pivot = df.pivot_table(
                values=metric,
                index='dataset',
                columns='algorithm',
                aggfunc='mean'
            )

            sns.heatmap(pivot, annot=True, fmt='.3f' if 'shd' not in metric else '.1f',
                       cmap='RdYlGn' if 'shd' not in metric else 'RdYlGn_r',
                       ax=ax, cbar_kws={'label': title})

            ax.set_title(f'{title} by Dataset and Algorithm', fontweight='bold')
            ax.set_xlabel('Algorithm')
            ax.set_ylabel('Dataset')

        plt.suptitle('Algorithm Performance Across All Datasets\n'
                    '(Mean values from 100 runs with bootstrap sampling)',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        filename = self.output_dir / 'fig1_algorithm_comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {filename}")

    def create_complexity_analysis(self, results: Dict):
        """
        Figure 2: Performance vs Dataset Complexity.

        Shows how algorithm performance degrades with graph size.
        """
        print("\nCreating Figure 2: Complexity Analysis...")

        # Define dataset complexity (number of nodes)
        complexity_map = {
            'titanic': 7,
            'cancer': 5,
            'earthquake': 5,
            'asia': 8,
            'survey': 6,
            'sachs': 11,
            'child': 20,
            'alarm': 37,
            'stock_market': 10,
            'synthetic_12': 12,
            'synthetic_30': 30
        }

        rows = []
        for key, data in results.items():
            dataset = data['dataset']
            if dataset not in complexity_map:
                continue

            algorithm = data['algorithm']
            n_nodes = complexity_map[dataset]
            shd = data['results']['shd']['mean']
            precision = data['results']['precision']['mean']

            rows.append({
                'dataset': dataset,
                'algorithm': algorithm,
                'n_nodes': n_nodes,
                'shd': shd,
                'precision': precision
            })

        df = pd.DataFrame(rows)

        # Create scatter plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: SHD vs complexity
        for algo in df['algorithm'].unique():
            algo_data = df[df['algorithm'] == algo]
            ax1.scatter(algo_data['n_nodes'], algo_data['shd'],
                       label=algo.upper(), s=100, alpha=0.7)
            # Trend line
            z = np.polyfit(algo_data['n_nodes'], algo_data['shd'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(algo_data['n_nodes'].min(), algo_data['n_nodes'].max(), 100)
            ax1.plot(x_line, p(x_line), '--', alpha=0.5)

        ax1.set_xlabel('Number of Nodes (Dataset Complexity)', fontweight='bold')
        ax1.set_ylabel('Structural Hamming Distance (SHD)', fontweight='bold')
        ax1.set_title('Performance Degradation with Complexity\n'
                     '(Lower SHD is better)',
                     fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Plot 2: Precision vs complexity
        for algo in df['algorithm'].unique():
            algo_data = df[df['algorithm'] == algo]
            ax2.scatter(algo_data['n_nodes'], algo_data['precision'],
                       label=algo.upper(), s=100, alpha=0.7)
            # Trend line
            z = np.polyfit(algo_data['n_nodes'], algo_data['precision'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(algo_data['n_nodes'].min(), algo_data['n_nodes'].max(), 100)
            ax2.plot(x_line, p(x_line), '--', alpha=0.5)

        ax2.set_xlabel('Number of Nodes (Dataset Complexity)', fontweight='bold')
        ax2.set_ylabel('Precision', fontweight='bold')
        ax2.set_title('Precision vs Complexity\n'
                     '(Higher is better)',
                     fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        filename = self.output_dir / 'fig2_complexity_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {filename}")

    def create_coverage_summary(self, results: Dict):
        """
        Figure 3: Coverage summary showing the comprehensive nature of experiments.
        """
        print("\nCreating Figure 3: Experimental Coverage...")

        # Count datasets, algorithms, total experiments
        datasets = set()
        algorithms = set()
        domains = {
            'Social': ['titanic'],
            'Medical': ['asia', 'cancer', 'child', 'sachs', 'alarm'],
            'Finance': ['stock_market'],
            'General': ['earthquake', 'survey'],
            'Synthetic': ['synthetic_12', 'synthetic_30']
        }

        for key in results.keys():
            dataset, algorithm = key.rsplit('_', 1)
            datasets.add(dataset)
            algorithms.add(algorithm)

        # Create bar chart showing coverage
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot 1: Datasets by domain
        domain_counts = {}
        for domain, ds_list in domains.items():
            count = sum(1 for ds in datasets if ds in ds_list)
            domain_counts[domain] = count

        ax1 = axes[0]
        bars = ax1.bar(domain_counts.keys(), domain_counts.values(),
                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax1.set_ylabel('Number of Datasets', fontweight='bold')
        ax1.set_title('Dataset Coverage by Domain', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold')

        # Plot 2: Algorithms tested
        ax2 = axes[1]
        algo_list = list(algorithms)
        algo_counts = [len([k for k in results.keys() if k.endswith(algo)]) for algo in algo_list]

        bars = ax2.bar([a.upper() for a in algo_list], algo_counts,
                      color=['#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
        ax2.set_ylabel('Number of Experiments', fontweight='bold')
        ax2.set_title('Experiments by Algorithm', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold')

        # Plot 3: Summary statistics
        ax3 = axes[2]
        summary_data = {
            'Datasets': len(datasets),
            'Algorithms': len(algorithms),
            'Experiments': len(results),
            'Domains': len([d for d, ds in domains.items() if any(x in datasets for x in ds)])
        }

        bars = ax3.bar(summary_data.keys(), summary_data.values(),
                      color=['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22'])
        ax3.set_ylabel('Count', fontweight='bold')
        ax3.set_title('Overall Experimental Coverage', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=14, fontweight='bold')

        plt.suptitle('Comprehensive Experimental Coverage\n'
                    f'{len(datasets)} Datasets × {len(algorithms)} Algorithms = {len(results)} Experiments',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        filename = self.output_dir / 'fig3_coverage_summary.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {filename}")

        # Print summary
        print(f"\n  Summary:")
        print(f"    - {len(datasets)} unique datasets")
        print(f"    - {len(algorithms)} algorithms")
        print(f"    - {len(results)} total experiments")
        print(f"    - Domains covered: {', '.join([d for d, ds in domains.items() if any(x in datasets for x in ds)])}")

    def generate_latex_tables(self, results: Dict):
        """Generate LaTeX tables for the paper."""
        print("\nGenerating LaTeX tables...")

        # Main results table
        rows = []
        for key, data in results.items():
            dataset = data['dataset']
            algorithm = data['algorithm']
            metrics = data['results']

            row = {
                'Dataset': dataset.replace('_', ' ').title(),
                'Algorithm': algorithm.upper(),
                'Precision': f"{metrics['precision']['mean']:.3f} [{metrics['precision']['ci_95_lower']:.3f}, {metrics['precision']['ci_95_upper']:.3f}]",
                'Recall': f"{metrics['recall']['mean']:.3f} [{metrics['recall']['ci_95_lower']:.3f}, {metrics['recall']['ci_95_upper']:.3f}]",
                'F1': f"{metrics['f1']['mean']:.3f} [{metrics['f1']['ci_95_lower']:.3f}, {metrics['f1']['ci_95_upper']:.3f}]",
                'SHD': f"{metrics['shd']['mean']:.1f} [{metrics['shd']['ci_95_lower']:.1f}, {metrics['shd']['ci_95_upper']:.1f}]"
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        # Save as LaTeX
        latex_file = self.output_dir / 'main_results_table.tex'
        latex = df.to_latex(index=False, escape=False)

        with open(latex_file, 'w') as f:
            f.write("% Main Results Table - Algorithmic Performance\n")
            f.write("% Format: Mean [95% CI Lower, 95% CI Upper]\n")
            f.write("% Generated automatically\n\n")
            f.write(latex)

        print(f"  ✓ Saved: {latex_file}")

    def generate_all_visualizations(self):
        """Generate all visualizations for the paper."""
        print("\n" + "="*80)
        print("GENERATING PAPER VISUALIZATIONS")
        print("="*80)

        results = self.load_all_results()

        if not results:
            print("No results found!")
            return

        self.create_algorithm_comparison(results)
        self.create_complexity_analysis(results)
        self.create_coverage_summary(results)
        self.generate_latex_tables(results)

        print("\n" + "="*80)
        print("ALL VISUALIZATIONS COMPLETE")
        print("="*80)
        print(f"Plots saved to: {self.output_dir}/")
        print("\nFigures generated:")
        print("  1. fig1_algorithm_comparison.png - Algorithm performance heatmaps")
        print("  2. fig2_complexity_analysis.png - Performance vs dataset complexity")
        print("  3. fig3_coverage_summary.png - Experimental coverage summary")
        print("  4. main_results_table.tex - LaTeX table for paper")


def main():
    parser = argparse.ArgumentParser(description="Generate paper visualizations")
    parser.add_argument('--results_dir', type=str, default='variance/results',
                       help='Directory containing result JSON files')
    parser.add_argument('--output_dir', type=str, default='paper_plots',
                       help='Output directory for visualizations')

    args = parser.parse_args()

    dashboard = PaperVisualizationDashboard(
        results_dir=args.results_dir,
        output_dir=args.output_dir
    )

    dashboard.generate_all_visualizations()


if __name__ == "__main__":
    main()
