#!/usr/bin/env python3
"""
Multi-LLM Experiment Runner
============================

Orchestrates experiments across multiple LLMs and prompt formulations.

Features:
- Queries 6 LLMs: Claude 3.5, Gemini 1.5, Qwen 2.5, Llama 3.3, GPT-5, DeepSeek R1
- Tests 3 prompt formulations for each
- Generates comparison visualizations
- Exports results for paper

Usage:
    python multi_llm_runner.py --dataset titanic --algorithm PC --output results/
    python multi_llm_runner.py --dataset titanic --algorithm PC --all-combos
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import argparse
from typing import Dict, List
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from llm_integration.claude_api import ClaudeClient
from llm_integration.gemini_api import GeminiClient
from llm_integration.qwen_api import QwenClient
from llm_integration.llama_api import LlamaClient
from llm_integration.gpt_api import GPTClient
from llm_integration.deepseek_api import DeepSeekClient
from prompt_variations.prompt_templates import get_all_formulations, generate_prompt
from prompt_variations.analyze_prompt_variance import (
    compare_prompt_formulations,
    visualize_prompt_variance
)


class MultiLLMRunner:
    """
    Runner for querying multiple LLMs across multiple prompt formulations.
    """

    def __init__(self, output_dir: str = "results/llm_experiments"):
        """
        Initialize multi-LLM runner.

        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Initialize clients
        self.clients = {}

        # Try to initialize Claude
        try:
            self.clients['claude'] = ClaudeClient()
            print("✓ Claude client initialized")
        except ValueError as e:
            print(f"✗ Claude client failed: {e}")

        # Try to initialize Gemini
        try:
            self.clients['gemini'] = GeminiClient()
            print("✓ Gemini client initialized")
        except ValueError as e:
            print(f"✗ Gemini client failed: {e}")

        # Try to initialize Qwen
        try:
            self.clients['qwen'] = QwenClient()
            print("✓ Qwen client initialized")
        except ValueError as e:
            print(f"✗ Qwen client failed: {e}")

        # Try to initialize Llama
        try:
            self.clients['llama'] = LlamaClient()
            print("✓ Llama client initialized")
        except ValueError as e:
            print(f"✗ Llama client failed: {e}")

        # Try to initialize GPT
        try:
            self.clients['gpt'] = GPTClient()
            print("✓ GPT client initialized")
        except ValueError as e:
            print(f"✗ GPT client failed: {e}")

        # Try to initialize DeepSeek
        try:
            self.clients['deepseek'] = DeepSeekClient()
            print("✓ DeepSeek client initialized")
        except ValueError as e:
            print(f"✗ DeepSeek client failed: {e}")

        if not self.clients:
            raise RuntimeError("No LLM clients initialized. Set API keys.")

    def run_experiment(self,
                      dataset_name: str,
                      algorithm_name: str,
                      n_samples: int = 1000) -> Dict:
        """
        Run experiments across all LLMs and all prompt formulations.

        Args:
            dataset_name: Name of dataset
            algorithm_name: Name of algorithm
            n_samples: Number of samples in dataset

        Returns:
            Dictionary with all results
        """
        print(f"\n{'='*80}")
        print(f"Running Multi-LLM Experiment: {dataset_name} + {algorithm_name}")
        print(f"{'='*80}")

        all_results = {}
        formulations = get_all_formulations()

        for llm_name, client in self.clients.items():
            print(f"\n--- {llm_name.upper()} ---")
            llm_results = {}

            for formulation in formulations:
                print(f"  Testing {formulation.name}...")

                # Generate prompt
                prompt = generate_prompt(
                    dataset_name=dataset_name,
                    algorithm_name=algorithm_name,
                    formulation=formulation,
                    n_samples=n_samples
                )

                try:
                    # Query LLM
                    parsed = client.query_and_parse(prompt, temperature=0.7)

                    llm_results[f'formulation_{formulation.formulation_id}'] = parsed

                    print(f"    ✓ Got results: {list(parsed.keys())}")

                except Exception as e:
                    print(f"    ✗ Error: {e}")
                    continue

            all_results[llm_name] = llm_results

        # Save raw results
        output_file = self.output_dir / f"{dataset_name}_{algorithm_name}_llm_results.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n✓ Results saved to {output_file}")

        return all_results

    def generate_comparison_plots(self, all_results: Dict, dataset_name: str, algorithm_name: str):
        """
        Generate comparison visualizations.

        Args:
            all_results: Results from run_experiment()
            dataset_name: Name of dataset
            algorithm_name: Name of algorithm
        """
        print(f"\nGenerating comparison plots...")

        # 1. Prompt variance analysis
        variance_df = compare_prompt_formulations(all_results, dataset_name, algorithm_name)

        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        visualize_prompt_variance(variance_df, plots_dir)

        # 2. Cross-LLM comparison (averaged across prompts)
        self._plot_cross_llm_comparison(all_results, plots_dir, dataset_name, algorithm_name)

        print(f"✓ Plots saved to {plots_dir}/")

    def _plot_cross_llm_comparison(self, all_results: Dict, output_dir: Path, dataset: str, algorithm: str):
        """Generate cross-LLM comparison plots."""

        # Extract midpoints for each LLM (averaged across formulations)
        llm_midpoints = {}
        metrics = ['precision', 'recall', 'f1', 'shd']

        for llm_name, formulations in all_results.items():
            midpoints = {m: [] for m in metrics}

            for form_id, results in formulations.items():
                for metric in metrics:
                    if metric in results:
                        lower, upper = results[metric]
                        midpoints[metric].append((lower + upper) / 2)

            # Average across formulations
            llm_midpoints[llm_name] = {
                metric: np.mean(vals) if vals else 0
                for metric, vals in midpoints.items()
            }

        # Create comparison bar plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            llms = list(llm_midpoints.keys())
            values = [llm_midpoints[llm][metric] for llm in llms]

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            bars = ax.bar(llms, values, alpha=0.7, color=colors[:len(llms)])

            # Add value labels
            for i, (llm, val) in enumerate(zip(llms, values)):
                ax.text(i, val + 0.01 * max(values), f'{val:.3f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

            ax.set_ylabel(metric.upper(), fontsize=12)
            ax.set_title(f'{metric.upper()} Estimates\n(Averaged across prompt formulations)',
                        fontsize=12, fontweight='bold')
            ax.set_ylim(0, max(values) * 1.15)
            ax.grid(axis='y', alpha=0.3)

        plt.suptitle(f'Cross-LLM Comparison: {dataset.title()} + {algorithm}\n'
                    f'(Each bar = average of 3 prompt formulations)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        filename = output_dir / f"{dataset}_{algorithm}_cross_llm_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  - {filename.name}")


def run_multi_llm_experiments(datasets: List[str],
                              algorithms: List[str],
                              output_dir: str = "results/llm_experiments"):
    """
    Run experiments across multiple datasets and algorithms.

    Args:
        datasets: List of dataset names
        algorithms: List of algorithm names
        output_dir: Directory to save results
    """
    runner = MultiLLMRunner(output_dir=output_dir)

    all_experiments = {}

    for dataset in datasets:
        for algorithm in algorithms:
            key = f"{dataset}_{algorithm}"
            print(f"\n\n{'#'*80}")
            print(f"# Experiment: {dataset} + {algorithm}")
            print(f"{'#'*80}")

            results = runner.run_experiment(dataset, algorithm)
            runner.generate_comparison_plots(results, dataset, algorithm)

            all_experiments[key] = results

    # Generate overall summary
    summary_file = Path(output_dir) / "all_experiments_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_experiments, f, indent=2)

    print(f"\n\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"{'='*80}")
    print(f"Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Run multi-LLM experiments")
    parser.add_argument('--dataset', type=str, default='titanic',
                       help='Dataset name (or "all" for all datasets)')
    parser.add_argument('--algorithm', type=str, default='PC',
                       help='Algorithm name (or "all" for all algorithms)')
    parser.add_argument('--output', type=str, default='results/llm_experiments',
                       help='Output directory')
    parser.add_argument('--all-combos', action='store_true',
                       help='Run ALL dataset x algorithm combinations')

    args = parser.parse_args()

    ALL_DATASETS = [
        'titanic', 'sachs', 'alarm', 'stock_market', 'insurance', 'barley',
        'asia', 'cancer', 'earthquake', 'survey', 'child',
        'synthetic_12', 'synthetic_30'
    ]
    ALL_ALGORITHMS = ['PC', 'LiNGAM', 'FCI', 'NOTEARS', 'GES', 'GRaSP']

    if args.all_combos:
        run_multi_llm_experiments(ALL_DATASETS, ALL_ALGORITHMS, args.output)
    elif args.dataset == 'all' and args.algorithm == 'all':
        run_multi_llm_experiments(ALL_DATASETS, ALL_ALGORITHMS, args.output)
    elif args.dataset == 'all':
        run_multi_llm_experiments(ALL_DATASETS, [args.algorithm], args.output)
    elif args.algorithm == 'all':
        run_multi_llm_experiments([args.dataset], ALL_ALGORITHMS, args.output)
    else:
        runner = MultiLLMRunner(output_dir=args.output)
        results = runner.run_experiment(
            dataset_name=args.dataset,
            algorithm_name=args.algorithm
        )
        runner.generate_comparison_plots(results, args.dataset, args.algorithm)

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Results: {args.output}/")


if __name__ == "__main__":
    main()
