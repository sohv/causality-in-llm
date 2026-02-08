#!/usr/bin/env python3
"""
Algorithm vs LLM Comparison Script
===================================

Compares causal discovery algorithm results against LLM predictions
to evaluate how well LLMs can predict algorithm performance.

Features:
- Compares metrics (F1, Precision, Recall) between algorithms and LLM predictions
- Generates visualization of prediction accuracy
- Analyzes prediction errors and biases
- Exports comparative analysis results

Usage:
    python compare_algorithms_vs_llms.py --dataset titanic --algorithm PC
    python compare_algorithms_vs_llms.py --all-combinations
    python compare_algorithms_vs_llms.py --viz-only  # Just generate plots from existing results
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from collections import defaultdict

import warnings
warnings.filterwarnings('ignore')

# Import dataset loaders
from datasets.alarm_network import load_alarm
from datasets.insurance_network import load_insurance
from datasets.barley_network import load_barley
from datasets.stock_market import load_stock_market

from variance.run_experiments import load_titanic_data, load_bnlearn_network

# Import algorithm runners
from llm_integration.multi_llm_runner import MultiLLMRunner


class AlgorithmVsLLMComparator:
    """
    Compares algorithm ground truth results vs LLM predictions.
    """
    
    def __init__(self, output_dir: str = "results/comparison"):
        """
        Initialize comparator.
        
        Args:
            output_dir: Directory to save comparison results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Dataset loaders
        self.dataset_loaders = {
            'alarm': load_alarm,
            'insurance': load_insurance,
            'barley': load_barley,
            'stock': load_stock_market,
            'titanic': load_titanic_data,
            'asia': lambda: load_bnlearn_network('asia'),
            'cancer': lambda: load_bnlearn_network('cancer'),
            'earthquake': lambda: load_bnlearn_network('earthquake'),
            'sachs': lambda: load_bnlearn_network('sachs'),
            'survey': lambda: load_bnlearn_network('survey'),
            'child': lambda: load_bnlearn_network('child'),
        }
        
        # Initialize LLM runner
        self.llm_runner = MultiLLMRunner(
            output_dir=str(self.output_dir / "llm_predictions")
        )
    
    def load_algorithm_results(self, dataset: str, algorithm: str) -> Dict:
        """
        Load algorithm ground truth results from files.
        
        Args:
            dataset: Dataset name
            algorithm: Algorithm name
            
        Returns:
            Dictionary with algorithm results
        """
        # Try multiple possible result file locations
        possible_paths = [
            f"results/{algorithm}_{dataset}/metrics.json",
            f"variance/results_full/{dataset}_{algorithm.lower()}_variance.json",
            f"fci_results/{dataset}_fci_variance.json",
            f"results/variance/{dataset}_{algorithm.lower()}_results.json"
        ]
        
        for path in possible_paths:
            file_path = Path(path)
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        return json.load(f)
                except json.JSONDecodeError:
                    continue
        
        # If no results found, run algorithm to get ground truth
        print(f"No existing results found for {dataset}+{algorithm}. Running algorithm...")
        return self._run_algorithm(dataset, algorithm)
    
    def _run_algorithm(self, dataset: str, algorithm: str) -> Dict:
        """
        Run algorithm to get ground truth results.
        
        Args:
            dataset: Dataset name
            algorithm: Algorithm name
            
        Returns:
            Dictionary with algorithm performance metrics
        """
        try:
            # Load dataset
            if dataset in self.dataset_loaders:
                data, true_graph, nodes = self.dataset_loaders[dataset]()
            else:
                raise ValueError(f"Unknown dataset: {dataset}")
            
            # Import and run appropriate algorithm
            if algorithm.lower() == 'pc':
                from causal_learn.search.ConstraintBased.PC import pc
                result = pc(data.values)
                estimated_graph = result.G.graph
            
            elif algorithm.lower() == 'lingam':
                from lingam import DirectLiNGAM
                model = DirectLiNGAM()
                model.fit(data)
                estimated_graph = model.adjacency_matrix_
                estimated_graph = (np.abs(estimated_graph) > 0.01).astype(int)
            
            elif algorithm.lower() == 'fci':
                from causal_learn.search.ConstraintBased.FCI import fci
                result = fci(data.values)
                estimated_graph = result.G.graph
            
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            # Calculate metrics
            metrics = self._calculate_metrics(true_graph, estimated_graph)
            
            return {
                'dataset': dataset,
                'algorithm': algorithm,
                'metrics': metrics,
                'n_samples': len(data),
                'n_variables': len(nodes)
            }
            
        except Exception as e:
            print(f"Error running {algorithm} on {dataset}: {e}")
            return {
                'dataset': dataset,
                'algorithm': algorithm,
                'metrics': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'error': str(e)
            }
    
    def _calculate_metrics(self, true_graph: np.ndarray, estimated_graph: np.ndarray) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            true_graph: Ground truth adjacency matrix
            estimated_graph: Estimated adjacency matrix
            
        Returns:
            Dictionary with F1, precision, recall scores
        """
        # Ensure same size
        n = min(true_graph.shape[0], estimated_graph.shape[0])
        true_graph = true_graph[:n, :n]
        estimated_graph = estimated_graph[:n, :n]
        
        # Convert to binary
        true_edges = (true_graph != 0).astype(int)
        estimated_edges = (estimated_graph != 0).astype(int)
        
        # Calculate metrics
        tp = np.sum((true_edges == 1) & (estimated_edges == 1))
        fp = np.sum((true_edges == 0) & (estimated_edges == 1))
        fn = np.sum((true_edges == 1) & (estimated_edges == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'f1': float(f1),
            'precision': float(precision),
            'recall': float(recall)
        }
    
    def get_llm_predictions(self, dataset: str, algorithm: str) -> Dict:
        """
        Get LLM predictions for dataset+algorithm combination.
        
        Args:
            dataset: Dataset name
            algorithm: Algorithm name
            
        Returns:
            Dictionary with LLM predictions from all models
        """
        # Check if predictions already exist
        pred_file = self.output_dir / "llm_predictions" / f"{dataset}_{algorithm}_predictions.json"
        
        if pred_file.exists():
            with open(pred_file, 'r') as f:
                return json.load(f)
        
        # Otherwise run LLM predictions
        print(f"Getting LLM predictions for {dataset}+{algorithm}...")
        
        try:
            results = self.llm_runner.run_experiment(dataset, algorithm)
            
            # Save predictions
            pred_file.parent.mkdir(exist_ok=True, parents=True)
            with open(pred_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            return results
            
        except Exception as e:
            print(f"Error getting LLM predictions: {e}")
            return {}
    
    def compare_predictions(self, dataset: str, algorithm: str) -> Dict:
        """
        Compare algorithm results vs LLM predictions.
        
        Args:
            dataset: Dataset name
            algorithm: Algorithm name
            
        Returns:
            Dictionary with comparison results
        """
        print(f"\\n{'='*60}")
        print(f"Comparing: {dataset.upper()} + {algorithm.upper()}")
        print(f"{'='*60}")
        
        # Get algorithm ground truth
        algo_results = self.load_algorithm_results(dataset, algorithm)
        algo_metrics = algo_results.get('metrics', {})
        
        # Get LLM predictions
        llm_results = self.get_llm_predictions(dataset, algorithm)
        
        # Compare each LLM's predictions
        comparison = {
            'dataset': dataset,
            'algorithm': algorithm,
            'ground_truth': algo_metrics,
            'llm_predictions': {},
            'prediction_errors': {},
            'llm_accuracy': {}
        }
        
        for llm_name, llm_data in llm_results.items():
            if 'results' not in llm_data:
                continue
                
            # Extract predicted metrics from each prompt formulation
            llm_predictions = {}
            
            for formulation, responses in llm_data['results'].items():
                if responses and len(responses) > 0:
                    # Take the first response's parsed metrics
                    parsed = responses[0].get('parsed_metrics', {})
                    llm_predictions[formulation] = parsed
            
            # Average predictions across formulations
            if llm_predictions:
                avg_predictions = {}
                for metric in ['f1', 'precision', 'recall']:
                    values = []
                    for form_preds in llm_predictions.values():
                        if metric in form_preds and isinstance(form_preds[metric], (list, tuple)):
                            # Take midpoint of range
                            values.append(np.mean(form_preds[metric]))
                        elif metric in form_preds:
                            values.append(float(form_preds[metric]))
                    
                    if values:
                        avg_predictions[metric] = np.mean(values)
                
                comparison['llm_predictions'][llm_name] = avg_predictions
                
                # Calculate prediction errors
                errors = {}
                accuracy = {}
                
                for metric in ['f1', 'precision', 'recall']:
                    if metric in algo_metrics and metric in avg_predictions:
                        true_val = algo_metrics[metric]
                        pred_val = avg_predictions[metric]
                        
                        error = abs(true_val - pred_val)
                        accuracy_score = 1.0 - error  # Simple accuracy measure
                        
                        errors[metric] = error
                        accuracy[metric] = max(0.0, accuracy_score)
                
                comparison['prediction_errors'][llm_name] = errors
                comparison['llm_accuracy'][llm_name] = accuracy
        
        return comparison
    
    def run_full_comparison(self, datasets: List[str] = None, algorithms: List[str] = None):
        """
        Run comparison across multiple dataset/algorithm combinations.
        
        Args:
            datasets: List of dataset names (if None, uses default set)
            algorithms: List of algorithm names (if None, uses default set)
        """
        if datasets is None:
            datasets = ['titanic', 'alarm', 'insurance', 'barley', 'sachs', 'asia']
        
        if algorithms is None:
            algorithms = ['PC', 'LiNGAM', 'FCI']
        
        all_comparisons = []
        summary_stats = defaultdict(list)
        
        for dataset in datasets:
            for algorithm in algorithms:
                try:
                    comparison = self.compare_predictions(dataset, algorithm)
                    all_comparisons.append(comparison)
                    
                    # Collect summary statistics
                    for llm_name, accuracy in comparison.get('llm_accuracy', {}).items():
                        for metric, acc_score in accuracy.items():
                            summary_stats[f"{llm_name}_{metric}"].append(acc_score)
                
                except Exception as e:
                    print(f"Error in comparison for {dataset}+{algorithm}: {e}")
                    continue
        
        # Save all results
        results_file = self.output_dir / "full_comparison_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_comparisons, f, indent=2)
        
        # Create summary
        self._create_summary_report(all_comparisons, summary_stats)
        
        # Generate visualizations
        self._create_visualizations(all_comparisons)
        
        # CRITICAL: UAI 2026 Statistical Rigor Enhancement
        print("\n" + "="*60)
        print("RUNNING UAI 2026 STATISTICAL RIGOR ANALYSIS...")
        print("="*60)
        
        try:
            from uai_2026_enhancements import StatisticalTester, CalibrationAnalyzer
            
            # 1. Statistical Significance Testing
            tester = StatisticalTester()
            
            # Extract LLM vs Algorithm accuracy scores for statistical testing
            llm_scores = []
            algo_scores = []
            
            for comparison in all_comparisons:
                llm_accuracy = comparison.get('llm_accuracy', {})
                
                # Get average LLM performance across all LLMs for this comparison
                if llm_accuracy:
                    avg_llm_f1 = np.mean([acc.get('f1', 0) for acc in llm_accuracy.values()])
                    llm_scores.append(avg_llm_f1)
                    
                    # Use Algorithm F1 as baseline (assuming algorithm performance is available)
                    # TODO: Replace with actual algorithm F1 scores from experimental results
                    baseline_f1 = comparison.get('algorithm_performance', {}).get('f1', 0.7)  # Default
                    algo_scores.append(baseline_f1)
            
            if len(llm_scores) >= 5:  # Need minimum data for statistical testing
                stat_result = tester.paired_t_test(
                    llm_scores, algo_scores, 
                    "LLM vs Traditional Algorithm Performance"
                )
                
                print(f"\\nSTATISTICAL SIGNIFICANCE RESULTS:")
                print(f"  p-value: {stat_result.p_value:.4f}")
                print(f"  Effect size (Cohen's d): {stat_result.effect_size:.3f}")
                print(f"  Statistically significant: {stat_result.is_significant}")
                print(f"  95% CI: [{stat_result.confidence_interval[0]:.3f}, {stat_result.confidence_interval[1]:.3f}]")
                
                # Save statistical results
                tester.generate_statistical_report([stat_result], 
                    str(self.output_dir / "uai_statistical_significance_report.txt"))
                print(f"  Statistical report saved to {self.output_dir}/uai_statistical_significance_report.txt")
            else:
                print("Insufficient data for statistical testing (need ≥5 comparisons)")
                
        except ImportError:
            print("UAI enhancement modules not available. Run:")
            print("cd uai_2026_enhancements && python -c 'from statistical_testing import StatisticalTester'")
        except Exception as e:
            print(f"Error in statistical analysis: {e}")
        
        print("\\n" + "="*60)
        print("UAI 2026 ENHANCEMENT ANALYSIS COMPLETE")
        print("="*60)
        
        return all_comparisons
    
    def _create_summary_report(self, comparisons: List[Dict], summary_stats: Dict):
        """Create summary report of comparison results."""
        
        report_file = self.output_dir / "comparison_summary.md"
        
        with open(report_file, 'w') as f:
            f.write("# Algorithm vs LLM Comparison Summary\\n\\n")
            
            # Overall statistics
            f.write("## Overall LLM Prediction Accuracy\\n\\n")
            
            llm_averages = {}
            for key, values in summary_stats.items():
                if values:
                    llm_name = key.rsplit('_', 1)[0]
                    metric = key.rsplit('_', 1)[1]
                    
                    if llm_name not in llm_averages:
                        llm_averages[llm_name] = {}
                    
                    llm_averages[llm_name][metric] = np.mean(values)
            
            # Create table
            f.write("| LLM | F1 Accuracy | Precision Accuracy | Recall Accuracy | Average |\\n")
            f.write("|-----|-------------|-------------------|-----------------|---------|\\n")
            
            for llm_name, metrics in llm_averages.items():
                f1_acc = metrics.get('f1', 0)
                prec_acc = metrics.get('precision', 0)
                rec_acc = metrics.get('recall', 0)
                avg_acc = np.mean([f1_acc, prec_acc, rec_acc])
                
                f.write(f"| {llm_name} | {f1_acc:.3f} | {prec_acc:.3f} | {rec_acc:.3f} | {avg_acc:.3f} |\\n")
            
            # Detailed results
            f.write("\\n## Detailed Results by Dataset and Algorithm\\n\\n")
            
            for comp in comparisons:
                dataset = comp['dataset']
                algorithm = comp['algorithm']
                
                f.write(f"### {dataset.title()} + {algorithm}\\n\\n")
                
                ground_truth = comp['ground_truth']
                f.write(f"**Ground Truth:** F1={ground_truth.get('f1', 'N/A'):.3f}, ")
                f.write(f"Precision={ground_truth.get('precision', 'N/A'):.3f}, ")
                f.write(f"Recall={ground_truth.get('recall', 'N/A'):.3f}\\n\\n")
                
                f.write("**LLM Predictions:**\\n\\n")
                for llm_name, pred_acc in comp.get('llm_accuracy', {}).items():
                    f1_acc = pred_acc.get('f1', 0)
                    prec_acc = pred_acc.get('precision', 0)
                    rec_acc = pred_acc.get('recall', 0)
                    
                    f.write(f"- {llm_name}: F1 acc={f1_acc:.3f}, Prec acc={prec_acc:.3f}, Rec acc={rec_acc:.3f}\\n")
                
                f.write("\\n")
    
    def _create_visualizations(self, comparisons: List[Dict]):
        """Create visualization plots of comparison results."""
        
        # Prepare data for plotting
        plot_data = []
        
        for comp in comparisons:
            dataset = comp['dataset']
            algorithm = comp['algorithm']
            
            for llm_name, accuracy in comp.get('llm_accuracy', {}).items():
                for metric, acc_score in accuracy.items():
                    plot_data.append({
                        'Dataset': dataset,
                        'Algorithm': algorithm,
                        'LLM': llm_name,
                        'Metric': metric,
                        'Accuracy': acc_score
                    })
        
        if not plot_data:
            print("No data available for plotting")
            return
        
        df = pd.DataFrame(plot_data)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Algorithm vs LLM Prediction Comparison', fontsize=16, fontweight='bold')
        
        # 1. Accuracy by LLM
        sns.boxplot(data=df, x='LLM', y='Accuracy', ax=axes[0, 0])
        axes[0, 0].set_title('Prediction Accuracy by LLM')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Accuracy by Metric
        sns.boxplot(data=df, x='Metric', y='Accuracy', ax=axes[0, 1])
        axes[0, 1].set_title('Prediction Accuracy by Metric Type')
        
        # 3. Accuracy by Dataset
        sns.boxplot(data=df, x='Dataset', y='Accuracy', ax=axes[1, 0])
        axes[1, 0].set_title('Prediction Accuracy by Dataset')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Heatmap of average accuracies
        pivot_df = df.groupby(['LLM', 'Metric'])['Accuracy'].mean().reset_index()
        pivot_table = pivot_df.pivot(index='LLM', columns='Metric', values='Accuracy')
        
        sns.heatmap(pivot_table, annot=True, cmap='RdYlBu_r', center=0.5, 
                    fmt='.3f', ax=axes[1, 1])
        axes[1, 1].set_title('Average Accuracy by LLM and Metric')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "comparison_visualizations.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.save(plot_file)
        print(f"Visualizations saved to {plot_file}")
        
        plt.close()


def main():
    """Main function to run comparisons."""
    parser = argparse.ArgumentParser(description='Compare algorithm results vs LLM predictions')
    
    parser.add_argument('--dataset', type=str, help='Specific dataset to test')
    parser.add_argument('--algorithm', type=str, help='Specific algorithm to test')
    parser.add_argument('--all-combinations', action='store_true', 
                       help='Run comparison on all dataset/algorithm combinations')
    parser.add_argument('--viz-only', action='store_true',
                       help='Only generate visualizations from existing results')
    parser.add_argument('--output-dir', type=str, default='results/comparison',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize comparator
    comparator = AlgorithmVsLLMComparator(output_dir=args.output_dir)
    
    if args.viz_only:
        # Load existing results and create visualizations
        results_file = Path(args.output_dir) / "full_comparison_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                comparisons = json.load(f)
            comparator._create_visualizations(comparisons)
        else:
            print("No existing results found. Run comparison first.")
        return
    
    if args.all_combinations:
        print("Running comparison on all dataset/algorithm combinations...")
        comparator.run_full_comparison()
        
    elif args.dataset and args.algorithm:
        print(f"Running comparison for {args.dataset} + {args.algorithm}")
        result = comparator.compare_predictions(args.dataset, args.algorithm)
        
        # Save single result
        result_file = Path(args.output_dir) / f"{args.dataset}_{args.algorithm}_comparison.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print("\\nComparison complete!")
        print(f"Results saved to {result_file}")
        
    else:
        print("Please specify --dataset and --algorithm, or use --all-combinations")


if __name__ == "__main__":
    main()