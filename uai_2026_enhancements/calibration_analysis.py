#!/usr/bin/env python3
"""
Calibration Analysis Module
============================

Analyzes how well LLM confidence intervals match actual accuracy.
Assesses whether LLMs "know what they don't know" - critical for trustworthy AI.

Features:
- Coverage probability analysis (% of truth values in predicted intervals)
- Interval width assessment (overconfident vs conservative)
- Calibration curve generation
- Reliability diagrams
- Calibration error metrics (ECE, MCE, ACE)
- Per-LLM calibration profiles

Usage:
    from uai_2026_enhancements.calibration_analysis import CalibrationAnalyzer
    
    analyzer = CalibrationAnalyzer()
    results = analyzer.analyze_llm_calibration(llm_predictions_with_intervals, ground_truth)
    analyzer.generate_calibration_report(results)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

@dataclass
class CalibrationMetrics:
    """Container for calibration analysis results."""
    llm_name: str
    dataset: str
    algorithm: str
    coverage_probability: float  # % of truth values within predicted intervals
    mean_interval_width: float  # Average width of confidence intervals
    expected_coverage: float  # What coverage should be (usually 0.95 for 95% CI)
    calibration_error: float  # Expected Calibration Error (ECE)
    max_calibration_error: float  # Maximum Calibration Error (MCE)
    over_confident_rate: float  # % of intervals too narrow
    under_confident_rate: float  # % of intervals too wide
    sharpness: float  # Average interval width (smaller = sharper)
    reliability: float  # How reliable are the confidence estimates
    n_predictions: int
    interval_coverage_by_bin: Dict[str, float]

class CalibrationAnalyzer:
    """
    Comprehensive calibration analysis for LLM confidence intervals.
    """
    
    def __init__(self, expected_coverage: float = 0.95, n_bins: int = 10):
        """
        Initialize calibration analyzer.
        
        Args:
            expected_coverage: Target coverage probability (e.g., 0.95 for 95% CI)
            n_bins: Number of bins for calibration curve
        """
        self.expected_coverage = expected_coverage
        self.n_bins = n_bins
    
    def analyze_single_llm_calibration(self,
                                     predicted_intervals: List[Tuple[float, float]],
                                     ground_truth_values: List[float],
                                     llm_name: str = "Unknown",
                                     dataset: str = "Unknown", 
                                     algorithm: str = "Unknown") -> CalibrationMetrics:
        """
        Analyze calibration for a single LLM-dataset-algorithm combination.
        
        Args:
            predicted_intervals: List of (lower_bound, upper_bound) tuples
            ground_truth_values: List of actual metric values
            llm_name: Name of the LLM
            dataset: Dataset name
            algorithm: Algorithm name
            
        Returns:
            CalibrationMetrics object with analysis results
        """
        if len(predicted_intervals) != len(ground_truth_values):
            raise ValueError("Predicted intervals and ground truth must have same length")
        
        n_predictions = len(predicted_intervals)
        
        # Convert to numpy arrays for easier manipulation
        intervals = np.array(predicted_intervals)
        truth = np.array(ground_truth_values)
        
        # Remove any NaN or invalid entries
        valid_mask = (
            ~np.isnan(intervals).any(axis=1) & 
            ~np.isnan(truth) &
            (intervals[:, 0] <= intervals[:, 1])  # Lower bound <= upper bound
        )
        
        intervals_clean = intervals[valid_mask]
        truth_clean = truth[valid_mask]
        n_valid = len(truth_clean)
        
        if n_valid == 0:
            raise ValueError("No valid prediction intervals found")
        
        # 1. Coverage probability
        within_interval = (
            (truth_clean >= intervals_clean[:, 0]) & 
            (truth_clean <= intervals_clean[:, 1])
        )
        coverage_probability = np.mean(within_interval)
        
        # 2. Interval widths
        interval_widths = intervals_clean[:, 1] - intervals_clean[:, 0]
        mean_interval_width = np.mean(interval_widths)
        
        # 3. Calibration error (Expected Calibration Error)
        calibration_error = self._compute_expected_calibration_error(
            predicted_intervals=intervals_clean,
            ground_truth=truth_clean
        )
        
        # 4. Maximum calibration error
        max_calibration_error = self._compute_max_calibration_error(
            predicted_intervals=intervals_clean,
            ground_truth=truth_clean
        )
        
        # 5. Over/under confidence rates
        coverage_gap = coverage_probability - self.expected_coverage
        over_confident_rate = max(0, -coverage_gap)  # Coverage < expected
        under_confident_rate = max(0, coverage_gap)   # Coverage > expected
        
        # 6. Sharpness (inverse of mean width - sharper is better)
        sharpness = 1.0 / (mean_interval_width + 1e-6)
        
        # 7. Reliability (how consistent is the coverage)
        reliability = 1.0 - abs(coverage_probability - self.expected_coverage)
        
        # 8. Coverage by confidence bins
        coverage_by_bin = self._compute_coverage_by_bin(
            intervals_clean, truth_clean, interval_widths
        )
        
        return CalibrationMetrics(
            llm_name=llm_name,
            dataset=dataset,
            algorithm=algorithm,
            coverage_probability=coverage_probability,
            mean_interval_width=mean_interval_width,
            expected_coverage=self.expected_coverage,
            calibration_error=calibration_error,
            max_calibration_error=max_calibration_error,
            over_confident_rate=over_confident_rate,
            under_confident_rate=under_confident_rate,
            sharpness=sharpness,
            reliability=reliability,
            n_predictions=n_valid,
            interval_coverage_by_bin=coverage_by_bin
        )
    
    def _compute_expected_calibration_error(self,
                                          predicted_intervals: np.ndarray,
                                          ground_truth: np.ndarray) -> float:
        """
        Compute Expected Calibration Error (ECE) for interval predictions.
        
        ECE measures the average difference between predicted and actual coverage
        across different confidence levels.
        
        Args:
            predicted_intervals: Array of shape (n, 2) with [lower, upper] bounds
            ground_truth: Array of actual values
            
        Returns:
            ECE value (0 = perfect calibration, 1 = worst calibration)
        """
        interval_widths = predicted_intervals[:, 1] - predicted_intervals[:, 0]
        
        # Bin intervals by width (proxy for confidence)
        bin_edges = np.quantile(interval_widths, np.linspace(0, 1, self.n_bins + 1))
        bin_edges = np.unique(bin_edges)  # Remove duplicates
        
        if len(bin_edges) < 2:
            return 0.0  # All intervals have same width
        
        total_error = 0.0
        total_weight = 0.0
        
        for i in range(len(bin_edges) - 1):
            # Find predictions in this bin
            in_bin = (interval_widths >= bin_edges[i]) & (interval_widths < bin_edges[i + 1])
            
            if i == len(bin_edges) - 2:  # Last bin includes upper edge
                in_bin = (interval_widths >= bin_edges[i]) & (interval_widths <= bin_edges[i + 1])
            
            if np.sum(in_bin) == 0:
                continue
            
            bin_intervals = predicted_intervals[in_bin]
            bin_truth = ground_truth[in_bin]
            
            # Compute actual coverage for this bin
            within_interval = (
                (bin_truth >= bin_intervals[:, 0]) & 
                (bin_truth <= bin_intervals[:, 1])
            )
            actual_coverage = np.mean(within_interval)
            
            # Weight by number of predictions in bin
            bin_weight = len(bin_truth)
            
            # Add to weighted error
            error = abs(actual_coverage - self.expected_coverage)
            total_error += bin_weight * error
            total_weight += bin_weight
        
        return total_error / (total_weight + 1e-10)
    
    def _compute_max_calibration_error(self,
                                     predicted_intervals: np.ndarray,
                                     ground_truth: np.ndarray) -> float:
        """
        Compute Maximum Calibration Error (MCE).
        
        MCE is the worst-case calibration error across all confidence bins.
        
        Args:
            predicted_intervals: Array of shape (n, 2) with [lower, upper] bounds  
            ground_truth: Array of actual values
            
        Returns:
            MCE value (maximum deviation from perfect calibration)
        """
        interval_widths = predicted_intervals[:, 1] - predicted_intervals[:, 0]
        
        # Bin intervals by width
        bin_edges = np.quantile(interval_widths, np.linspace(0, 1, self.n_bins + 1))
        bin_edges = np.unique(bin_edges)
        
        if len(bin_edges) < 2:
            return 0.0
        
        max_error = 0.0
        
        for i in range(len(bin_edges) - 1):
            in_bin = (interval_widths >= bin_edges[i]) & (interval_widths < bin_edges[i + 1])
            
            if i == len(bin_edges) - 2:
                in_bin = (interval_widths >= bin_edges[i]) & (interval_widths <= bin_edges[i + 1])
            
            if np.sum(in_bin) == 0:
                continue
            
            bin_intervals = predicted_intervals[in_bin]
            bin_truth = ground_truth[in_bin]
            
            within_interval = (
                (bin_truth >= bin_intervals[:, 0]) & 
                (bin_truth <= bin_intervals[:, 1])
            )
            actual_coverage = np.mean(within_interval)
            
            error = abs(actual_coverage - self.expected_coverage)
            max_error = max(max_error, error)
        
        return max_error
    
    def _compute_coverage_by_bin(self,
                               predicted_intervals: np.ndarray,
                               ground_truth: np.ndarray,
                               interval_widths: np.ndarray) -> Dict[str, float]:
        """
        Compute coverage probability for different interval width bins.
        
        Args:
            predicted_intervals: Array of interval bounds
            ground_truth: Array of actual values
            interval_widths: Array of interval widths
            
        Returns:
            Dictionary mapping bin description to coverage probability
        """
        # Define bins by width percentiles
        percentiles = [0, 25, 50, 75, 100]
        bin_edges = np.percentile(interval_widths, percentiles)
        bin_names = ['Narrowest', 'Narrow', 'Wide', 'Widest']
        
        coverage_by_bin = {}
        
        for i in range(len(percentiles) - 1):
            if i == 0:
                in_bin = interval_widths <= bin_edges[i + 1]
            elif i == len(percentiles) - 2:
                in_bin = interval_widths >= bin_edges[i]
            else:
                in_bin = (interval_widths > bin_edges[i]) & (interval_widths <= bin_edges[i + 1])
            
            if np.sum(in_bin) == 0:
                coverage_by_bin[bin_names[i]] = np.nan
                continue
            
            bin_intervals = predicted_intervals[in_bin]
            bin_truth = ground_truth[in_bin]
            
            within_interval = (
                (bin_truth >= bin_intervals[:, 0]) & 
                (bin_truth <= bin_intervals[:, 1])
            )
            coverage_by_bin[bin_names[i]] = np.mean(within_interval)
        
        return coverage_by_bin
    
    def analyze_comprehensive_calibration(self, 
                                        llm_prediction_data: Dict[str, Dict[str, Dict[str, Dict]]]) -> Dict[str, List[CalibrationMetrics]]:
        """
        Analyze calibration across all LLM-dataset-algorithm combinations.
        
        Args:
            llm_prediction_data: Nested dict {LLM: {dataset: {algorithm: {intervals, truth}}}}
                                 where intervals is list of (low, high) tuples
                                 and truth is list of actual values
            
        Returns:
            Dictionary mapping LLM names to lists of CalibrationMetrics
        """
        all_results = {}
        
        for llm_name, datasets in llm_prediction_data.items():
            all_results[llm_name] = []
            
            for dataset_name, algorithms in datasets.items():
                for algorithm_name, data in algorithms.items():
                    try:
                        # Extract intervals and ground truth
                        intervals = data.get('predicted_intervals', [])
                        truth = data.get('ground_truth', [])
                        
                        if not intervals or not truth:
                            print(f"Warning: No data for {llm_name}-{dataset_name}-{algorithm_name}")
                            continue
                        
                        # Analyze calibration
                        metrics = self.analyze_single_llm_calibration(
                            predicted_intervals=intervals,
                            ground_truth_values=truth,
                            llm_name=llm_name,
                            dataset=dataset_name,
                            algorithm=algorithm_name
                        )
                        
                        all_results[llm_name].append(metrics)
                        
                    except Exception as e:
                        print(f"Error analyzing {llm_name}-{dataset_name}-{algorithm_name}: {e}")
                        continue
        
        return all_results
    
    def create_calibration_plots(self, 
                               calibration_results: Dict[str, List[CalibrationMetrics]],
                               output_dir: Optional[str] = None):
        """
        Generate comprehensive calibration visualization plots.
        
        Args:
            calibration_results: Results from analyze_comprehensive_calibration
            output_dir: Directory to save plots (if None, displays only)
        """
        # Flatten results for plotting
        all_metrics = []
        for llm_name, metrics_list in calibration_results.items():
            all_metrics.extend(metrics_list)
        
        if not all_metrics:
            print("No calibration metrics to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('LLM Calibration Analysis: Confidence Interval Quality', fontsize=16, fontweight='bold')
        
        # 1. Coverage probability by LLM
        coverage_data = {llm: [m.coverage_probability for m in metrics] 
                        for llm, metrics in calibration_results.items()}
        
        llm_names = list(coverage_data.keys())
        coverage_means = [np.mean(coverage_data[llm]) for llm in llm_names]
        coverage_stds = [np.std(coverage_data[llm]) for llm in llm_names]
        
        bars = axes[0, 0].bar(llm_names, coverage_means, yerr=coverage_stds, 
                              alpha=0.7, capsize=5, color='skyblue')
        axes[0, 0].axhline(self.expected_coverage, color='red', linestyle='--', 
                          label=f'Target ({self.expected_coverage:.0%})')
        axes[0, 0].set_ylabel('Coverage Probability')
        axes[0, 0].set_title('Coverage Probability by LLM')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add percentage labels on bars
        for bar, mean in zip(bars, coverage_means):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{mean:.1%}', ha='center', va='bottom', fontsize=10)
        
        # 2. Calibration error by LLM
        error_data = {llm: [m.calibration_error for m in metrics] 
                     for llm, metrics in calibration_results.items()}
        
        error_means = [np.mean(error_data[llm]) for llm in llm_names]
        error_stds = [np.std(error_data[llm]) for llm in llm_names]
        
        axes[0, 1].bar(llm_names, error_means, yerr=error_stds, 
                      alpha=0.7, capsize=5, color='lightcoral')
        axes[0, 1].set_ylabel('Expected Calibration Error')
        axes[0, 1].set_title('Calibration Error by LLM (Lower = Better)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Reliability diagram (coverage vs expected)
        coverage_values = [m.coverage_probability for m in all_metrics]
        axes[0, 2].scatter(coverage_values, [self.expected_coverage] * len(coverage_values), 
                          alpha=0.6, color='purple')
        
        # Perfect calibration line
        min_val, max_val = min(coverage_values), max(coverage_values)
        perfect_line = np.linspace(min_val, max_val, 100)
        axes[0, 2].plot(perfect_line, perfect_line, 'r--', label='Perfect Calibration')
        
        axes[0, 2].axhline(self.expected_coverage, color='orange', linestyle=':', 
                          alpha=0.7, label=f'Target Coverage')
        axes[0, 2].set_xlabel('Actual Coverage')
        axes[0, 2].set_ylabel('Target Coverage')
        axes[0, 2].set_title('Reliability Diagram')
        axes[0, 2].legend()
        
        # 4. Interval width distribution
        all_widths = [m.mean_interval_width for m in all_metrics]
        axes[1, 0].hist(all_widths, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 0].set_xlabel('Mean Interval Width')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Interval Widths')
        
        # 5. Sharpness vs Reliability scatter
        sharpness_values = [m.sharpness for m in all_metrics]
        reliability_values = [m.reliability for m in all_metrics]
        llm_colors = {llm: plt.cm.tab10(i) for i, llm in enumerate(llm_names)}
        
        for metric in all_metrics:
            axes[1, 1].scatter(metric.sharpness, metric.reliability, 
                             c=[llm_colors[metric.llm_name]], 
                             alpha=0.6, s=50, label=metric.llm_name)
        
        # Remove duplicate legend entries
        handles, labels = axes[1, 1].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axes[1, 1].legend(by_label.values(), by_label.keys())
        
        axes[1, 1].set_xlabel('Sharpness (Higher = Better)')
        axes[1, 1].set_ylabel('Reliability (Higher = Better)')
        axes[1, 1].set_title('Sharpness vs Reliability Trade-off')
        
        # 6. Over/under confidence rates
        over_conf_data = {llm: [m.over_confident_rate for m in metrics] 
                         for llm, metrics in calibration_results.items()}
        under_conf_data = {llm: [m.under_confident_rate for m in metrics] 
                          for llm, metrics in calibration_results.items()}
        
        over_means = [np.mean(over_conf_data[llm]) for llm in llm_names]
        under_means = [np.mean(under_conf_data[llm]) for llm in llm_names]
        
        x_pos = np.arange(len(llm_names))
        width = 0.35
        
        axes[1, 2].bar(x_pos - width/2, over_means, width, label='Over-confident', 
                      alpha=0.7, color='red')
        axes[1, 2].bar(x_pos + width/2, under_means, width, label='Under-confident', 
                      alpha=0.7, color='blue')
        
        axes[1, 2].set_xlabel('LLM')
        axes[1, 2].set_ylabel('Rate')
        axes[1, 2].set_title('Over/Under-Confidence Rates')
        axes[1, 2].set_xticks(x_pos)
        axes[1, 2].set_xticklabels(llm_names, rotation=45)
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        if output_dir:
            output_path = Path(output_dir) / "calibration_analysis_plots.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Calibration plots saved to {output_path}")
        
        plt.show()
    
    def create_calibration_summary_table(self, 
                                       calibration_results: Dict[str, List[CalibrationMetrics]]) -> pd.DataFrame:
        """
        Create publication-ready calibration summary table.
        
        Args:
            calibration_results: Results from analyze_comprehensive_calibration
            
        Returns:
            DataFrame with calibration metrics by LLM
        """
        summary_rows = []
        
        for llm_name, metrics_list in calibration_results.items():
            if not metrics_list:
                continue
            
            # Aggregate metrics across all dataset-algorithm combinations
            coverage_probs = [m.coverage_probability for m in metrics_list]
            cal_errors = [m.calibration_error for m in metrics_list]
            interval_widths = [m.mean_interval_width for m in metrics_list]
            reliabilities = [m.reliability for m in metrics_list]
            sharpness_values = [m.sharpness for m in metrics_list]
            
            summary_rows.append({
                'LLM': llm_name,
                'Mean_Coverage': f"{np.mean(coverage_probs):.3f}",
                'Coverage_Std': f"{np.std(coverage_probs):.3f}",
                'Target_Coverage': f"{self.expected_coverage:.3f}",
                'Coverage_Gap': f"{np.mean(coverage_probs) - self.expected_coverage:+.3f}",
                'Mean_Cal_Error': f"{np.mean(cal_errors):.3f}",
                'Mean_Interval_Width': f"{np.mean(interval_widths):.3f}",
                'Mean_Reliability': f"{np.mean(reliabilities):.3f}",
                'Mean_Sharpness': f"{np.mean(sharpness_values):.2f}",
                'N_Comparisons': len(metrics_list),
                'Well_Calibrated': "✓" if abs(np.mean(coverage_probs) - self.expected_coverage) < 0.05 else "✗"
            })
        
        df = pd.DataFrame(summary_rows)
        
        # Sort by calibration quality (smaller calibration error is better)
        df['_cal_error_numeric'] = df['Mean_Cal_Error'].astype(float)
        df = df.sort_values('_cal_error_numeric').drop('_cal_error_numeric', axis=1)
        
        return df
    
    def generate_calibration_report(self, 
                                  calibration_results: Dict[str, List[CalibrationMetrics]],
                                  output_file: Optional[str] = None):
        """
        Generate comprehensive calibration analysis report.
        
        Args:
            calibration_results: Results from analyze_comprehensive_calibration
            output_file: File path for report (if None, prints to console)
        """
        # Flatten all metrics for overall analysis
        all_metrics = []
        for metrics_list in calibration_results.values():
            all_metrics.extend(metrics_list)
        
        if not all_metrics:
            print("No calibration metrics available for report")
            return
        
        # Compute overall statistics
        all_coverages = [m.coverage_probability for m in all_metrics]
        all_errors = [m.calibration_error for m in all_metrics]
        all_widths = [m.mean_interval_width for m in all_metrics]
        
        well_calibrated = sum(1 for c in all_coverages if abs(c - self.expected_coverage) < 0.05)
        
        report_lines = [
            "=" * 100,
            "LLM CALIBRATION ANALYSIS REPORT",
            "=" * 100,
            "",
            f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Target Coverage: {self.expected_coverage:.0%}",
            f"Calibration Tolerance: ±5%",
            "",
            "OVERALL CALIBRATION SUMMARY",
            "-" * 50,
            f"Total Comparisons: {len(all_metrics)}",
            f"Mean Coverage Probability: {np.mean(all_coverages):.3f} (±{np.std(all_coverages):.3f})",
            f"Coverage Gap from Target: {np.mean(all_coverages) - self.expected_coverage:+.3f}",
            f"Well-Calibrated Predictions: {well_calibrated}/{len(all_metrics)} ({well_calibrated/len(all_metrics):.1%})",
            "",
            "CALIBRATION QUALITY METRICS",
            "-" * 50,
            f"Mean Calibration Error (ECE): {np.mean(all_errors):.3f}",
            f"Std Calibration Error: {np.std(all_errors):.3f}",
            f"Best Calibrated (Lowest ECE): {np.min(all_errors):.3f}",
            f"Worst Calibrated (Highest ECE): {np.max(all_errors):.3f}",
            "",
            "INTERVAL CHARACTERISTICS",
            "-" * 50,
            f"Mean Interval Width: {np.mean(all_widths):.3f} (±{np.std(all_widths):.3f})",
            f"Narrowest Mean Width: {np.min(all_widths):.3f}",
            f"Widest Mean Width: {np.max(all_widths):.3f}",
            "",
            "CALIBRATION BY LLM",
            "-" * 50,
        ]
        
        # Add per-LLM analysis
        for llm_name, metrics_list in calibration_results.items():
            if not metrics_list:
                continue
            
            coverages = [m.coverage_probability for m in metrics_list]
            errors = [m.calibration_error for m in metrics_list]
            
            mean_coverage = np.mean(coverages)
            mean_error = np.mean(errors)
            coverage_gap = mean_coverage - self.expected_coverage
            
            # Calibration quality assessment
            if abs(coverage_gap) < 0.02:
                quality = "Excellent"
            elif abs(coverage_gap) < 0.05:
                quality = "Good" 
            elif abs(coverage_gap) < 0.10:
                quality = "Fair"
            else:
                quality = "Poor"
            
            # Over/under confidence
            if coverage_gap < -0.02:
                confidence_type = "Over-confident (intervals too narrow)"
            elif coverage_gap > 0.02:
                confidence_type = "Under-confident (intervals too wide)"
            else:
                confidence_type = "Well-calibrated"
            
            report_lines.extend([
                f"",
                f"{llm_name.upper()}:",
                f"  • Coverage: {mean_coverage:.3f} (gap: {coverage_gap:+.3f})",
                f"  • Calibration Error: {mean_error:.3f}",
                f"  • Quality: {quality}",
                f"  • Assessment: {confidence_type}",
                f"  • Comparisons: {len(metrics_list)}"
            ])
        
        report_lines.extend([
            "",
            "INTERPRETATION GUIDE",
            "-" * 50,
            "Coverage Probability:",
            "  • Should equal target (0.95 for 95% confidence intervals)",
            "  • > target = Under-confident (intervals too wide)",
            "  • < target = Over-confident (intervals too narrow)",
            "",
            "Calibration Error (ECE):",
            "  • 0.00 = Perfect calibration",
            "  • < 0.05 = Well calibrated",
            "  • 0.05-0.10 = Moderately calibrated",
            "  • > 0.10 = Poorly calibrated",
            "",
            "RECOMMENDATIONS",
            "-" * 50,
        ])
        
        # Add recommendations based on results
        best_llm = min(calibration_results.keys(), 
                      key=lambda x: np.mean([m.calibration_error for m in calibration_results[x]]))
        worst_llm = max(calibration_results.keys(),
                       key=lambda x: np.mean([m.calibration_error for m in calibration_results[x]]))
        
        report_lines.extend([
            f"• Best Calibrated LLM: {best_llm}",
            f"• Most Reliable for Uncertainty: {best_llm}",
            f"• Needs Calibration Improvement: {worst_llm}",
            "",
            "For practical use:",
            "• Trust narrow intervals from well-calibrated LLMs",
            "• Be cautious with over-confident models",  
            "• Consider ensemble methods for better calibration",
            "",
            "=" * 100
        ])
        
        report_text = "\\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Calibration report saved to {output_file}")
        else:
            print(report_text)
        
        return report_text


def example_usage():
    """
    Example showing how to use CalibrationAnalyzer.
    """
    # Initialize analyzer
    analyzer = CalibrationAnalyzer(expected_coverage=0.95, n_bins=10)
    
    # Simulate LLM prediction data with confidence intervals
    np.random.seed(42)
    
    llm_prediction_data = {}
    llms = ['GPT', 'Claude', 'Gemini', 'DeepSeek']
    datasets = ['titanic', 'sachs', 'alarm']
    algorithms = ['PC', 'LiNGAM', 'FCI']
    
    for llm in llms:
        llm_prediction_data[llm] = {}
        for dataset in datasets:
            llm_prediction_data[llm][dataset] = {}
            for algorithm in algorithms:
                n_predictions = 30
                
                # Ground truth algorithm performance
                ground_truth = np.random.beta(6, 4, n_predictions)
                
                # LLM predictions with different calibration quality per LLM
                if llm == 'GPT':
                    # Well-calibrated: intervals contain truth ~95% of time
                    interval_width = np.random.uniform(0.15, 0.25, n_predictions)
                    lower_bounds = ground_truth - interval_width * 0.5
                    upper_bounds = ground_truth + interval_width * 0.5
                    
                elif llm == 'Claude':
                    # Under-confident: intervals too wide
                    interval_width = np.random.uniform(0.25, 0.40, n_predictions)
                    lower_bounds = ground_truth - interval_width * 0.5
                    upper_bounds = ground_truth + interval_width * 0.5
                    
                elif llm == 'Gemini':
                    # Over-confident: intervals too narrow
                    interval_width = np.random.uniform(0.05, 0.12, n_predictions)
                    lower_bounds = ground_truth - interval_width * 0.5
                    upper_bounds = ground_truth + interval_width * 0.5
                    
                else:  # DeepSeek
                    # Poorly calibrated: biased predictions
                    bias = np.random.normal(0.05, 0.02, n_predictions)
                    interval_width = np.random.uniform(0.15, 0.25, n_predictions)
                    lower_bounds = ground_truth + bias - interval_width * 0.5
                    upper_bounds = ground_truth + bias + interval_width * 0.5
                
                # Ensure bounds are valid
                lower_bounds = np.clip(lower_bounds, 0, 1)
                upper_bounds = np.clip(upper_bounds, 0, 1)
                upper_bounds = np.maximum(upper_bounds, lower_bounds + 0.01)
                
                predicted_intervals = list(zip(lower_bounds, upper_bounds))
                
                llm_prediction_data[llm][dataset][algorithm] = {
                    'predicted_intervals': predicted_intervals,
                    'ground_truth': ground_truth.tolist()
                }
    
    # Run comprehensive calibration analysis
    print("Running comprehensive calibration analysis...")
    results = analyzer.analyze_comprehensive_calibration(llm_prediction_data)
    
    # Generate calibration report
    analyzer.generate_calibration_report(results)
    
    # Create summary table
    summary_table = analyzer.create_calibration_summary_table(results)
    print("\\n" + "="*100)
    print("CALIBRATION SUMMARY TABLE")
    print("="*100)
    print(summary_table.to_string(index=False))
    
    # Generate plots
    analyzer.create_calibration_plots(results)
    
    return results, summary_table


if __name__ == "__main__":
    results, table = example_usage()