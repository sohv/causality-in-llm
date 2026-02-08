#!/usr/bin/env python3
"""
Statistical Significance Testing Module
=======================================

Robust statistical testing for causal discovery LLM evaluation.
Implements formal hypothesis testing with multiple comparison correction.

Features:
- Paired t-tests (LLM predictions vs algorithm ground truth)
- Bootstrap confidence intervals 
- Multiple comparison correction (Bonferroni, FDR)
- Effect size computation (Cohen's d)
- Power analysis
- Comprehensive reporting

Usage:
    from uai_2026_enhancements.statistical_testing import StatisticalTester
    
    tester = StatisticalTester()
    results = tester.run_full_analysis(llm_predictions, ground_truth_results)
    tester.generate_report(results)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import bootstrap
import warnings
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings('ignore', category=RuntimeWarning)

@dataclass
class TestResult:
    """Container for statistical test results."""
    test_name: str
    t_statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    cohens_d: float
    effect_size_interpretation: str
    power: float
    significant: bool
    n_samples: int
    mean_difference: float
    std_error: float

class StatisticalTester:
    """
    Comprehensive statistical testing for LLM vs Algorithm comparisons.
    """
    
    def __init__(self, alpha: float = 0.05, power_threshold: float = 0.8):
        """
        Initialize statistical tester.
        
        Args:
            alpha: Significance level (default: 0.05)
            power_threshold: Minimum statistical power (default: 0.8)
        """
        self.alpha = alpha
        self.power_threshold = power_threshold
        self.results_cache = {}
    
    def paired_t_test(self, 
                     llm_predictions: np.ndarray, 
                     algorithm_truth: np.ndarray,
                     test_name: str = "Unnamed Test") -> TestResult:
        """
        Perform paired t-test with comprehensive statistics.
        
        Tests: H0: mean(LLM_pred - algo_truth) = 0
               Ha: mean(LLM_pred - algo_truth) ≠ 0
        
        Args:
            llm_predictions: Array of LLM-predicted metric values
            algorithm_truth: Array of corresponding algorithm ground truth
            test_name: Descriptive name for this test
            
        Returns:
            TestResult with all statistics
        """
        if len(llm_predictions) != len(algorithm_truth):
            raise ValueError("Prediction and truth arrays must have same length")
        
        # Remove any NaN pairs
        valid_mask = ~(np.isnan(llm_predictions) | np.isnan(algorithm_truth))
        llm_clean = llm_predictions[valid_mask]
        algo_clean = algorithm_truth[valid_mask]
        
        if len(llm_clean) < 3:
            raise ValueError(f"Insufficient valid data points: {len(llm_clean)}")
        
        # Compute differences
        differences = llm_clean - algo_clean
        n = len(differences)
        
        # Paired t-test
        t_stat, p_value = stats.ttest_1samp(differences, 0)
        
        # Cohen's d effect size (for one-sample: mean / std)
        cohens_d = np.mean(differences) / (np.std(differences, ddof=1) + 1e-10)
        
        # Effect size interpretation
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            effect_interpretation = "negligible"
        elif abs_d < 0.5:
            effect_interpretation = "small"
        elif abs_d < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        # Confidence interval for mean difference (bootstrap)
        ci_lower, ci_upper = self._bootstrap_ci(differences)
        
        # Power analysis (approximate)
        power = self._compute_power(cohens_d, n, self.alpha)
        
        # Standard error
        std_error = np.std(differences, ddof=1) / np.sqrt(n)
        
        return TestResult(
            test_name=test_name,
            t_statistic=t_stat,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            cohens_d=cohens_d,
            effect_size_interpretation=effect_interpretation,
            power=power,
            significant=p_value < self.alpha,
            n_samples=n,
            mean_difference=np.mean(differences),
            std_error=std_error
        )
    
    def _bootstrap_ci(self, 
                     data: np.ndarray, 
                     confidence: float = 0.95,
                     n_bootstrap: int = 10000) -> Tuple[float, float]:
        """
        Compute bootstrap confidence interval for mean.
        
        Args:
            data: Sample data
            confidence: Confidence level
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            (lower_bound, upper_bound)
        """
        rng = np.random.default_rng(42)
        
        def mean_statistic(x, axis=0):
            return np.mean(x, axis=axis)
        
        try:
            res = bootstrap(
                (data,), 
                mean_statistic, 
                n_resamples=n_bootstrap,
                confidence_level=confidence,
                random_state=rng,
                method='percentile'
            )
            return res.confidence_interval.low, res.confidence_interval.high
        except Exception:
            # Fallback to manual percentile method
            bootstrap_means = []
            for _ in range(n_bootstrap):
                bootstrap_sample = rng.choice(data, size=len(data), replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))
            
            alpha_level = 1 - confidence
            lower_p = (alpha_level / 2) * 100
            upper_p = (1 - alpha_level / 2) * 100
            
            return np.percentile(bootstrap_means, lower_p), np.percentile(bootstrap_means, upper_p)
    
    def _compute_power(self, effect_size: float, sample_size: int, alpha: float) -> float:
        """
        Compute statistical power for t-test (approximate).
        
        Args:
            effect_size: Cohen's d
            sample_size: Sample size
            alpha: Significance level
            
        Returns:
            Statistical power (0-1)
        """
        # Critical t-value
        df = sample_size - 1
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        # Non-centrality parameter
        ncp = abs(effect_size) * np.sqrt(sample_size)
        
        # Power computation using non-central t-distribution
        power = 1 - stats.nct.cdf(t_critical, df, ncp) + stats.nct.cdf(-t_critical, df, ncp)
        
        return max(0.0, min(1.0, power))
    
    def multiple_comparison_correction(self, 
                                     p_values: List[float],
                                     method: str = 'bonferroni') -> Tuple[List[float], List[bool]]:
        """
        Apply multiple comparison correction.
        
        Args:
            p_values: List of uncorrected p-values
            method: 'bonferroni', 'holm', or 'fdr_bh'
            
        Returns:
            (corrected_p_values, is_significant_list)
        """
        n_tests = len(p_values)
        p_array = np.array(p_values)
        
        if method == 'bonferroni':
            corrected_p = np.minimum(1.0, p_array * n_tests)
            
        elif method == 'holm':
            # Holm-Bonferroni step-down procedure
            sorted_indices = np.argsort(p_array)
            sorted_p = p_array[sorted_indices]
            
            corrected_p = np.zeros(n_tests)
            for i, idx in enumerate(sorted_indices):
                corrected_p[idx] = min(1.0, sorted_p[i] * (n_tests - i))
        
        elif method == 'fdr_bh':
            # Benjamini-Hochberg FDR control
            sorted_indices = np.argsort(p_array)
            sorted_p = p_array[sorted_indices]
            
            # Compute adjusted p-values
            ranks = np.arange(1, n_tests + 1)
            corrected_sorted = np.minimum(1.0, sorted_p * n_tests / ranks)
            
            # Ensure monotonicity (reverse cumulative minimum)
            for i in range(n_tests - 2, -1, -1):
                corrected_sorted[i] = min(corrected_sorted[i], corrected_sorted[i + 1])
            
            corrected_p = np.zeros(n_tests)
            corrected_p[sorted_indices] = corrected_sorted
            
        else:
            raise ValueError(f"Unknown correction method: {method}")
        
        is_significant = corrected_p < self.alpha
        
        return corrected_p.tolist(), is_significant.tolist()
    
    def run_comprehensive_analysis(self, 
                                 comparison_data: Dict[str, Dict[str, Dict[str, Dict]]],
                                 correction_method: str = 'fdr_bh') -> Dict:
        """
        Run comprehensive statistical analysis across all comparisons.
        
        Args:
            comparison_data: Nested dict {LLM: {dataset: {algorithm: {metrics}}}}
            correction_method: Multiple comparison correction method
            
        Returns:
            Dictionary with all statistical results
        """
        all_tests = []
        test_results = {}
        
        # Run all individual tests
        for llm_name, datasets in comparison_data.items():
            test_results[llm_name] = {}
            
            for dataset_name, algorithms in datasets.items():
                test_results[llm_name][dataset_name] = {}
                
                for algorithm_name, data in algorithms.items():
                    test_name = f"{llm_name}_{dataset_name}_{algorithm_name}"
                    
                    try:
                        # Extract prediction and truth arrays
                        llm_preds = np.array(data.get('llm_predictions', []))
                        algo_truth = np.array(data.get('algorithm_truth', []))
                        
                        # Run paired t-test
                        result = self.paired_t_test(llm_preds, algo_truth, test_name)
                        
                        test_results[llm_name][dataset_name][algorithm_name] = result
                        all_tests.append(result)
                        
                    except Exception as e:
                        print(f"Warning: Failed test for {test_name}: {e}")
                        continue
        
        # Apply multiple comparison correction
        if len(all_tests) > 1:
            p_values = [test.p_value for test in all_tests]
            corrected_p, is_sig_corrected = self.multiple_comparison_correction(
                p_values, method=correction_method
            )
            
            # Update significance flags
            for i, test in enumerate(all_tests):
                test.corrected_p_value = corrected_p[i]
                test.significant_corrected = is_sig_corrected[i]
        
        # Compute summary statistics
        summary = self._compute_summary_statistics(all_tests)
        
        return {
            'individual_tests': test_results,
            'all_tests': all_tests,
            'summary': summary,
            'correction_method': correction_method,
            'n_total_tests': len(all_tests)
        }
    
    def _compute_summary_statistics(self, test_results: List[TestResult]) -> Dict:
        """
        Compute summary statistics across all tests.
        
        Args:
            test_results: List of TestResult objects
            
        Returns:
            Summary statistics dictionary
        """
        if not test_results:
            return {}
        
        p_values = [t.p_value for t in test_results]
        effect_sizes = [t.cohens_d for t in test_results]
        powers = [t.power for t in test_results]
        
        return {
            'total_tests': len(test_results),
            'significant_uncorrected': sum(1 for t in test_results if t.significant),
            'significant_corrected': sum(1 for t in test_results if hasattr(t, 'significant_corrected') and t.significant_corrected),
            'mean_effect_size': np.mean(effect_sizes),
            'median_effect_size': np.median(effect_sizes),
            'large_effects': sum(1 for d in effect_sizes if abs(d) > 0.8),
            'mean_power': np.mean(powers),
            'underpowered_tests': sum(1 for p in powers if p < 0.8),
            'median_p_value': np.median(p_values),
            'min_p_value': np.min(p_values),
            'effect_size_distribution': {
                'negligible': sum(1 for d in effect_sizes if abs(d) < 0.2),
                'small': sum(1 for d in effect_sizes if 0.2 <= abs(d) < 0.5),
                'medium': sum(1 for d in effect_sizes if 0.5 <= abs(d) < 0.8),
                'large': sum(1 for d in effect_sizes if abs(d) >= 0.8)
            }
        }
    
    def create_summary_table(self, analysis_results: Dict) -> pd.DataFrame:
        """
        Create publication-ready summary table.
        
        Args:
            analysis_results: Results from run_comprehensive_analysis
            
        Returns:
            DataFrame with formatted results
        """
        rows = []
        
        all_tests = analysis_results['all_tests']
        
        for test in all_tests:
            # Parse test name
            parts = test.test_name.split('_')
            if len(parts) >= 3:
                llm, dataset, algorithm = parts[0], parts[1], parts[2]
            else:
                llm, dataset, algorithm = "Unknown", "Unknown", "Unknown"
            
            # Format confidence interval
            ci_lower, ci_upper = test.confidence_interval
            ci_str = f"[{ci_lower:.3f}, {ci_upper:.3f}]"
            
            # Significance stars
            p_corrected = getattr(test, 'corrected_p_value', test.p_value)
            if p_corrected < 0.001:
                sig_star = "***"
            elif p_corrected < 0.01:
                sig_star = "**" 
            elif p_corrected < 0.05:
                sig_star = "*"
            else:
                sig_star = ""
            
            rows.append({
                'LLM': llm,
                'Dataset': dataset,
                'Algorithm': algorithm,
                'Mean_Diff': f"{test.mean_difference:+.3f}",
                'SE': f"{test.std_error:.3f}",
                't_stat': f"{test.t_statistic:.2f}",
                'p_value': f"{test.p_value:.3f}",
                'p_corrected': f"{p_corrected:.3f}" if hasattr(test, 'corrected_p_value') else "—",
                'Cohen_d': f"{test.cohens_d:+.3f}",
                'Effect_Size': test.effect_size_interpretation,
                'CI_95': ci_str,
                'Power': f"{test.power:.2f}",
                'Sig': sig_star,
                'N': test.n_samples
            })
        
        df = pd.DataFrame(rows)
        return df.sort_values(['p_corrected' if 'p_corrected' in df.columns else 'p_value'])
    
    def generate_statistical_plots(self, analysis_results: Dict, output_dir: str = None):
        """
        Generate comprehensive statistical visualization plots.
        
        Args:
            analysis_results: Results from run_comprehensive_analysis
            output_dir: Directory to save plots (if None, displays only)
        """
        all_tests = analysis_results['all_tests']
        
        if not all_tests:
            print("No test results to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Statistical Analysis: LLM vs Algorithm Performance', fontsize=16, fontweight='bold')
        
        # 1. P-value distribution
        p_values = [t.p_value for t in all_tests]
        axes[0, 0].hist(p_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(0.05, color='red', linestyle='--', label='α = 0.05')
        axes[0, 0].set_xlabel('P-values')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of P-values')
        axes[0, 0].legend()
        
        # 2. Effect sizes
        effect_sizes = [t.cohens_d for t in all_tests]
        axes[0, 1].hist(effect_sizes, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].axvline(0, color='black', linestyle='-', alpha=0.5)
        axes[0, 1].axvline(-0.8, color='red', linestyle='--', alpha=0.7, label='Large effect')
        axes[0, 1].axvline(0.8, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_xlabel("Cohen's d")
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Effect Sizes')
        axes[0, 1].legend()
        
        # 3. Power analysis
        powers = [t.power for t in all_tests]
        axes[0, 2].hist(powers, bins=20, alpha=0.7, color='gold', edgecolor='black')
        axes[0, 2].axvline(0.8, color='red', linestyle='--', label='Power threshold')
        axes[0, 2].set_xlabel('Statistical Power')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Distribution of Statistical Power')
        axes[0, 2].legend()
        
        # 4. Effect size vs p-value scatter
        axes[1, 0].scatter(effect_sizes, p_values, alpha=0.6, color='purple')
        axes[1, 0].axhline(0.05, color='red', linestyle='--', label='α = 0.05')
        axes[1, 0].axvline(0, color='black', linestyle='-', alpha=0.5)
        axes[1, 0].set_xlabel("Cohen's d")
        axes[1, 0].set_ylabel('P-value')
        axes[1, 0].set_title('Effect Size vs P-value')
        axes[1, 0].legend()
        
        # 5. Summary by LLM
        llm_data = {}
        for test in all_tests:
            llm = test.test_name.split('_')[0]
            if llm not in llm_data:
                llm_data[llm] = {'effects': [], 'significant': 0, 'total': 0}
            llm_data[llm]['effects'].append(abs(test.cohens_d))
            llm_data[llm]['total'] += 1
            if getattr(test, 'significant_corrected', test.significant):
                llm_data[llm]['significant'] += 1
        
        llms = list(llm_data.keys())
        mean_effects = [np.mean(llm_data[llm]['effects']) for llm in llms]
        sig_rates = [llm_data[llm]['significant'] / llm_data[llm]['total'] for llm in llms]
        
        bars = axes[1, 1].bar(llms, mean_effects, alpha=0.7, color='coral')
        axes[1, 1].set_xlabel('LLM')
        axes[1, 1].set_ylabel('Mean |Effect Size|')
        axes[1, 1].set_title('Mean Effect Size by LLM')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Add significance rate as text on bars
        for i, (bar, rate) in enumerate(zip(bars, sig_rates)):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{rate:.1%}', ha='center', va='bottom', fontsize=10)
        
        # 6. QQ plot for normality check
        from scipy.stats import probplot
        probplot(effect_sizes, dist="norm", plot=axes[1, 2])
        axes[1, 2].set_title('Q-Q Plot: Effect Sizes vs Normal')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            output_path = Path(output_dir) / "statistical_analysis_plots.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to {output_path}")
        
        plt.show()
    
    def generate_report(self, analysis_results: Dict, output_file: str = None):
        """
        Generate comprehensive statistical report.
        
        Args:
            analysis_results: Results from run_comprehensive_analysis
            output_file: File path for report (if None, prints to console)
        """
        summary = analysis_results['summary']
        correction_method = analysis_results['correction_method']
        
        report_lines = [
            "=" * 100,
            "COMPREHENSIVE STATISTICAL ANALYSIS REPORT",
            "=" * 100,
            "",
            f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Significance Level: α = {self.alpha}",
            f"Multiple Comparison Correction: {correction_method.upper()}",
            f"Power Threshold: {self.power_threshold}",
            "",
            "SUMMARY STATISTICS",
            "-" * 50,
            f"Total Comparisons: {summary['total_tests']}",
            f"Significant (Uncorrected): {summary['significant_uncorrected']} ({summary['significant_uncorrected']/summary['total_tests']:.1%})",
            f"Significant (Corrected): {summary['significant_corrected']} ({summary['significant_corrected']/summary['total_tests']:.1%})",
            "",
            "EFFECT SIZES",
            "-" * 50,
            f"Mean Effect Size (|Cohen's d|): {abs(summary['mean_effect_size']):.3f}",
            f"Median Effect Size: {abs(summary['median_effect_size']):.3f}",
            "",
            "Effect Size Distribution:",
            f"  • Negligible (|d| < 0.2): {summary['effect_size_distribution']['negligible']} ({summary['effect_size_distribution']['negligible']/summary['total_tests']:.1%})",
            f"  • Small (0.2 ≤ |d| < 0.5): {summary['effect_size_distribution']['small']} ({summary['effect_size_distribution']['small']/summary['total_tests']:.1%})",
            f"  • Medium (0.5 ≤ |d| < 0.8): {summary['effect_size_distribution']['medium']} ({summary['effect_size_distribution']['medium']/summary['total_tests']:.1%})",
            f"  • Large (|d| ≥ 0.8): {summary['effect_size_distribution']['large']} ({summary['effect_size_distribution']['large']/summary['total_tests']:.1%})",
            "",
            "STATISTICAL POWER",
            "-" * 50,
            f"Mean Statistical Power: {summary['mean_power']:.3f}",
            f"Underpowered Tests (< {self.power_threshold}): {summary['underpowered_tests']} ({summary['underpowered_tests']/summary['total_tests']:.1%})",
            "",
            "P-VALUE ANALYSIS",
            "-" * 50,
            f"Median P-value: {summary['median_p_value']:.4f}",
            f"Minimum P-value: {summary['min_p_value']:.2e}",
            "",
            "=" * 100,
        ]
        
        report_text = "\\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Statistical report saved to {output_file}")
        else:
            print(report_text)
        
        return report_text


def example_usage():
    """
    Example showing how to use StatisticalTester.
    """
    # Initialize tester
    tester = StatisticalTester(alpha=0.05)
    
    # Simulate some comparison data
    np.random.seed(42)
    comparison_data = {}
    
    llms = ['GPT', 'Claude', 'Gemini', 'DeepSeek']
    datasets = ['titanic', 'sachs', 'alarm']
    algorithms = ['PC', 'LiNGAM', 'FCI']
    
    for llm in llms:
        comparison_data[llm] = {}
        for dataset in datasets:
            comparison_data[llm][dataset] = {}
            for algorithm in algorithms:
                # Simulate LLM predictions and algorithm ground truth
                n_runs = 50
                
                # Ground truth (algorithm performance)
                algo_truth = np.random.beta(6, 4, n_runs)  # Skewed toward higher performance
                
                # LLM predictions (correlated with truth + bias + noise)
                llm_bias = np.random.normal(0, 0.05)  # Small systematic bias
                noise = np.random.normal(0, 0.1, n_runs)  # Prediction noise
                llm_preds = algo_truth + llm_bias + noise
                llm_preds = np.clip(llm_preds, 0, 1)  # Keep in [0,1] range
                
                comparison_data[llm][dataset][algorithm] = {
                    'llm_predictions': llm_preds,
                    'algorithm_truth': algo_truth
                }
    
    # Run comprehensive analysis
    print("Running comprehensive statistical analysis...")
    results = tester.run_comprehensive_analysis(comparison_data, correction_method='fdr_bh')
    
    # Generate report
    tester.generate_report(results)
    
    # Create summary table
    summary_table = tester.create_summary_table(results)
    print("\\n" + "="*100)
    print("DETAILED RESULTS TABLE")
    print("="*100)
    print(summary_table.head(10).to_string(index=False))
    
    # Generate plots
    tester.generate_statistical_plots(results)
    
    return results, summary_table


if __name__ == "__main__":
    results, table = example_usage()