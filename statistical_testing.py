#!/usr/bin/env python3
"""
Statistical Significance Testing for Algorithm vs LLM Comparison
==================================================================

Implementation guide for adding rigorous statistical tests to 
compare_algorithms_vs_llms.py to boost UAI 2026 acceptance.

This module adds:
1. Bootstrap confidence intervals for prediction accuracy
2. Paired t-tests (LLM predictions vs algorithm ground truth)
3. Significance level reporting (p-values, effect sizes)
4. Multiple comparison correction (Bonferroni)
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class StatisticalComparator:
    """
    Performs rigorous statistical testing on algorithm vs LLM comparisons.
    """
    
    def __init__(self, alpha: float = 0.05, n_bootstrap: int = 10000):
        """
        Args:
            alpha: Significance level (default 0.05 for 95% CI)
            n_bootstrap: Number of bootstrap samples for CI computation
        """
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
    
    def compute_bootstrap_ci(self, 
                            values: np.ndarray, 
                            confidence: float = 0.95,
                            seed: int = 42) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence interval for a metric.
        
        Args:
            values: Array of metric values (e.g., accuracies from multiple runs)
            confidence: Confidence level (default 95%)
            seed: Random seed
            
        Returns:
            (lower_ci, point_estimate, upper_ci)
        """
        rng = np.random.default_rng(seed)
        n = len(values)
        
        # Compute bootstrap replicates
        bootstrap_means = []
        for _ in range(self.n_bootstrap):
            bootstrap_sample = rng.choice(values, size=n, replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Compute confidence interval
        alpha = 1 - confidence
        lower_percentile = (alpha/2) * 100
        upper_percentile = (1 - alpha/2) * 100
        
        lower_ci = np.percentile(bootstrap_means, lower_percentile)
        upper_ci = np.percentile(bootstrap_means, upper_percentile)
        point_estimate = np.mean(values)
        
        return lower_ci, point_estimate, upper_ci
    
    def paired_ttest(self,
                     llm_predictions: np.ndarray,
                     algorithm_ground_truth: np.ndarray,
                     alt_hypothesis: str = 'two-sided') -> Dict:
        """
        Paired t-test comparing LLM predictions vs algorithm ground truth.
        
        Tests: H0: mean(LLM_pred) = mean(algo_truth)
               Ha: mean(LLM_pred) ≠ mean(algo_truth)
        
        Args:
            llm_predictions: Array of LLM prediction accuracies
            algorithm_ground_truth: Array of algorithm actual accuracies
            alt_hypothesis: 'two-sided', 'less', or 'greater'
            
        Returns:
            Dictionary with test results
        """
        if len(llm_predictions) != len(algorithm_ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        # Compute paired differences
        differences = llm_predictions - algorithm_ground_truth
        
        # Standard paired t-test
        t_stat, p_value = stats.ttest_1samp(differences, 0, 
                                            alternative=alt_hypothesis)
        
        # Cohen's d effect size
        cohens_d = np.mean(differences) / (np.std(differences) + 1e-10)
        
        # Interpretation
        if abs(cohens_d) < 0.2:
            effect_size = 'negligible'
        elif abs(cohens_d) < 0.5:
            effect_size = 'small'
        elif abs(cohens_d) < 0.8:
            effect_size = 'medium'
        else:
            effect_size = 'large'
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'effect_size': effect_size,
            'mean_difference': np.mean(differences),
            'std_difference': np.std(differences),
            'significant': p_value < self.alpha,
            'n_samples': len(llm_predictions)
        }
    
    def multiple_comparison_correction(self,
                                     p_values: List[float],
                                     method: str = 'bonferroni') -> Tuple[List[float], List[bool]]:
        """
        Correct p-values for multiple comparisons.
        
        Args:
            p_values: List of p-values from multiple tests
            method: 'bonferroni' or 'fdr' (false discovery rate)
            
        Returns:
            (corrected_p_values, is_significant_list)
        """
        if method == 'bonferroni':
            # Bonferroni correction: multiply each p-value by number of tests
            corrected_p = [min(1.0, p * len(p_values)) for p in p_values]
        
        elif method == 'fdr':
            # Benjamini-Hochberg FDR correction
            sorted_indices = np.argsort(p_values)
            sorted_p = np.array(p_values)[sorted_indices]
            
            # Compute FDR threshold
            num_tests = len(p_values)
            ranks = np.arange(1, num_tests + 1)
            thresholds = self.alpha * ranks / num_tests
            
            # Find largest p <= threshold
            valid_tests = sorted_p <= thresholds
            if np.any(valid_tests):
                cutoff_index = np.max(np.where(valid_tests))
                corrected_p = [0.0] * num_tests
                for i in sorted_indices[:cutoff_index + 1]:
                    corrected_p[i] = p_values[i]
            else:
                corrected_p = [1.0] * num_tests
        
        else:
            raise ValueError(f"Unknown correction method: {method}")
        
        # Determine significance
        is_significant = [p < self.alpha for p in corrected_p]
        
        return corrected_p, is_significant
    
    def create_significance_summary(self,
                                  results_dict: Dict[str, Dict]) -> pd.DataFrame:
        """
        Create summary table of statistical test results.
        
        Args:
            results_dict: Results from multiple comparisons
                         format: {llm_name: {dataset: {algo: stats}}}
        
        Returns:
            DataFrame with one row per comparison
        """
        rows = []
        
        for llm_name, datasets in results_dict.items():
            for dataset_name, algos in datasets.items():
                for algo_name, stats_result in algos.items():
                    rows.append({
                        'LLM': llm_name,
                        'Dataset': dataset_name,
                        'Algorithm': algo_name,
                        'Mean_Pred_Accuracy': stats_result.get('mean_prediction', np.nan),
                        'Mean_Algo_Accuracy': stats_result.get('mean_algorithm', np.nan),
                        'Mean_Difference': stats_result['mean_difference'],
                        'Std_Difference': stats_result['std_difference'],
                        'T_Statistic': stats_result['t_statistic'],
                        'P_Value': stats_result['p_value'],
                        'Cohens_D': stats_result['cohens_d'],
                        'Effect_Size': stats_result['effect_size'],
                        'Significant_α=0.05': stats_result['significant'],
                        'N_Samples': stats_result['n_samples']
                    })
        
        df = pd.DataFrame(rows)
        
        # Sort by p-value
        df = df.sort_values('P_Value')
        
        return df
    
    def print_report(self, summary_df: pd.DataFrame, method: str = 'bonferroni'):
        """
        Print formatted statistical significance report.
        
        Args:
            summary_df: DataFrame from create_significance_summary
            method: Multiple comparison correction method
        """
        print("\\n" + "="*100)
        print("STATISTICAL SIGNIFICANCE TESTING REPORT")
        print("="*100)
        
        # Apply multiple comparison correction
        corrected_p, is_sig = self.multiple_comparison_correction(
            summary_df['P_Value'].values,
            method=method
        )
        summary_df['P_Value_Corrected'] = corrected_p
        summary_df['Significant_Corrected'] = is_sig
        
        # Overall statistics
        print(f"\\nTotal comparisons: {len(summary_df)}")
        print(f"Significant (uncorrected, α={self.alpha}): {summary_df['Significant_α=0.05'].sum()}")
        print(f"Significant (corrected, {method}): {sum(is_sig)}")
        
        # Show significant results
        sig_results = summary_df[summary_df['Significant_Corrected']]
        if len(sig_results) > 0:
            print(f"\\nSignificant Findings ({len(sig_results)}):")
            print("-" * 100)
            for _, row in sig_results.iterrows():
                print(f"  {row['LLM']:15s} × {row['Dataset']:15s} × {row['Algorithm']:10s}")
                print(f"    Mean Diff: {row['Mean_Difference']:+.4f}, " +
                      f"Cohen's d: {row['Cohens_D']:+.3f} ({row['Effect_Size']}) " +
                      f"p={row['P_Value_Corrected']:.2e}")
        else:
            print("\\nNo significant differences after multiple comparison correction.")
        
        # By LLM summary
        print(f"\\nSignificance by LLM:")
        print("-" * 100)
        llm_summary = sig_results.groupby('LLM').size()
        for llm, count in llm_summary.items():
            total = len(summary_df[summary_df['LLM'] == llm])
            print(f"  {llm:20s}: {count:3d}/{total:3d} comparisons significant")
        
        # By Dataset summary
        print(f"\\nSignificance by Dataset:")
        print("-" * 100)
        dataset_summary = sig_results.groupby('Dataset').size()
        for dataset, count in dataset_summary.items():
            total = len(summary_df[summary_df['Dataset'] == dataset])
            print(f"  {dataset:20s}: {count:3d}/{total:3d} comparisons significant")
        
        print("\\n" + "="*100 + "\\n")


# Example usage for integration into compare_algorithms_vs_llms.py
def example_integration():
    """
    Shows how to integrate StatisticalComparator into existing code.
    """
    
    # Initialize
    comparator = StatisticalComparator(alpha=0.05, n_bootstrap=10000)
    
    # Simulate comparison results
    # (In real code, these come from algorithm runs and LLM predictions)
    np.random.seed(42)
    
    results_dict = {}
    
    for llm in ['GPT', 'Claude', 'Gemini', 'DeepSeek']:
        results_dict[llm] = {}
        
        for dataset in ['titanic', 'sachs', 'alarm']:
            results_dict[llm][dataset] = {}
            
            for algo in ['PC', 'LiNGAM', 'FCI']:
                # Simulate: LLM predictions slightly correlated with ground truth
                algo_accuracy = np.random.beta(8, 2, 100)  # Ground truth
                llm_accuracy = algo_accuracy + np.random.normal(0, 0.1, 100)
                llm_accuracy = np.clip(llm_accuracy, 0, 1)
                
                # Run paired t-test
                test_result = comparator.paired_ttest(llm_accuracy, algo_accuracy)
                
                results_dict[llm][dataset][algo] = {
                    'mean_prediction': np.mean(llm_accuracy),
                    'mean_algorithm': np.mean(algo_accuracy),
                    **test_result
                }
    
    # Create summary
    summary_df = comparator.create_significance_summary(results_dict)
    
    # Print report
    comparator.print_report(summary_df, method='bonferroni')
    
    # Save to CSV
    summary_df.to_csv('statistical_significance_results.csv', index=False)
    print("Results saved to statistical_significance_results.csv")
    
    return summary_df


if __name__ == '__main__':
    # Run example
    df = example_integration()
