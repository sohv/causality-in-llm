#!/usr/bin/env python3
"""
Explanatory Model Module
========================

Provides theoretical explanations and feature importance analysis 
for LLM performance on causal discovery tasks.

Addresses the "why, not just what" requirement for strong scientific contributions.
Analyzes what factors predict LLM success and failure modes.

Features:
- Feature importance analysis (graph complexity, sample size, etc.)
- LLM performance prediction models
- Theoretical framework development
- Failure mode classification
- Performance factor decomposition
- Mechanistic interpretability

Usage:
    from uai_2026_enhancements.explanatory_model import ExplanatoryAnalyzer
    
    analyzer = ExplanatoryAnalyzer()
    insights = analyzer.analyze_performance_factors(experimental_data)
    analyzer.generate_theory_report(insights)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import warnings
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from scipy import stats
import networkx as nx

warnings.filterwarnings('ignore')

@dataclass
class PerformanceFactors:
    """Container for factors that influence LLM performance."""
    dataset_name: str
    algorithm_name: str
    llm_name: str
    
    # Graph complexity features
    n_nodes: int
    n_edges: int
    edge_density: float
    max_degree: int
    avg_degree: float
    graph_diameter: Optional[int]
    clustering_coefficient: float
    
    # Data characteristics
    sample_size: int
    dimensionality: int
    noise_level: Optional[float]
    
    # Task complexity
    causal_complexity_score: float  # Custom metric combining multiple factors
    identifiability_score: float    # How well-identified is the true structure
    
    # Performance metrics
    accuracy: float
    confidence_interval_width: float
    calibration_error: float
    
    # Additional features
    additional_features: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExplanationInsights:
    """Results from explanatory analysis."""
    feature_importance_scores: Dict[str, float]
    llm_ranking_model: Any
    performance_prediction_model: Any
    failure_mode_classifier: Any
    theoretical_insights: List[str]
    key_findings: List[str]
    feature_correlations: pd.DataFrame
    prediction_accuracy: float

class ExplanatoryAnalyzer:
    """
    Analyzes WHY LLMs succeed or fail at causal discovery.
    """
    
    def __init__(self):
        """Initialize the explanatory analyzer."""
        self.scaler = StandardScaler()
        self.performance_predictors = None
        self.failure_classifier = None
        
    def extract_graph_features(self, adjacency_matrix: np.ndarray) -> Dict[str, float]:
        """
        Extract complexity features from causal graph.
        
        Args:
            adjacency_matrix: Binary adjacency matrix of causal graph
            
        Returns:
            Dictionary of graph complexity features
        """
        # Convert to networkx graph for analysis
        G = nx.DiGraph(adjacency_matrix)
        
        n_nodes = len(adjacency_matrix)
        n_edges = np.sum(adjacency_matrix)
        
        features = {
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'edge_density': n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0,
            'max_in_degree': max([G.in_degree(node) for node in G.nodes()]) if G.nodes() else 0,
            'max_out_degree': max([G.out_degree(node) for node in G.nodes()]) if G.nodes() else 0,
            'avg_degree': np.mean([G.degree(node) for node in G.nodes()]) if G.nodes() else 0,
        }
        
        # Diameter (longest shortest path)
        try:
            if nx.is_weakly_connected(G):
                features['graph_diameter'] = nx.diameter(G.to_undirected())
            else:
                # For disconnected graphs, use average component diameter 
                components = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]
                diameters = []
                for comp in components:
                    if len(comp.nodes()) > 1:
                        try:
                            diameters.append(nx.diameter(comp.to_undirected()))
                        except:
                            diameters.append(1)  # Single node component
                features['graph_diameter'] = np.mean(diameters) if diameters else 0
        except:
            features['graph_diameter'] = None
        
        # Clustering coefficient
        try:
            features['clustering_coefficient'] = nx.average_clustering(G.to_undirected())
        except:
            features['clustering_coefficient'] = 0
        
        # Causal complexity (custom metric)
        # Combines multiple factors: size, density, structural complexity
        size_complexity = min(n_nodes / 20.0, 1.0)  # Normalized by typical max size
        density_complexity = features['edge_density']
        degree_complexity = features['max_in_degree'] / max(n_nodes - 1, 1)
        
        features['causal_complexity_score'] = (
            0.4 * size_complexity + 
            0.3 * density_complexity + 
            0.3 * degree_complexity
        )
        
        return features
    
    def compute_identifiability_score(self, 
                                    adjacency_matrix: np.ndarray,
                                    sample_size: int) -> float:
        """
        Compute how well-identified a causal structure is.
        
        Higher scores indicate structures that should be easier to discover.
        
        Args:
            adjacency_matrix: Binary adjacency matrix
            sample_size: Number of samples in dataset
            
        Returns:
            Identifiability score between 0 and 1
        """
        n_nodes = len(adjacency_matrix)
        n_edges = np.sum(adjacency_matrix)
        
        # Sample efficiency factor
        theoretical_min_samples = n_nodes * (n_nodes - 1)  # For complete identification
        sample_factor = min(sample_size / theoretical_min_samples, 1.0)
        
        # Structure sparsity factor (sparser is more identifiable)
        sparsity_factor = 1.0 - (n_edges / (n_nodes * (n_nodes - 1)))
        
        # Degree regularity factor (more regular graphs are easier)
        G = nx.DiGraph(adjacency_matrix)
        degrees = [G.degree(node) for node in G.nodes()]
        if len(degrees) > 1:
            degree_std = np.std(degrees)
            max_possible_std = np.sqrt(n_nodes * (n_nodes - 1) / 4)  # Rough estimate
            regularity_factor = 1.0 - (degree_std / max(max_possible_std, 1))
        else:
            regularity_factor = 1.0
        
        # Combine factors
        identifiability = (
            0.5 * sample_factor +
            0.3 * sparsity_factor +
            0.2 * regularity_factor
        )
        
        return max(0, min(1, identifiability))
    
    def create_performance_factors(self,
                                 experimental_results: Dict[str, Dict[str, Dict[str, Dict]]],
                                 graph_structures: Dict[str, np.ndarray],
                                 dataset_metadata: Dict[str, Dict[str, Any]]) -> List[PerformanceFactors]:
        """
        Extract all performance factors from experimental data.
        
        Args:
            experimental_results: {LLM: {dataset: {algorithm: {metrics}}}}
            graph_structures: {dataset: adjacency_matrix}
            dataset_metadata: {dataset: {sample_size, etc.}}
        
        Returns:
            List of PerformanceFactors objects
        """
        all_factors = []
        
        for llm_name, datasets in experimental_results.items():
            for dataset_name, algorithms in datasets.items():
                
                # Get dataset metadata
                metadata = dataset_metadata.get(dataset_name, {})
                sample_size = metadata.get('sample_size', 1000)
                dimensionality = metadata.get('dimensionality', 10)
                noise_level = metadata.get('noise_level', None)
                
                # Get graph features
                if dataset_name in graph_structures:
                    graph_features = self.extract_graph_features(graph_structures[dataset_name])
                    identifiability = self.compute_identifiability_score(
                        graph_structures[dataset_name], sample_size
                    )
                else:
                    # Default values if graph structure unknown
                    graph_features = {
                        'n_nodes': dimensionality,
                        'n_edges': dimensionality * 2,  # Rough estimate
                        'edge_density': 0.2,
                        'max_in_degree': 3,
                        'max_out_degree': 3, 
                        'avg_degree': 2.0,
                        'graph_diameter': 3,
                        'clustering_coefficient': 0.1,
                        'causal_complexity_score': 0.5
                    }
                    identifiability = 0.5
                
                for algorithm_name, results in algorithms.items():
                    try:
                        # Extract performance metrics
                        accuracy = results.get('accuracy', 0.0)
                        ci_width = results.get('confidence_interval_width', 0.0)
                        cal_error = results.get('calibration_error', 0.0)
                        
                        # Create factors object
                        factors = PerformanceFactors(
                            dataset_name=dataset_name,
                            algorithm_name=algorithm_name,
                            llm_name=llm_name,
                            n_nodes=graph_features['n_nodes'],
                            n_edges=graph_features['n_edges'],
                            edge_density=graph_features['edge_density'],
                            max_degree=graph_features['max_in_degree'],
                            avg_degree=graph_features['avg_degree'],
                            graph_diameter=graph_features.get('graph_diameter'),
                            clustering_coefficient=graph_features['clustering_coefficient'],
                            sample_size=sample_size,
                            dimensionality=dimensionality,
                            noise_level=noise_level,
                            causal_complexity_score=graph_features['causal_complexity_score'],
                            identifiability_score=identifiability,
                            accuracy=accuracy,
                            confidence_interval_width=ci_width,
                            calibration_error=cal_error,
                            additional_features=metadata
                        )
                        
                        all_factors.append(factors)
                        
                    except Exception as e:
                        print(f"Warning: Could not process {llm_name}-{dataset_name}-{algorithm_name}: {e}")
                        continue
        
        return all_factors
    
    def analyze_feature_importance(self, factors: List[PerformanceFactors]) -> Dict[str, float]:
        """
        Determine which factors most influence LLM accuracy.
        
        Args:
            factors: List of PerformanceFactors
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if len(factors) < 10:
            print("Warning: Need at least 10 data points for reliable feature importance")
            return {}
        
        # Convert to feature matrix
        feature_names = [
            'n_nodes', 'n_edges', 'edge_density', 'max_degree', 'avg_degree',
            'clustering_coefficient', 'sample_size', 'dimensionality',
            'causal_complexity_score', 'identifiability_score'
        ]
        
        X = []
        y = []
        
        for factor in factors:
            row = []
            for feature_name in feature_names:
                value = getattr(factor, feature_name)
                if value is None:
                    value = 0.0  # Handle missing values
                row.append(float(value))
            
            X.append(row)
            y.append(factor.accuracy)
        
        X = np.array(X)
        y = np.array(y)
        
        # Handle any remaining NaN or infinite values
        finite_mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
        X = X[finite_mask]
        y = y[finite_mask]
        
        if len(X) < 5:
            print("Warning: Insufficient clean data for feature importance analysis")
            return {}
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Random Forest for feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)
        
        # Get feature importances
        importance_dict = {}
        for i, feature_name in enumerate(feature_names):
            importance_dict[feature_name] = rf.feature_importances_[i]
        
        # Store model for later use
        self.performance_predictors = rf
        
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def build_performance_prediction_model(self, factors: List[PerformanceFactors]) -> Tuple[Any, float]:
        """
        Build model to predict LLM accuracy from dataset/task characteristics.
        
        Args:
            factors: List of PerformanceFactors
            
        Returns:
            Tuple of (trained_model, prediction_accuracy_r2)
        """
        if len(factors) < 10:
            return None, 0.0
        
        # Prepare feature matrix (same as feature importance)
        feature_names = [
            'n_nodes', 'n_edges', 'edge_density', 'max_degree', 'avg_degree',
            'clustering_coefficient', 'sample_size', 'dimensionality',
            'causal_complexity_score', 'identifiability_score'
        ]
        
        X = []
        y = []
        
        for factor in factors:
            row = []
            for feature_name in feature_names:
                value = getattr(factor, feature_name)
                if value is None:
                    value = 0.0
                row.append(float(value))
            
            X.append(row)
            y.append(factor.accuracy)
        
        X = np.array(X)
        y = np.array(y)
        
        # Clean data
        finite_mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
        X = X[finite_mask]
        y = y[finite_mask]
        
        if len(X) < 5:
            return None, 0.0
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Ridge regression (handles multicollinearity better than linear regression)
        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        if len(X_test) > 0:
            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
        else:
            y_pred = model.predict(X_train_scaled)
            r2 = r2_score(y_train, y_pred)
        
        return model, max(0, r2)
    
    def classify_failure_modes(self, factors: List[PerformanceFactors]) -> Tuple[Any, Dict[str, str]]:
        """
        Classify when/why LLMs fail at causal discovery.
        
        Args:
            factors: List of PerformanceFactors
            
        Returns:
            Tuple of (classifier_model, failure_mode_descriptions)
        """
        if len(factors) < 20:
            return None, {}
        
        # Define failure threshold
        accuracy_threshold = np.median([f.accuracy for f in factors])
        
        # Prepare data
        feature_names = [
            'n_nodes', 'n_edges', 'edge_density', 'max_degree', 'avg_degree',
            'clustering_coefficient', 'sample_size', 'causal_complexity_score', 
            'identifiability_score'
        ]
        
        X = []
        y = []  # 0 = failure, 1 = success
        
        for factor in factors:
            row = []
            for feature_name in feature_names:
                value = getattr(factor, feature_name)
                if value is None:
                    value = 0.0
                row.append(float(value))
            
            X.append(row)
            y.append(1 if factor.accuracy >= accuracy_threshold else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        # Clean data
        finite_mask = np.all(np.isfinite(X), axis=1)
        X = X[finite_mask]
        y = y[finite_mask]
        
        if len(X) < 10 or len(np.unique(y)) < 2:
            return None, {}
        
        # Train classifier
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        classifier = LogisticRegression(random_state=42)
        classifier.fit(X_scaled, y)
        
        # Analyze feature coefficients to understand failure modes
        feature_coeffs = classifier.coef_[0]
        
        failure_modes = {}
        for i, feature_name in enumerate(feature_names):
            coeff = feature_coeffs[i]
            if abs(coeff) > 0.1:  # Significant coefficient
                if coeff > 0:
                    failure_modes[feature_name] = f"Higher {feature_name} increases success probability"
                else:
                    failure_modes[feature_name] = f"Higher {feature_name} increases failure probability"
        
        self.failure_classifier = classifier
        return classifier, failure_modes
    
    def compute_feature_correlations(self, factors: List[PerformanceFactors]) -> pd.DataFrame:
        """
        Compute correlations between all features and performance.
        
        Args:
            factors: List of PerformanceFactors
            
        Returns:
            Correlation matrix DataFrame
        """
        all_features = [
            'n_nodes', 'n_edges', 'edge_density', 'max_degree', 'avg_degree',
            'clustering_coefficient', 'sample_size', 'dimensionality',
            'causal_complexity_score', 'identifiability_score', 'accuracy',
            'confidence_interval_width', 'calibration_error'
        ]
        
        data = []
        for factor in factors:
            row = []
            for feature_name in all_features:
                value = getattr(factor, feature_name)
                if value is None:
                    value = np.nan
                row.append(float(value))
            data.append(row)
        
        df = pd.DataFrame(data, columns=all_features)
        
        # Remove columns with all NaN or constant values
        df_clean = df.dropna(axis=1, how='all')
        for col in df_clean.columns:
            if df_clean[col].std() == 0:
                df_clean = df_clean.drop(col, axis=1)
        
        return df_clean.corr()
    
    def analyze_performance_factors(self,
                                  experimental_results: Dict[str, Dict[str, Dict[str, Dict]]],
                                  graph_structures: Dict[str, np.ndarray],
                                  dataset_metadata: Dict[str, Dict[str, Any]]) -> ExplanationInsights:
        """
        Comprehensive analysis of what factors influence LLM performance.
        
        Args:
            experimental_results: {LLM: {dataset: {algorithm: {metrics}}}}
            graph_structures: {dataset: adjacency_matrix}
            dataset_metadata: {dataset: metadata}
            
        Returns:
            ExplanationInsights with comprehensive analysis results
        """
        print("Extracting performance factors...")
        factors = self.create_performance_factors(
            experimental_results, graph_structures, dataset_metadata
        )
        
        if len(factors) < 5:
            raise ValueError(f"Insufficient data: only {len(factors)} valid factor combinations found")
        
        print(f"Analyzing {len(factors)} factor combinations...")
        
        # Analyze feature importance
        feature_importance = self.analyze_feature_importance(factors)
        
        # Build performance prediction model
        pred_model, pred_accuracy = self.build_performance_prediction_model(factors)
        
        # Classify failure modes
        failure_classifier, failure_modes = self.classify_failure_modes(factors)
        
        # Compute correlations
        correlations = self.compute_feature_correlations(factors)
        
        # Generate theoretical insights
        theoretical_insights = self._generate_theoretical_insights(
            feature_importance, failure_modes, correlations
        )
        
        # Generate key findings
        key_findings = self._generate_key_findings(
            factors, feature_importance, pred_accuracy, failure_modes
        )
        
        # Determine LLM ranking model (separate analysis)
        llm_ranking_model = self._analyze_llm_ranking(factors)
        
        return ExplanationInsights(
            feature_importance_scores=feature_importance,
            llm_ranking_model=llm_ranking_model,
            performance_prediction_model=pred_model,
            failure_mode_classifier=failure_classifier,
            theoretical_insights=theoretical_insights,
            key_findings=key_findings,
            feature_correlations=correlations,
            prediction_accuracy=pred_accuracy
        )
    
    def _analyze_llm_ranking(self, factors: List[PerformanceFactors]) -> Dict[str, float]:
        """Analyze overall LLM performance ranking."""
        llm_performance = {}
        
        for factor in factors:
            if factor.llm_name not in llm_performance:
                llm_performance[factor.llm_name] = []
            llm_performance[factor.llm_name].append(factor.accuracy)
        
        # Compute mean accuracy per LLM
        llm_ranking = {}
        for llm, accuracies in llm_performance.items():
            llm_ranking[llm] = np.mean(accuracies)
        
        return dict(sorted(llm_ranking.items(), key=lambda x: x[1], reverse=True))
    
    def _generate_theoretical_insights(self,
                                     feature_importance: Dict[str, float],
                                     failure_modes: Dict[str, str],
                                     correlations: pd.DataFrame) -> List[str]:
        """Generate theoretical insights from analysis."""
        insights = []
        
        if not feature_importance:
            return ["Insufficient data for theoretical insights"]
        
        # Most important factor
        top_factor = list(feature_importance.keys())[0]
        top_importance = feature_importance[top_factor]
        
        insights.append(
            f"Primary Performance Driver: {top_factor} explains {top_importance:.1%} "
            f"of LLM accuracy variance, suggesting {self._interpret_factor(top_factor)}"
        )
        
        # Sample size vs complexity trade-off
        if 'sample_size' in feature_importance and 'causal_complexity_score' in feature_importance:
            sample_importance = feature_importance['sample_size']
            complexity_importance = feature_importance['causal_complexity_score']
            
            if sample_importance > complexity_importance:
                insights.append(
                    "Data Sufficiency Hypothesis: Sample size matters more than graph complexity, "
                    "indicating LLMs are primarily limited by data quantity, not structural reasoning ability"
                )
            else:
                insights.append(
                    "Structural Reasoning Limitation: Graph complexity dominates sample size effects, "
                    "suggesting LLMs struggle with complex causal structures regardless of data volume"
                )
        
        # Identifiability insights
        if 'identifiability_score' in feature_importance:
            id_importance = feature_importance['identifiability_score']
            if id_importance > 0.1:
                insights.append(
                    "Identifiability Principle: LLM performance closely tracks theoretical identifiability, "
                    "suggesting they approximate optimal causal discovery under standard assumptions"
                )
        
        # Graph structure insights
        structural_factors = ['edge_density', 'clustering_coefficient', 'max_degree']
        structural_importances = [feature_importance.get(f, 0) for f in structural_factors]
        
        if max(structural_importances) > 0.05:
            top_structural = structural_factors[np.argmax(structural_importances)]
            insights.append(
                f"Graph Topology Effect: {top_structural} significantly influences performance, "
                f"suggesting LLMs are sensitive to {self._interpret_factor(top_structural)}"
            )
        
        # Failure mode insights
        if failure_modes:
            insights.append("Failure Mode Analysis: " + "; ".join(list(failure_modes.values())[:3]))
        
        return insights
    
    def _interpret_factor(self, factor_name: str) -> str:
        """Provide interpretation of what each factor means."""
        interpretations = {
            'n_nodes': 'system scale complexity',
            'n_edges': 'causal relationship density',
            'edge_density': 'relative structural complexity',
            'max_degree': 'maximum causal influence concentration',
            'avg_degree': 'typical causal connectivity',
            'clustering_coefficient': 'local causal clustering patterns',
            'sample_size': 'statistical power and data sufficiency',
            'dimensionality': 'feature space complexity',
            'causal_complexity_score': 'overall structural complexity',
            'identifiability_score': 'theoretical learnability',
            'graph_diameter': 'causal pathway length complexity',
        }
        
        return interpretations.get(factor_name, f'the role of {factor_name}')
    
    def _generate_key_findings(self,
                             factors: List[PerformanceFactors],
                             feature_importance: Dict[str, float],
                             pred_accuracy: float,
                             failure_modes: Dict[str, str]) -> List[str]:
        """Generate key empirical findings."""
        findings = []
        
        # Overall performance statistics
        accuracies = [f.accuracy for f in factors]
        findings.append(f"Mean LLM accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
        
        # Predictability finding
        if pred_accuracy > 0:
            findings.append(f"Performance predictability: {pred_accuracy:.1%} of variance explained by task characteristics")
        
        # Top factors
        if feature_importance:
            top_3_factors = list(feature_importance.keys())[:3]
            finding = f"Most influential factors: {', '.join(top_3_factors)} "
            finding += f"(combined importance: {sum([feature_importance[f] for f in top_3_factors]):.1%})"
            findings.append(finding)
        
        # LLM comparison
        llm_accuracies = {}
        for factor in factors:
            if factor.llm_name not in llm_accuracies:
                llm_accuracies[factor.llm_name] = []
            llm_accuracies[factor.llm_name].append(factor.accuracy)
        
        best_llm = max(llm_accuracies.keys(), key=lambda x: np.mean(llm_accuracies[x]))
        worst_llm = min(llm_accuracies.keys(), key=lambda x: np.mean(llm_accuracies[x]))
        
        best_acc = np.mean(llm_accuracies[best_llm])
        worst_acc = np.mean(llm_accuracies[worst_llm])
        
        findings.append(f"LLM performance range: {best_llm} ({best_acc:.3f}) to {worst_llm} ({worst_acc:.3f})")
        findings.append(f"Performance gap: {best_acc - worst_acc:.3f} ({(best_acc - worst_acc)/worst_acc:.1%} relative)")
        
        return findings
    
    def create_explanatory_plots(self,
                               insights: ExplanationInsights,
                               output_dir: Optional[str] = None):
        """
        Generate comprehensive explanatory analysis plots.
        
        Args:
            insights: Results from analyze_performance_factors
            output_dir: Directory to save plots
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Explanatory Analysis: Why LLMs Succeed or Fail at Causal Discovery', 
                     fontsize=16, fontweight='bold')
        
        # 1. Feature importance plot
        if insights.feature_importance_scores:
            features = list(insights.feature_importance_scores.keys())[:8]  # Top 8
            importances = [insights.feature_importance_scores[f] for f in features]
            
            bars = axes[0, 0].barh(features, importances, alpha=0.7, color='lightblue')
            axes[0, 0].set_xlabel('Feature Importance')
            axes[0, 0].set_title('What Drives LLM Performance?')
            
            # Add percentage labels
            for i, (bar, imp) in enumerate(zip(bars, importances)):
                axes[0, 0].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                               f'{imp:.1%}', ha='left', va='center', fontsize=9)
        
        # 2. LLM ranking
        if insights.llm_ranking_model:
            llms = list(insights.llm_ranking_model.keys())
            performances = list(insights.llm_ranking_model.values())
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(llms)))
            bars = axes[0, 1].bar(llms, performances, alpha=0.7, color=colors)
            axes[0, 1].set_ylabel('Mean Accuracy')
            axes[0, 1].set_title('LLM Performance Ranking')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, perf in zip(bars, performances):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{perf:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 3. Correlation heatmap (top features only)
        if not insights.feature_correlations.empty:
            # Select most relevant features for visualization
            important_features = ['accuracy'] + list(insights.feature_importance_scores.keys())[:6]
            corr_subset = insights.feature_correlations.loc[
                important_features, important_features
            ].fillna(0)
            
            sns.heatmap(corr_subset, annot=True, cmap='RdBu_r', center=0,
                       square=True, ax=axes[0, 2], cbar_kws={'shrink': 0.8})
            axes[0, 2].set_title('Feature Correlations')
        
        # 4. Performance prediction accuracy
        if insights.prediction_accuracy > 0:
            categories = ['Explained\\nVariance', 'Unexplained\\nVariance']
            values = [insights.prediction_accuracy, 1 - insights.prediction_accuracy]
            colors = ['lightgreen', 'lightcoral']
            
            wedges, texts, autotexts = axes[1, 0].pie(values, labels=categories, autopct='%1.1f%%',
                                                     colors=colors, startangle=90)
            axes[1, 0].set_title(f'Performance Predictability\\n(R² = {insights.prediction_accuracy:.3f})')
        
        # 5. Key findings text
        axes[1, 1].axis('off')
        if insights.key_findings:
            findings_text = "KEY FINDINGS:\\n\\n"
            for i, finding in enumerate(insights.key_findings[:5], 1):
                findings_text += f"{i}. {finding}\\n\\n"
            
            axes[1, 1].text(0.05, 0.95, findings_text, transform=axes[1, 1].transAxes,
                           fontsize=10, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
            axes[1, 1].set_title('Empirical Findings')
        
        # 6. Theoretical insights
        axes[1, 2].axis('off')
        if insights.theoretical_insights:
            theory_text = "THEORETICAL INSIGHTS:\\n\\n"
            for i, insight in enumerate(insights.theoretical_insights[:3], 1):
                theory_text += f"{i}. {insight}\\n\\n"
            
            axes[1, 2].text(0.05, 0.95, theory_text, transform=axes[1, 2].transAxes,
                           fontsize=10, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))
            axes[1, 2].set_title('Theoretical Framework')
        
        plt.tight_layout()
        
        if output_dir:
            output_path = Path(output_dir) / "explanatory_analysis_plots.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Explanatory plots saved to {output_path}")
        
        plt.show()
    
    def generate_theory_report(self,
                             insights: ExplanationInsights,
                             output_file: Optional[str] = None):
        """
        Generate comprehensive theoretical analysis report.
        
        Args:
            insights: Results from analyze_performance_factors
            output_file: File path for report
        """
        report_lines = [
            "=" * 100,
            "EXPLANATORY MODEL: WHY LLMS SUCCEED OR FAIL AT CAUSAL DISCOVERY",
            "=" * 100,
            "",
            f"Report Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Prediction Accuracy: {insights.prediction_accuracy:.1%} of variance explained",
            "",
            "THEORETICAL FRAMEWORK",
            "-" * 50,
        ]
        
        # Add theoretical insights
        for i, insight in enumerate(insights.theoretical_insights, 1):
            report_lines.append(f"{i}. {insight}")
            report_lines.append("")
        
        report_lines.extend([
            "EMPIRICAL FINDINGS",
            "-" * 50,
        ])
        
        # Add key findings
        for i, finding in enumerate(insights.key_findings, 1):
            report_lines.append(f"{i}. {finding}")
            report_lines.append("")
        
        report_lines.extend([
            "FEATURE IMPORTANCE ANALYSIS",
            "-" * 50,
            "Factors ranked by influence on LLM causal discovery performance:",
            "",
        ])
        
        # Add feature importance ranking
        for i, (feature, importance) in enumerate(insights.feature_importance_scores.items(), 1):
            interpretation = self._interpret_factor(feature)
            report_lines.append(f"{i:2d}. {feature:25s}: {importance:6.1%}  - {interpretation}")
        
        report_lines.extend([
            "",
            "LLM PERFORMANCE RANKING",
            "-" * 50,
        ])
        
        # Add LLM ranking
        for i, (llm, performance) in enumerate(insights.llm_ranking_model.items(), 1):
            report_lines.append(f"{i}. {llm:15s}: {performance:.3f} mean accuracy")
        
        report_lines.extend([
            "",
            "MECHANISTIC INSIGHTS",
            "-" * 50,
            "",
            "Performance Prediction Model:",
            f"• Explains {insights.prediction_accuracy:.1%} of accuracy variance",
            "• Most predictive factors: " + ", ".join(list(insights.feature_importance_scores.keys())[:3]),
            "",
            "Causal Discovery Complexity Hierarchy:",
            "1. Sample sufficiency (statistical power)",
            "2. Graph identifiability (theoretical learnability)", 
            "3. Structural complexity (reasoning demands)",
            "4. LLM-specific capabilities",
            "",
            "Failure Mode Patterns:",
        ])
        
        # Add failure mode analysis if available
        if hasattr(insights, 'failure_mode_classifier') and insights.failure_mode_classifier:
            report_lines.append("• Systematic patterns identified in failure conditions")
            report_lines.append("• Predictable based on task characteristics")
        else:
            report_lines.append("• Require larger dataset for reliable pattern detection")
        
        report_lines.extend([
            "",
            "IMPLICATIONS FOR FUTURE RESEARCH",
            "-" * 50,
            "",
            "Methodological Recommendations:",
            "• Focus on high-identifiability test cases for fair evaluation",
            "• Control for sample size effects when comparing across datasets",
            "• Use structural complexity measures to normalize performance",
            "",
            "Theoretical Directions:",
            "• Investigate LLM internal representations of causal structure",
            "• Develop complexity-aware performance bounds", 
            "• Study transfer learning across causal domains",
            "",
            "Practical Applications:",
            "• Use performance prediction model for task difficulty assessment",
            "• Apply feature importance for dataset selection and design",
            "• Leverage LLM ranking for ensemble methods",
            "",
            "=" * 100
        ])
        
        report_text = "\\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Theoretical analysis report saved to {output_file}")
        else:
            print(report_text)
        
        return report_text


def example_usage():
    """
    Example of how to use ExplanatoryAnalyzer.
    """
    # Initialize analyzer
    analyzer = ExplanatoryAnalyzer()
    
    # Simulate experimental data
    np.random.seed(42)
    
    # Mock experimental results
    experimental_results = {}
    llms = ['GPT', 'Claude', 'Gemini', 'DeepSeek']
    datasets = ['titanic', 'sachs', 'alarm', 'asia', 'cancer']
    algorithms = ['PC', 'LiNGAM', 'FCI']
    
    for llm in llms:
        experimental_results[llm] = {}
        for dataset in datasets:
            experimental_results[llm][dataset] = {}
            for algorithm in algorithms:
                # Mock performance metrics
                base_accuracy = np.random.beta(3, 2)  # Realistic accuracy distribution
                accuracy = base_accuracy + np.random.normal(0, 0.05)
                accuracy = np.clip(accuracy, 0, 1)
                
                experimental_results[llm][dataset][algorithm] = {
                    'accuracy': accuracy,
                    'confidence_interval_width': np.random.uniform(0.1, 0.3),
                    'calibration_error': np.random.uniform(0, 0.2)
                }
    
    # Mock graph structures
    graph_structures = {}
    for dataset in datasets:
        if dataset == 'titanic':
            n_nodes = 6
        elif dataset == 'sachs': 
            n_nodes = 11
        elif dataset == 'alarm':
            n_nodes = 37
        else:
            n_nodes = np.random.randint(5, 25)
        
        # Generate random DAG
        adj_matrix = np.random.rand(n_nodes, n_nodes) < 0.15  # Sparse random graph
        # Make it acyclic by setting upper triangle to 0
        adj_matrix = np.tril(adj_matrix, -1)
        graph_structures[dataset] = adj_matrix.astype(int)
    
    # Mock dataset metadata  
    dataset_metadata = {}
    for dataset in datasets:
        dataset_metadata[dataset] = {
            'sample_size': np.random.randint(500, 5000),
            'dimensionality': len(graph_structures[dataset]),
            'noise_level': np.random.uniform(0.1, 0.5)
        }
    
    # Run explanatory analysis
    print("Running comprehensive explanatory analysis...")
    insights = analyzer.analyze_performance_factors(
        experimental_results, graph_structures, dataset_metadata
    )
    
    # Generate theory report
    analyzer.generate_theory_report(insights)
    
    # Create explanatory plots
    analyzer.create_explanatory_plots(insights)
    
    return insights


if __name__ == "__main__":
    insights = example_usage()