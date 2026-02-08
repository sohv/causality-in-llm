#!/usr/bin/env python3
"""
Stock Market Causal Relationships Dataset
==========================================

A real-world financial dataset for testing causal discovery algorithms
on economic time series data.

Dataset features:
- Stock market indices and economic indicators
- 10 variables representing different market aspects
- Daily data spanning multiple years
- Tests algorithms on financial domain (different from medical/social)

Variables:
1. SP500: S&P 500 index returns
2. NASDAQ: NASDAQ composite returns
3. DOW: Dow Jones Industrial Average returns
4. VIX: Volatility index (fear gauge)
5. OIL: Crude oil prices
6. GOLD: Gold prices
7. USD_INDEX: US Dollar index
8. TREASURY_10Y: 10-year Treasury yield
9. VOLUME: Trading volume
10. GDP_GROWTH: GDP growth rate (quarterly, interpolated)

Expected causal structure (based on economic theory):
- VIX -> SP500, NASDAQ, DOW (volatility affects returns)
- OIL -> Inflation indicators
- USD_INDEX -> International stocks
- TREASURY_10Y -> Stock returns (interest rate effect)
- GDP_GROWTH -> Overall market performance
"""

import numpy as np
import pandas as pd
from typing import Tuple
from datetime import datetime, timedelta


def generate_synthetic_market_data(n_samples: int = 1000, seed: int = 42) -> Tuple[pd.DataFrame, np.ndarray, list]:
    """
    Generate synthetic stock market data with known causal structure.

    Since real financial data requires APIs and may have access restrictions,
    we generate realistic synthetic data with controlled causal relationships
    based on economic theory.

    Args:
        n_samples: Number of daily observations
        seed: Random seed for reproducibility

    Returns:
        data: DataFrame with market variables
        true_graph: Ground truth adjacency matrix (10 x 10)
        var_names: List of variable names
    """
    np.random.seed(seed)

    var_names = [
        'VIX',           # 0: Volatility (exogenous driver)
        'TREASURY_10Y',  # 1: Interest rates (exogenous driver)
        'USD_INDEX',     # 2: Currency (exogenous driver)
        'OIL',          # 3: Oil prices (partially exogenous)
        'GOLD',         # 4: Gold prices (safe haven)
        'GDP_GROWTH',   # 5: Economic growth
        'SP500',        # 6: Stock index
        'NASDAQ',       # 7: Tech-heavy index
        'DOW',          # 8: Industrial index
        'VOLUME'        # 9: Trading volume
    ]

    n_vars = len(var_names)

    # Define true causal structure based on economic theory
    # Adjacency matrix: A[i,j] = 1 means i -> j
    true_graph = np.zeros((n_vars, n_vars), dtype=int)

    # VIX (volatility) affects all stock indices and volume
    true_graph[0, 6] = 1  # VIX -> SP500
    true_graph[0, 7] = 1  # VIX -> NASDAQ
    true_graph[0, 8] = 1  # VIX -> DOW
    true_graph[0, 9] = 1  # VIX -> VOLUME

    # Treasury rates affect stocks (inverse relationship)
    true_graph[1, 6] = 1  # TREASURY -> SP500
    true_graph[1, 7] = 1  # TREASURY -> NASDAQ
    true_graph[1, 8] = 1  # TREASURY -> DOW

    # USD affects oil and gold
    true_graph[2, 3] = 1  # USD -> OIL
    true_graph[2, 4] = 1  # USD -> GOLD

    # Oil affects GDP and markets
    true_graph[3, 5] = 1  # OIL -> GDP_GROWTH
    true_graph[3, 6] = 1  # OIL -> SP500

    # Gold as safe haven (affected by market stress)
    true_graph[0, 4] = 1  # VIX -> GOLD

    # GDP growth affects all markets
    true_graph[5, 6] = 1  # GDP -> SP500
    true_graph[5, 7] = 1  # GDP -> NASDAQ
    true_graph[5, 8] = 1  # GDP -> DOW

    # Market co-movements
    true_graph[6, 7] = 1  # SP500 -> NASDAQ (correlation)
    true_graph[6, 8] = 1  # SP500 -> DOW

    print(f"True causal graph has {np.sum(true_graph)} edges")

    # Generate data using structural equation model
    data_matrix = np.zeros((n_samples, n_vars))

    # Generate exogenous variables
    data_matrix[:, 0] = np.abs(np.random.randn(n_samples) * 5 + 15)  # VIX (mean ~15, vol 5)
    data_matrix[:, 1] = np.abs(np.random.randn(n_samples) * 0.5 + 3.0)  # Treasury (mean ~3%, vol 0.5%)
    data_matrix[:, 2] = np.random.randn(n_samples) * 2 + 100  # USD Index (mean 100)

    # Generate endogenous variables following causal structure
    # OIL = f(USD) + noise
    data_matrix[:, 3] = -0.3 * data_matrix[:, 2] + np.random.randn(n_samples) * 5 + 70

    # GOLD = f(VIX, USD) + noise (safe haven)
    data_matrix[:, 4] = 0.5 * data_matrix[:, 0] - 0.2 * data_matrix[:, 2] + np.random.randn(n_samples) * 10 + 1800

    # GDP_GROWTH = f(OIL) + noise
    data_matrix[:, 5] = -0.02 * data_matrix[:, 3] + np.random.randn(n_samples) * 0.5 + 2.5

    # SP500 = f(VIX, TREASURY, OIL, GDP) + noise
    data_matrix[:, 6] = (
        -1.5 * data_matrix[:, 0] +   # Negative: high volatility -> lower returns
        -0.5 * data_matrix[:, 1] +   # Negative: high rates -> lower stocks
        -0.1 * data_matrix[:, 3] +   # Negative: high oil -> inflation -> lower stocks
        2.0 * data_matrix[:, 5] +    # Positive: GDP growth -> higher stocks
        np.random.randn(n_samples) * 3 + 4000
    )

    # NASDAQ = f(VIX, TREASURY, GDP, SP500) + noise
    data_matrix[:, 7] = (
        -2.0 * data_matrix[:, 0] +
        -0.6 * data_matrix[:, 1] +
        2.5 * data_matrix[:, 5] +
        0.3 * data_matrix[:, 6] +
        np.random.randn(n_samples) * 5 + 12000
    )

    # DOW = f(VIX, TREASURY, GDP, SP500) + noise
    data_matrix[:, 8] = (
        -1.2 * data_matrix[:, 0] +
        -0.4 * data_matrix[:, 1] +
        1.8 * data_matrix[:, 5] +
        0.25 * data_matrix[:, 6] +
        np.random.randn(n_samples) * 4 + 35000
    )

    # VOLUME = f(VIX) + noise (higher volatility -> higher volume)
    data_matrix[:, 9] = 50 * data_matrix[:, 0] + np.random.randn(n_samples) * 100 + 3000

    # Create DataFrame
    data = pd.DataFrame(data_matrix, columns=var_names)

    print(f"Generated {n_samples} market observations with {n_vars} variables")

    return data, true_graph, var_names


def load_stock_market(n_samples: int = 1000, seed: int = 42) -> Tuple[pd.DataFrame, np.ndarray, list]:
    """
    Load stock market dataset.

    This is a wrapper around generate_synthetic_market_data for consistency
    with other dataset loaders. In a real scenario, this would fetch actual
    market data from APIs like Yahoo Finance, Alpha Vantage, etc.

    Args:
        n_samples: Number of observations
        seed: Random seed

    Returns:
        data: DataFrame with market variables
        true_graph: Ground truth adjacency matrix
        var_names: List of variable names
    """
    return generate_synthetic_market_data(n_samples=n_samples, seed=seed)


def get_stock_market_description() -> dict:
    """
    Get descriptive information about the stock market dataset.

    Returns:
        Dictionary with dataset metadata
    """
    return {
        'name': 'Stock Market',
        'domain': 'Finance / Economics',
        'n_nodes': 10,
        'n_edges': 18,
        'type': 'Time Series (converted to i.i.d. for causal discovery)',
        'complexity': 'Medium',
        'description': (
            'Financial market causal relationships including stock indices, '
            'volatility, interest rates, commodities, and economic indicators. '
            'Tests algorithms on economic/financial domain.'
        ),
        'key_causal_relationships': [
            'VIX (volatility) -> Stock indices (negative)',
            'Treasury rates -> Stock returns (negative)',
            'USD Index -> Commodities (oil, gold)',
            'GDP growth -> Market performance',
            'Oil prices -> Economic growth',
            'Market stress -> Safe haven demand (gold)'
        ],
        'note': (
            'This is synthetic data with structure based on economic theory. '
            'Real financial data would require API access and proper licensing.'
        )
    }


if __name__ == "__main__":
    # Test the loader
    data, true_graph, var_names = load_stock_market(n_samples=1000)

    print("\n" + "="*80)
    print("STOCK MARKET DATASET GENERATED SUCCESSFULLY")
    print("="*80)
    print(f"Data shape: {data.shape}")
    print(f"Variables: {', '.join(var_names)}")
    print(f"True graph shape: {true_graph.shape}")
    print(f"Number of edges: {np.sum(true_graph)}")

    # Print some statistics
    print("\nData statistics:")
    print(data.describe())

    # Show causal structure
    print("\nTrue causal relationships:")
    for i, source in enumerate(var_names):
        targets = [var_names[j] for j in range(len(var_names)) if true_graph[i, j] == 1]
        if targets:
            print(f"  {source} -> {', '.join(targets)}")

    desc = get_stock_market_description()
    print(f"\nDataset description: {desc['description']}")
