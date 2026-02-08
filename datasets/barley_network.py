#!/usr/bin/env python3
"""
Barley Network Dataset
=======================

48-node Bayesian network for barley crop production.
Domain: Agriculture / Botany
Source: pgmpy benchmark networks

Models relationships between soil conditions, weather,
management practices, and crop yield outcomes.
Tests algorithm scalability on large graphs.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_barley(n_samples: int = 3000):
    """
    Load Barley network from bnlearn with real data only.

    Args:
        n_samples: Ignored - uses actual dataset size

    Returns:
        data: pd.DataFrame with numeric data
        true_graph: np.ndarray adjacency matrix
        node_names: list of variable names
    """
    try:
        import bnlearn as bn
    except ImportError:
        print("Installing bnlearn...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'bnlearn'])
        import bnlearn as bn

    # Load the DAG and get real data
    dag = bn.import_DAG('barley')
    
    # Get the real data
    real_data = bn.import_example('barley')
    if real_data is not None and hasattr(real_data, 'keys') and 'df' in real_data:
        data = real_data['df']
        print(f"Using real Barley dataset with {len(data)} samples")
    elif real_data is not None and isinstance(real_data, pd.DataFrame):
        data = real_data
        print(f"Using real Barley dataset with {len(data)} samples")
    else:
        raise ValueError("No real Barley dataset available in bnlearn")

    # Extract ground truth graph
    nodes = sorted(dag['model'].nodes())
    n = len(nodes)
    true_graph = np.zeros((n, n))
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    for edge in dag['model'].edges():
        i = node_to_idx[edge[0]]
        j = node_to_idx[edge[1]]
        true_graph[i, j] = 1

    # Ensure data has the right column order
    data = data[nodes] if isinstance(data, pd.DataFrame) else pd.DataFrame(data, columns=nodes)

    # Encode categorical columns to numeric
    for col in data.columns:
        if data[col].dtype == 'object' or str(data[col].dtype) == 'category':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))

    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    data = data.astype(np.float64)

    print(f"Barley network: {n} nodes, {int(np.sum(true_graph))} edges, {len(data)} samples")

    return data, true_graph, nodes


if __name__ == "__main__":
    data, graph, names = load_barley()
    print(f"Variables: {names}")
    print(f"Shape: {data.shape}")
    print(f"Edges: {int(np.sum(graph))}")
