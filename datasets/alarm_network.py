#!/usr/bin/env python3
"""
Alarm Network Dataset Loader
=============================

The ALARM (A Logical Alarm Reduction Mechanism) network is a well-known
Bayesian network developed for monitoring patients in intensive care.

Network properties:
- 37 nodes (variables)
- 46 edges
- Medical domain (patient monitoring)
- Tests scalability to larger graphs

Variables represent physiological parameters, diseases, and medical measurements:
- Cardiovascular (CVP, PCWP, HR, CO)
- Respiratory (MinVol, VentMach, Disconnect)
- Metabolic (FiO2, PVSat, SAO2)
- And more...

Citation:
Beinlich, I., Suermondt, H. J., Chavez, R. M., & Cooper, G. F. (1989).
The ALARM monitoring system: A case study with two probabilistic inference
techniques for belief networks.
"""

import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import LabelEncoder


def load_alarm(n_samples: int = 5000, seed: int = 42) -> Tuple[pd.DataFrame, np.ndarray, list]:
    """
    Load the Alarm network dataset using bnlearn real data only.

    Args:
        n_samples: Ignored - uses actual dataset size
        seed: Ignored - uses real data

    Returns:
        data: DataFrame with real data
        true_graph: Ground truth adjacency matrix
        node_names: List of variable names
    """
    try:
        import bnlearn as bn
    except ImportError:
        print("Installing bnlearn for Alarm network...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'bnlearn'])
        import bnlearn as bn

    print("Loading Alarm network from bnlearn...")

    # Load the DAG and get real data
    dag = bn.import_DAG('alarm')
    
    # Get the real data
    real_data = bn.import_example('alarm')
    if real_data is not None and hasattr(real_data, 'keys') and 'df' in real_data:
        data = real_data['df']
        print(f"Using real Alarm dataset with {len(data)} samples")
    elif real_data is not None and isinstance(real_data, pd.DataFrame):
        data = real_data
        print(f"Using real Alarm dataset with {len(data)} samples")
    else:
        raise ValueError("No real Alarm dataset available in bnlearn")

    # Extract node names
    node_names = sorted(dag['model'].nodes())
    n_nodes = len(node_names)

    print(f"Alarm network: {n_nodes} nodes, {len(dag['model'].edges())} edges")

    # Create adjacency matrix from edges
    true_graph = np.zeros((n_nodes, n_nodes), dtype=int)
    node_to_idx = {node: i for i, node in enumerate(node_names)}

    for edge in dag['model'].edges():
        i = node_to_idx[edge[0]]
        j = node_to_idx[edge[1]]
        true_graph[i, j] = 1

    # Ensure data has the right column order
    data = data[node_names] if isinstance(data, pd.DataFrame) else pd.DataFrame(data, columns=node_names)

    # Encode categorical variables to numeric
    # Alarm network has discrete variables, encode them
    for col in data.columns:
        if data[col].dtype == 'object' or str(data[col].dtype) == 'category':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))

    # Convert all to float64 for numerical stability
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    data = data.astype(np.float64)

    print(f"Final dataset shape: {data.shape}")
    print(f"True graph has {np.sum(true_graph)} directed edges")

    return data, true_graph, node_names


def get_alarm_description() -> dict:
    """
    Get descriptive information about the Alarm network.

    Returns:
        Dictionary with network metadata
    """
    return {
        'name': 'Alarm',
        'domain': 'Medical (Intensive Care Monitoring)',
        'n_nodes': 37,
        'n_edges': 46,
        'type': 'Bayesian Network',
        'complexity': 'High (largest benchmark)',
        'description': (
            'ALARM network for patient monitoring in intensive care. '
            'Variables represent physiological parameters, diseases, and '
            'medical measurements. Tests algorithm scalability on larger graphs.'
        ),
        'key_variables': [
            'CVP (Central Venous Pressure)',
            'PCWP (Pulmonary Capillary Wedge Pressure)',
            'History (Patient History)',
            'MinVol (Minute Volume)',
            'TPR (Total Peripheral Resistance)',
            'Anaphylaxis',
            'Intubation',
            'VentMach (Ventilation Machine)',
            'Disconnect',
            'FiO2 (Fraction of Inspired Oxygen)'
        ],
        'citation': (
            'Beinlich et al. (1989). The ALARM monitoring system: '
            'A case study with two probabilistic inference techniques.'
        )
    }


if __name__ == "__main__":
    # Test the loader
    data, true_graph, node_names = load_alarm(n_samples=1000)

    print("\n" + "="*80)
    print("ALARM NETWORK LOADED SUCCESSFULLY")
    print("="*80)
    print(f"Data shape: {data.shape}")
    print(f"Variables: {', '.join(node_names[:10])}...")
    print(f"True graph shape: {true_graph.shape}")
    print(f"Number of edges: {np.sum(true_graph)}")

    # Print some statistics
    print("\nData statistics:")
    print(data.describe())

    desc = get_alarm_description()
    print(f"\nNetwork description: {desc['description']}")
