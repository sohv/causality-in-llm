"""
Additional Real-World Datasets for Causal Discovery
===================================================

This module provides loaders for additional real-world datasets
to strengthen the empirical coverage of the study.

Datasets:
1. Alarm Network - Medical diagnosis (37 nodes)
2. Stock Market - Financial causal relationships
"""

from .alarm_network import load_alarm
from .stock_market import load_stock_market

__all__ = ['load_alarm', 'load_stock_market']
