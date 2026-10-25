"""Compatibility import; grid-search helpers live in void_search.py."""

from .void_search import find_local_minima, cluster_minima, clustered_local_minima

__all__ = ["find_local_minima", "cluster_minima", "clustered_local_minima"]
