"""A portable, model-agnostic deep-research agent.

Public surface:
    from deep_research_agent import make_graph, ResearchConfig
"""

from .agent import make_graph
from .config import ResearchConfig

__all__ = ["make_graph", "ResearchConfig"]
