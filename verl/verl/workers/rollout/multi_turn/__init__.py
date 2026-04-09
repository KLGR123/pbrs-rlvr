"""
Multi-turn rollout utilities for IGPO (Information Gain-based Policy Optimization).

Modules
-------
info_gain   – IGRewardComputer: per-turn GT log-prob and IG reward computation.
generation  – MultiTurnRolloutManager: generic multi-turn rollout loop with
              pluggable ToolExecutor for retrieval and code-sandbox backends.
"""

from verl.workers.rollout.multi_turn.info_gain import (
    IGRewardComputer,
    IGRewardConfig,
)
from verl.workers.rollout.multi_turn.generation import (
    MultiTurnRolloutManager,
    RolloutConfig,
    ToolExecutor,
)

__all__ = [
    "IGRewardComputer",
    "IGRewardConfig",
    "MultiTurnRolloutManager",
    "RolloutConfig",
    "ToolExecutor",
]
