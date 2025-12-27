"""Configuration placeholders for retrieval approaches.

This module intentionally stays minimal: it provides a structured place to keep
approach parameters without forcing a heavyweight config system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class ApproachConfig:
    """Generic approach config container."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    candidates_depth: int | None = None


def default_approach2_config() -> ApproachConfig:
    """Default (placeholder) config for Approach 2."""
    return ApproachConfig(
        name="approach2_template",
        params={
            # Put approach-specific knobs here (placeholder).
            # Example: "some_param": 123
        },
        candidates_depth=None,
    )


def default_approach3_config() -> ApproachConfig:
    """Default (placeholder) config for Approach 3."""
    return ApproachConfig(
        name="approach3_template",
        params={
            # Put approach-specific knobs here (placeholder).
            # Example: "rrf_k": 60
        },
        # For multi-stage methods, candidates_depth is often > topk (e.g., 2000-5000).
        candidates_depth=2000,
    )


