"""UL2 data utilities backed by `UL2_5`."""

from UL2_5.config import DenoiserSpec, Task, UL25Config

from .ul2_collator import (
    UL2DataCollator,
    create_collator_from_config,
    t5gemma2_config,
    ul2_recommended_config,
    ul2_recommended_with_curriculum_config,
)

__all__ = [
    "UL2DataCollator",
    "create_collator_from_config",
    "ul2_recommended_config",
    "ul2_recommended_with_curriculum_config",
    "t5gemma2_config",  # Deprecated, kept for backwards compat
    # UL2_5 configuration
    "UL25Config",
    "DenoiserSpec",
    "Task",
]
