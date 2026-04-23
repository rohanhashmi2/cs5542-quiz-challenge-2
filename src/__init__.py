"""CS 5542 Quiz Challenge 2 — Controlled Music / Sound Generation with MusicGen."""

from .scene_spec import SceneSpec, CATALOG
from .prompts import (
    naive_prompt,
    structured_prompt,
    structured_plus_guidance_prompt,
    semantic_description,
    STRATEGIES,
)

__all__ = [
    "SceneSpec",
    "CATALOG",
    "naive_prompt",
    "structured_prompt",
    "structured_plus_guidance_prompt",
    "semantic_description",
    "STRATEGIES",
]
