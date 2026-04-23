"""
Three prompt generation strategies, progressively more controlled.

The experimental design holds the input (a SceneSpec) constant and varies
only the prompt strategy. This isolates the effect of prompt engineering
from any other source of variation.

Strategies
----------
- naive                        : minimal information baseline (genre only)
- structured                   : full metadata compiled via template
- structured_plus_guidance     : structured + higher CFG + quality-floor tokens

A critical note on the third strategy
-------------------------------------
MusicGen does NOT support negative prompts the way Stable Diffusion does.
It is an autoregressive transformer, not a denoising diffusion model, and
has no mechanism for explicit negative conditioning.

The closest control mechanisms MusicGen supports are:
  1. Classifier-free guidance (CFG) — higher scale = tighter prompt adherence
     (default 3.0; we use 5.0 for the guidance strategy)
  2. Quality-floor tokens appended to the positive prompt — phrases like
     "clean mix, no distortion, no vocals, no clipping" that act as
     inverse-negatives by pulling generation toward the intended sonic space

This is the Challenge 2 analog of Challenge 1's "structured + negative"
strategy, adapted for MusicGen's architecture. Same goal (tightest control),
different mechanism — which is itself a finding about how control transfers
across modalities.

All three functions return a (prompt, negative) tuple for interface parity
with Challenge 1. The negative slot is always empty string for MusicGen.
"""

from typing import Tuple

from .scene_spec import SceneSpec


def naive_prompt(spec: SceneSpec) -> Tuple[str, str]:
    """Baseline: minimal information — what a casual user would type."""
    return spec.genre, ""


def structured_prompt(spec: SceneSpec) -> Tuple[str, str]:
    """Full metadata compiled via template, with quality modifiers appended."""
    instruments_str = ", ".join(spec.instruments)
    descriptors_str = ", ".join(spec.key_descriptors)
    prompt = (
        f"{spec.genre}, "
        f"{spec.tempo_bpm} BPM, "
        f"{instruments_str}, "
        f"{spec.mood}, "
        f"{descriptors_str}, "
        f"professional studio production, high quality recording"
    )
    return prompt, ""


def structured_plus_guidance_prompt(spec: SceneSpec) -> Tuple[str, str]:
    """Structured + Guidance — MusicGen's architectural analog to negative prompts.

    Positive prompt adds quality-floor tokens ('clean mix, no distortion,
    no vocals, no clipping'). The paired guidance_scale=5.0 is applied at
    generation time by src.generate (see STRATEGY_GUIDANCE there).
    """
    positive, _ = structured_prompt(spec)
    positive += ", clean mix, no distortion, no vocals, no clipping"
    return positive, ""


# --------------------------------------------------------------------------
# Canonical semantic description — used for "fair CLAP" evaluation.
# Strips out prompt-engineering filler so every strategy is scored against
# the same scene concept, controlling for prompt-length effects.
# --------------------------------------------------------------------------
def semantic_description(spec: SceneSpec) -> str:
    """Return the canonical semantic content of the spec — no quality modifiers."""
    instruments_str = ", ".join(spec.instruments)
    return (
        f"{spec.genre} music at {spec.tempo_bpm} BPM with "
        f"{instruments_str}, {spec.mood}"
    )


# Strategy registry — matches the Colab notebook STRATEGIES dict
STRATEGIES = {
    "naive":               naive_prompt,
    "structured":          structured_prompt,
    "structured_guidance": structured_plus_guidance_prompt,
}
