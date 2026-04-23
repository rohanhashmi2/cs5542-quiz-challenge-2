# Notebook ↔ Source Module Map

This document maps each function in `src/` to the Colab notebook cell it
was extracted from. The notebook (`quiz_challenge_2.ipynb`) is the source
of truth; the `src/` modules are a clean, importable reorganization of
the same logic.

## Why have both?

The notebook is what was actually executed to produce the results in
`results/`. The `src/` modules package that same logic into a reusable
Python API so future experiments don't require copying cells.

All logic is identical up to cosmetic refactoring (e.g., wrapping inline
model loading into a `load_pipeline()` function).

## Cell map

| `src/` module & function                      | Notebook cell | Status |
|------------------------------------------------|:-------------:|:------:|
| `scene_spec.SceneSpec` + `CATALOG`            | 11            | identical |
| `prompts.naive_prompt`                        | 13            | identical |
| `prompts.structured_prompt`                   | 13            | identical |
| `prompts.structured_plus_guidance_prompt`     | 13            | identical |
| `prompts.semantic_description`                | 13            | identical |
| `prompts.STRATEGIES`                          | 17            | identical |
| `generate.load_pipeline`                      | 5             | refactored inline → function |
| `generate.generate_audio`                     | 15            | identical |
| `generate.save_audio`                         | 9             | identical |
| `generate.plot_spectrogram`                   | 9             | identical |
| `generate.run_generation_suite`               | 17            | identical |
| `generate.STRATEGY_GUIDANCE`                  | 17            | identical |
| `evaluate.load_clap_model`                    | 22            | refactored inline → function |
| `evaluate.clap_score`                         | 24            | identical |
| `evaluate.compute_raw_clap`                   | 24            | identical |
| `evaluate.compute_semantic_clap`              | 26            | identical |
| `evaluate._mel_spec_flat`                     | 28            | identical |
| `evaluate.compute_consistency`                | 28            | identical |
| `evaluate.compute_diversity`                  | 30            | identical |
| `evaluate.compute_tempo_accuracy`             | 32            | identical |
| `evaluate.evaluate_all`                       | —             | new convenience wrapper |

## Key architectural note

Challenge 2 uses the `transformers` library (not `diffusers` as in
Challenge 1). MusicGen is an autoregressive transformer, so it loads
differently:

```python
# Challenge 1 (image diffusion)
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/...")

# Challenge 2 (audio transformer)
from transformers import MusicgenForConditionalGeneration, AutoProcessor
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
```

Generation also takes `max_new_tokens` (audio tokens) instead of
`num_inference_steps` (denoising steps). 512 tokens ≈ 10 seconds of
audio at 32 kHz.

## Reproducibility check

Running either the notebook or the `src/` modules on the same catalog,
seeds (`[42, 123, 2024]`), and guidance scales should produce:

- **Bit-for-bit identical**: tempo accuracy, consistency distance, diversity
  distance (seeded with `random.Random(0)` for the 30-pair diversity sample)
- **~2% variance** on raw and semantic CLAP: the CLAP model itself has minor
  CUDA non-determinism. Rankings are stable across runs.

This was verified by running the full pipeline twice during development —
see the reproducibility note in `docs/failure_analysis.md`.
