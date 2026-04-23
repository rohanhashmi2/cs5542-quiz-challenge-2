# Failure Analysis

This document catalogs specific failure modes observed in the generations
and attributes each to a likely cause. Failure analysis produces the
analytical insight that distinguishes a working system from an understood one.

Five failure modes are cataloged below, spanning model-level (output quality),
measurement-level (metric reliability), and methodology-level (cross-modality
transfer of findings).

---

## Failure 1 — Naive prompt collapse on niche genres

**Example**: `scene03_naive_seed{42,123,2024}.wav` — chiptune_platformer, naive prompt

**Observation**: All three seeds for "8-bit chiptune" under the naive prompt
score near-zero on raw CLAP (6.68, 8.62, 4.03 — mean 6.44). Audibly, the
clips don't sound like chiptune. Detected tempos also miss — 101 BPM when
the target is 150 BPM for two of three seeds.

**Attribution**: MusicGen Small's training corpus is weighted toward
popular-music captions scraped from the web. Chiptune/NES music is
a niche genre with sparse representation. Without the structural anchors
provided by the structured prompt (`square wave lead`, `triangle wave
bass`, `noise channel drums`, `NES-style`), the model has no reliable
prior to fall back on and produces generic electronic output that fails
to match the requested genre.

**Mitigation**: Under the structured strategy, CLAP scores for chiptune
jump dramatically and tempo accuracy sharpens — strategy 3 (structured +
guidance) pushes this further. **This is the Challenge 2 equivalent of the
Japandi finding from Challenge 1**: prompt engineering impact is inversely
correlated with training-data familiarity. Common genres need less
scaffolding than rare ones.

---

## Failure 2 — Beat tracker failure on sparse cinematic audio

**Example**: `scene01_structured_seed123.wav` — cinematic_suspense, structured prompt

**Observation**: Librosa's beat tracker reports 250 BPM when the target
was 90 BPM — a 160-point error. Additional outliers appear on other
cinematic_suspense clips: 178.6 BPM (exactly 2× the target) on two
structured+guidance seeds.

**Attribution**: This is almost certainly a **measurement failure, not a
generation failure**. Cinematic suspense music is slow, sparse, and
rhythmically ambiguous — long string tremolos, pulsing timpani without a
locked groove. Librosa's beat tracker uses onset detection and tempogram
analysis that implicitly assumes a locked, drum-driven pulse. When no
such pulse exists, it locks onto sixteenth-note subdivisions or simply
guesses wildly, producing near-exact doublings of the true tempo.

**Mitigation / lesson**: Report both **mean** and **median** tempo error.
The median is robust to these outliers and reflects typical performance.
For structured+guidance, median tempo error is **1.2 BPM** while mean is
20.82 BPM — the mean is dominated by the cinematic_suspense outliers.
Future work would use a neural beat tracker (like beat-this or TCN-based
methods) which handle sparse textures better than onset-based approaches.

---

## Failure 3 — Raw CLAP bias differs from raw CLIP bias

**Observation**: In Challenge 1, raw CLIP favored the naive prompt
(30.80 vs 28.08 for structured) — a length bias that required introducing
semantic CLIP to correct. In Challenge 2, raw CLAP showed the *opposite*
pattern: structured won by a large margin on raw CLAP (41.65 vs 19.78).

**Attribution**: CLIP and CLAP inherit biases from different training
distributions. CLIP was trained on mixed-length web image captions.
CLAP was trained on music-captioning datasets where detailed descriptions
(genre tags, instrument lists, BPM mentions) are the norm. CLAP therefore
rewards prompt detail rather than penalizing prompt length.

**Mitigation / lesson**: The bias doesn't disappear — semantic CLAP still
narrows the gap (from 28 points on raw to 17 points on semantic),
indicating that CLAP is mildly favoring longer prompts even when measured
against a fair reference. But the *direction* of the effect is opposite
to CLIP. **This is a generalization of Challenge 1's single-metric
lesson**: evaluation metrics inherit biases from their training data,
and those biases do not transfer cleanly across modalities. Reporting
multiple alignment metrics is essential for both image and audio
generation, but which biases you are guarding against differs.

---

## Failure 4 — No true negative prompt mechanism in MusicGen

**Observation**: MusicGen does not accept negative prompts the way Stable
Diffusion does. Attempting to pass a `negative_prompt` argument to
`model.generate()` raises warnings and is silently ignored.

**Attribution**: MusicGen is an autoregressive transformer built on
EnCodec tokens, not a denoising diffusion model. Negative prompts in
Stable Diffusion work by conditioning the noise prediction at each
denoising step toward the positive prompt *and away* from the negative
one. This mechanism has no natural analog in token-level autoregressive
generation.

**Mitigation / framing**: The Challenge 2 "structured + guidance" strategy
approximates the intent of a negative prompt using two mechanisms MusicGen
does support:

1. **Classifier-free guidance scale** raised from 3.0 to 5.0 — pushes
   generation more strictly toward the prompt distribution and away
   from unconditional output
2. **Quality-floor tokens** appended to the positive prompt — phrases
   like "clean mix, no distortion, no vocals, no clipping" that act as
   inverse-negatives by pulling toward the intended sonic space

This is **itself a finding**: control methodology from Challenge 1 did
not transfer mechanically to Challenge 2 because the underlying
architectures differ. Replicating "structured + negative" across
modalities required re-deriving the appropriate control levers.

---

## Failure 5 — Reproducibility caveat for CLAP

**Observation**: Running the full evaluation twice on seed-pinned
generations produced bit-for-bit identical tempo, consistency, and
diversity metrics — but raw and semantic CLAP scores varied by
up to 0.9 points between runs (≤2% of the measured values).

**Attribution**: CLAP's model has minor non-determinism in its CUDA
operations. This produces tiny run-to-run variance in both the audio
embedding and text embedding stages, which amplify slightly in the
cosine similarity computation.

**Mitigation / lesson**: Findings are stable despite this noise —
the ranking of strategies never changes, and the gap between strategies
(27+ points between naive and structured+guidance on raw CLAP) is an
order of magnitude larger than the run-to-run variance. A rigorous
future study would average CLAP scores over multiple runs and report
confidence intervals. For this project, the single-run numbers are
reported with the reproducibility caveat noted.
