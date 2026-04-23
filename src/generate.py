"""
Generation runner: for each (scene, strategy, seed) triple, produce a
10-second audio clip with MusicGen Small and save the WAV + spectrogram.

This module mirrors the notebook's Cell 9 logic, packaged as a reusable
Python API. Results from running this module on the same catalog, seeds,
and guidance scales should match the notebook output bit-for-bit on the
deterministic metrics (tempo, consistency, diversity).

Usage
-----
    from pathlib import Path
    from src.generate import load_pipeline, run_generation_suite
    from src.scene_spec import CATALOG
    from src.prompts import STRATEGIES

    model, processor = load_pipeline()
    manifest = run_generation_suite(
        model=model,
        processor=processor,
        catalog=CATALOG,
        strategies=STRATEGIES,
        seeds=[42, 123, 2024],
        output_dir=Path("generations"),
    )
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from tqdm.auto import tqdm

from .scene_spec import SceneSpec


MODEL_ID = "facebook/musicgen-small"
DEFAULT_SEEDS = [42, 123, 2024]

# MusicGen does NOT support negative prompts. The closest analog to the
# "structured + negative" strategy from Challenge 1 is to raise the
# classifier-free guidance scale for the guidance strategy.
STRATEGY_GUIDANCE = {
    "naive":               3.0,
    "structured":           3.0,
    "structured_guidance": 5.0,
}


def load_pipeline(model_id: str = MODEL_ID, device: str = "cuda"):
    """Load MusicGen Small in fp16 on the given device.

    Returns (model, processor) — both are needed for generation.
    """
    from transformers import MusicgenForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16,
    ).to(device)
    return model, processor


def generate_audio(
    model,
    processor,
    prompt: str,
    seed: int = None,
    guidance_scale: float = 3.0,
    max_new_tokens: int = 512,  # ~10 seconds at 32 kHz
    device: str = "cuda",
) -> np.ndarray:
    """Generate one audio clip from a text prompt."""
    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
    if seed is not None:
        torch.manual_seed(seed)
    with torch.no_grad():
        audio_values = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            guidance_scale=guidance_scale,
        )
    return audio_values[0, 0].cpu().float().numpy()


def save_audio(audio_array: np.ndarray, path: Path, sample_rate: int):
    """Save a 1-D float audio array as a WAV file."""
    sf.write(path, audio_array, sample_rate)


def plot_spectrogram(
    audio_array: np.ndarray,
    title: str,
    save_path: Path,
    sample_rate: int,
) -> None:
    """Render a mel spectrogram and save it as PNG. Does not display."""
    mel = librosa.feature.melspectrogram(
        y=audio_array, sr=sample_rate, n_mels=128, fmax=8000,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(12, 4))
    img = librosa.display.specshow(
        mel_db, sr=sample_rate, x_axis="time", y_axis="mel",
        fmax=8000, ax=ax, cmap="magma",
    )
    ax.set_title(title, fontsize=12)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_generation_suite(
    model,
    processor,
    catalog: List[SceneSpec],
    strategies: Dict[str, Callable[[SceneSpec], Tuple[str, str]]],
    seeds: List[int],
    output_dir: Path,
) -> List[dict]:
    """Generate every (scene, strategy, seed) combination.

    Writes WAVs + spectrograms + manifest.json. Returns the manifest.
    """
    output_dir = Path(output_dir)
    audio_dir = output_dir / "audio"
    spec_dir = output_dir / "spectrograms"
    audio_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = model.config.audio_encoder.sampling_rate

    manifest: List[dict] = []
    total = len(catalog) * len(strategies) * len(seeds)
    pbar = tqdm(total=total, desc="Generating")

    for spec_idx, scene in enumerate(catalog):
        for strat_name, strat_fn in strategies.items():
            prompt, _ = strat_fn(scene)
            guidance = STRATEGY_GUIDANCE[strat_name]

            for seed in seeds:
                stem = f"scene{spec_idx:02d}_{strat_name}_seed{seed}"

                audio = generate_audio(
                    model=model, processor=processor,
                    prompt=prompt, seed=seed, guidance_scale=guidance,
                )
                save_audio(audio, audio_dir / f"{stem}.wav", sample_rate)
                plot_spectrogram(
                    audio,
                    title=f"{scene.scene_id} · {strat_name} · seed={seed}",
                    save_path=spec_dir / f"{stem}.png",
                    sample_rate=sample_rate,
                )

                # Tempo detection — librosa's beat tracker can be unreliable on
                # sparse/free-time audio, which we honestly report in analysis.
                tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
                tempo_val = float(tempo) if np.isscalar(tempo) else float(tempo[0])

                manifest.append({
                    "spec_idx":       spec_idx,
                    "scene_id":       scene.scene_id,
                    "genre":          scene.genre,
                    "target_bpm":     scene.tempo_bpm,
                    "detected_bpm":   round(tempo_val, 1),
                    "strategy":       strat_name,
                    "seed":           seed,
                    "guidance_scale": guidance,
                    "prompt":         prompt,
                    "audio_filename": f"{stem}.wav",
                    "spec_filename":  f"{stem}.png",
                })
                pbar.update(1)

    pbar.close()

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest
