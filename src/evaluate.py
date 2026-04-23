"""
Evaluation pipeline computing five metrics across all generated audio.

Metrics
-------
- raw_clap             : CLAP similarity against the strategy's full prompt
- semantic_clap        : CLAP similarity against a canonical scene description,
                         controlling for prompt-length effects
- consistency          : cosine distance on mel spectrograms within same scene,
                         across seeds (lower = more consistent)
- diversity            : cosine distance on mel spectrograms across different
                         scenes within same strategy (higher = more varied)
- tempo_accuracy       : |detected BPM - target BPM|; reported as mean AND
                         median because librosa's beat tracker can produce
                         outlier readings on sparse/free-time audio

These are analogs of the Challenge 1 metrics (CLIP, LPIPS, FID), adapted for
the audio modality.
"""

import json
import random
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.spatial.distance import cosine

from .scene_spec import SceneSpec
from .prompts import semantic_description


# --------------------------------------------------------------------------
# Audio loading utility
# --------------------------------------------------------------------------
def load_audio_dict(manifest: List[dict], audio_dir: Path,
                    sample_rate: int = 32000) -> Dict[str, np.ndarray]:
    """Load every audio clip referenced in the manifest into memory."""
    audios = {}
    for entry in manifest:
        y, _ = librosa.load(audio_dir / entry["audio_filename"], sr=sample_rate)
        audios[entry["audio_filename"]] = y
    return audios


def group_by_strategy(manifest: List[dict]) -> Dict[str, List[dict]]:
    """Return {strategy_name: [manifest_entry, ...]}."""
    out: Dict[str, List[dict]] = {}
    for entry in manifest:
        out.setdefault(entry["strategy"], []).append(entry)
    return out


# --------------------------------------------------------------------------
# CLAP scoring — audio-text joint embedding (audio analog of CLIP)
# --------------------------------------------------------------------------
def load_clap_model(device: str = "cuda"):
    """Load LAION-CLAP with the default music checkpoint."""
    import laion_clap
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    model.eval()
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    return model


@torch.no_grad()
def clap_score(
    clap_model, audio_array: np.ndarray, text: str,
    source_sample_rate: int = 32000, device: str = "cuda",
) -> float:
    """Cosine similarity between CLAP audio and text embeddings, scaled to ~[0, 100].

    CLAP expects 48 kHz audio, so we resample from MusicGen's 32 kHz output.
    """
    if source_sample_rate != 48000:
        audio_48k = librosa.resample(
            audio_array, orig_sr=source_sample_rate, target_sr=48000,
        )
    else:
        audio_48k = audio_array

    audio_tensor = torch.from_numpy(audio_48k).float().unsqueeze(0)

    audio_emb = clap_model.get_audio_embedding_from_data(x=audio_tensor, use_tensor=True)
    text_emb = clap_model.get_text_embedding([text], use_tensor=True)

    audio_emb = audio_emb / audio_emb.norm(dim=-1, keepdim=True)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    return (audio_emb @ text_emb.T).item() * 100


def compute_raw_clap(clap_model, manifest: List[dict],
                     audios: Dict[str, np.ndarray]) -> dict:
    """Per-strategy raw CLAP: score each clip against its own prompt."""
    by_strat = group_by_strategy(manifest)
    results = {}
    for strat, entries in by_strat.items():
        scores = [
            clap_score(clap_model, audios[e["audio_filename"]], e["prompt"])
            for e in entries
        ]
        results[strat] = {
            "mean": float(np.mean(scores)),
            "std":  float(np.std(scores)),
            "n":    len(scores),
            "all_scores": scores,
        }
    return results


def compute_semantic_clap(clap_model, manifest: List[dict],
                          audios: Dict[str, np.ndarray],
                          catalog: List[SceneSpec]) -> dict:
    """Per-strategy semantic CLAP: score each clip against its scene's canonical description.

    Controls for prompt-length effects by using the same reference text per scene
    regardless of strategy.
    """
    by_strat = group_by_strategy(manifest)
    semantic_by_idx = {i: semantic_description(s) for i, s in enumerate(catalog)}

    results = {}
    for strat, entries in by_strat.items():
        scores = []
        for e in entries:
            ref = semantic_by_idx[e["spec_idx"]]
            scores.append(clap_score(clap_model, audios[e["audio_filename"]], ref))
        results[strat] = {
            "mean": float(np.mean(scores)),
            "std":  float(np.std(scores)),
            "n":    len(scores),
            "all_scores": scores,
        }
    return results


# --------------------------------------------------------------------------
# Mel spectrogram distance — consistency and diversity
# --------------------------------------------------------------------------
def _mel_spec_flat(audio: np.ndarray, sample_rate: int = 32000) -> np.ndarray:
    """Flatten a mel spectrogram to a 1-D vector for distance computation."""
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128, fmax=8000)
    return librosa.power_to_db(mel, ref=np.max).flatten()


def compute_consistency(manifest: List[dict],
                        audios: Dict[str, np.ndarray]) -> dict:
    """Within-scene across-seed pairs. Lower = more consistent."""
    by_strat = group_by_strategy(manifest)
    results = {}
    for strat, entries in by_strat.items():
        by_scene: Dict[int, List[dict]] = {}
        for e in entries:
            by_scene.setdefault(e["spec_idx"], []).append(e)

        distances = []
        for scene_entries in by_scene.values():
            specs = {
                e["audio_filename"]: _mel_spec_flat(audios[e["audio_filename"]])
                for e in scene_entries
            }
            for fa, fb in combinations(specs.keys(), 2):
                va, vb = specs[fa], specs[fb]
                L = min(len(va), len(vb))
                distances.append(cosine(va[:L], vb[:L]))

        results[strat] = {
            "mean":    float(np.mean(distances)),
            "std":     float(np.std(distances)),
            "n_pairs": len(distances),
        }
    return results


def compute_diversity(manifest: List[dict],
                      audios: Dict[str, np.ndarray],
                      n_pairs: int = 30, seed: int = 0) -> dict:
    """Across-scene pairs within a strategy. Higher = more diverse.

    Samples n_pairs random different-scene pairs per strategy (with fixed seed
    for reproducibility).
    """
    rng = random.Random(seed)
    by_strat = group_by_strategy(manifest)
    results = {}
    for strat, entries in by_strat.items():
        distances = []
        attempts = 0
        while len(distances) < n_pairs and attempts < n_pairs * 5:
            attempts += 1
            e1, e2 = rng.sample(entries, 2)
            if e1["spec_idx"] == e2["spec_idx"]:
                continue
            v1 = _mel_spec_flat(audios[e1["audio_filename"]])
            v2 = _mel_spec_flat(audios[e2["audio_filename"]])
            L = min(len(v1), len(v2))
            distances.append(cosine(v1[:L], v2[:L]))
        results[strat] = {
            "mean":    float(np.mean(distances)),
            "std":     float(np.std(distances)),
            "n_pairs": len(distances),
        }
    return results


# --------------------------------------------------------------------------
# Tempo accuracy
# --------------------------------------------------------------------------
def compute_tempo_accuracy(manifest: List[dict]) -> dict:
    """Per-strategy |detected BPM - target BPM|.

    Reports BOTH mean and median because librosa's beat tracker can produce
    outlier readings on sparse/slow/free-time audio (most notably the
    cinematic_suspense scene). Median is robust to those outliers.
    """
    df = pd.DataFrame(manifest)
    df["bpm_error"] = (df["detected_bpm"] - df["target_bpm"]).abs()

    results = {}
    for strat in df["strategy"].unique():
        errs = df[df["strategy"] == strat]["bpm_error"].values
        results[strat] = {
            "mean_error":   float(np.mean(errs)),
            "median_error": float(np.median(errs)),
            "std_error":    float(np.std(errs)),
            "max_error":    float(np.max(errs)),
            "n":            len(errs),
        }
    return results


# --------------------------------------------------------------------------
# Convenience: run everything, return one nested dict
# --------------------------------------------------------------------------
def evaluate_all(manifest_path: Path, generations_dir: Path,
                 catalog: List[SceneSpec], device: str = "cuda") -> dict:
    """Run every metric and return a single nested dict."""
    manifest = json.loads(Path(manifest_path).read_text())
    audio_dir = Path(generations_dir) / "audio"
    audios = load_audio_dict(manifest, audio_dir)

    clap_model = load_clap_model(device=device)

    return {
        "raw_clap":       compute_raw_clap(clap_model, manifest, audios),
        "semantic_clap":  compute_semantic_clap(clap_model, manifest, audios, catalog),
        "consistency":    compute_consistency(manifest, audios),
        "diversity":      compute_diversity(manifest, audios),
        "tempo_accuracy": compute_tempo_accuracy(manifest),
    }
