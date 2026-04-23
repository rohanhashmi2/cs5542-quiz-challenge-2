"""
Scene specification schema and the five-entry catalog used throughout the
CS 5542 Quiz Challenge 2 music/sound generation pipeline.

A SceneSpec captures the fields a real content-creator or game-audio platform
would receive from a user brief: genre, tempo, instrumentation, mood, etc.
The same spec feeds all three prompt strategies, so any difference in
generated output is attributable to prompt engineering rather than input.

Direct parallel to Challenge 1's RoomSpec — same pattern, audio domain.
"""

from dataclasses import dataclass, asdict
from typing import List


@dataclass
class SceneSpec:
    """Structured metadata describing an audio generation brief."""

    scene_id: str              # short identifier for filenames: "workout_pop"
    use_case: str              # "gym workout background playlist"
    genre: str                 # "upbeat electronic pop"
    tempo_bpm: int             # target BPM — used for the tempo accuracy metric
    instruments: List[str]     # ["driving four-on-the-floor drums", ...]
    mood: str                  # "high energy, motivational, uplifting"
    key_descriptors: List[str] # ["anthemic", "sidechained pumping bass", ...]

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Catalog of 5 diverse audio scenes.
# Chosen to span genres with different degrees of representation in MusicGen's
# training data — pop/lo-fi are well-represented, chiptune is niche. This
# spread surfaces the "prompt sensitivity varies by genre familiarity"
# finding (analog of Challenge 1's Japandi insight).
# --------------------------------------------------------------------------
CATALOG: List[SceneSpec] = [
    SceneSpec(
        scene_id="workout_pop",
        use_case="gym workout background playlist",
        genre="upbeat electronic pop",
        tempo_bpm=128,
        instruments=[
            "driving four-on-the-floor drums",
            "punchy synth bass",
            "bright lead synth",
        ],
        mood="high energy, motivational, uplifting",
        key_descriptors=[
            "anthemic",
            "sidechained pumping bass",
            "modern EDM production",
        ],
    ),
    SceneSpec(
        scene_id="cinematic_suspense",
        use_case="thriller movie trailer",
        genre="cinematic orchestral suspense",
        tempo_bpm=90,
        instruments=[
            "low string tremolos",
            "pulsing timpani",
            "subtle brass stabs",
            "metallic percussion",
        ],
        mood="tense, dark, unsettling",
        key_descriptors=[
            "slow building intensity",
            "minor key",
            "sparse texture",
            "film score",
        ],
    ),
    SceneSpec(
        scene_id="lofi_study",
        use_case="YouTube study-with-me background",
        genre="lo-fi hip-hop",
        tempo_bpm=75,
        instruments=[
            "dusty drum break",
            "warm Rhodes piano",
            "jazz bass",
            "vinyl crackle",
        ],
        mood="chill, nostalgic, calming",
        key_descriptors=[
            "mellow",
            "lowpass filtered",
            "laid-back swing",
            "mpc-style drums",
        ],
    ),
    SceneSpec(
        scene_id="chiptune_platformer",
        use_case="indie 2D platformer level theme",
        genre="8-bit chiptune",
        tempo_bpm=150,
        instruments=[
            "square wave lead",
            "triangle wave bass",
            "noise channel drums",
            "arpeggiated chords",
        ],
        mood="playful, adventurous, nostalgic",
        key_descriptors=[
            "NES-style",
            "rapid arpeggios",
            "looping chip music",
            "retro videogame",
        ],
    ),
    SceneSpec(
        scene_id="coffee_shop_folk",
        use_case="café ambience for a cozy vlog",
        genre="acoustic indie folk",
        tempo_bpm=95,
        instruments=[
            "fingerpicked acoustic guitar",
            "soft brush drums",
            "upright bass",
            "subtle mandolin",
        ],
        mood="warm, intimate, relaxed",
        key_descriptors=[
            "organic",
            "singer-songwriter style",
            "gentle sway",
        ],
    ),
]
