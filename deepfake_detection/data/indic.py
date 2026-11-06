# -*- coding: utf-8 -*-
"""Indic target-domain loaders: IndicSynth (fake) + Google FLEURS (real).

Languages
---------
Gujarati : IndicSynth `name="Gujarati"` (118,778 utts) + FLEURS `gu_in` test (1,000)
Hindi    : IndicSynth `name="Hindi"`    (205,938 utts) + FLEURS `hi_in` test (418)

Hindi was specified in the original project proposal and had not previously been
implemented; it is added here following the Gujarati pattern exactly.

Corpus substitution
-------------------
The proposal specified the IIT-Mandi / AIR corpus for bonafide Hindi and
Gujarati speech. It is not publicly mirrored (access is by email request) and
was not obtained, so FLEURS is used instead. This is the same substitution the
paper documents, and it is a real limitation: FLEURS is read speech recorded
under controlled conditions, whereas AIR is broadcast material.

Both sources are opened with `streaming=True` so nothing is bulk-downloaded, and
the `Audio` column is cast to `decode=False` so decoding goes through soundfile
rather than torchcodec (which needs FFmpeg shared libraries).
"""
import itertools
import time

from datasets import Audio, load_dataset

LANGS = {
    "Gujarati": {"indicsynth": "Gujarati", "fleurs": "gu_in"},
    "Hindi":    {"indicsynth": "Hindi",    "fleurs": "hi_in"},
}

REAL, FAKE = 0, 1        # matches ASVspoof: 0 = bonafide/real, 1 = spoof/fake


def _take(stream, n, seed, buffer_size, label, verbose, tag):
    t0, out = time.time(), []
    shuffled = stream.shuffle(seed=seed, buffer_size=buffer_size)
    for ex in itertools.islice(shuffled, n):
        blob = ex["audio"]
        blob = blob["bytes"] if isinstance(blob, dict) else blob
        if not blob:
            continue
        out.append({"audio": blob, "label": label})
    if verbose:
        print(f"  {tag}: {len(out)} clips in {time.time() - t0:.0f}s", flush=True)
    return out


def load_fleurs_real(language, n, split="test", seed=42, buffer_size=40, verbose=True):
    """Genuine human speech for `language` from FLEURS (label 0)."""
    cfg = LANGS[language]["fleurs"]
    ds = load_dataset("google/fleurs", cfg, split=split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    return _take(ds, n, seed, buffer_size, REAL, verbose, f"FLEURS {cfg} real")


def load_indicsynth_fake(language, n, seed=42, buffer_size=40, verbose=True):
    """TTS-synthesised speech for `language` from IndicSynth (label 1)."""
    cfg = LANGS[language]["indicsynth"]
    ds = load_dataset("vdivyasharma/IndicSynth", name=cfg, split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    return _take(ds, n, seed, buffer_size, FAKE, verbose, f"IndicSynth {cfg} fake")


def load_language(language, n_real, n_fake, seed=42, verbose=True):
    """Both halves of one Indic language, as one list of {'audio','label'}."""
    real = load_fleurs_real(language, n_real, seed=seed, verbose=verbose)
    fake = load_indicsynth_fake(language, n_fake, seed=seed, verbose=verbose)
    return real + fake
