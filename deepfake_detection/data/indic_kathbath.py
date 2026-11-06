# -*- coding: utf-8 -*-
"""Speaker-matched Indic sets: IndicSynth (fake) + Kathbath (real, same speakers).

Why this replaces FLEURS
------------------------
`indic.py` pairs IndicSynth fakes with FLEURS real speech. Those are different
corpora (recording chain, speakers, read-speech style), so a detector can score
well -- or badly -- by telling the corpora apart rather than by detecting
synthesis. Phase 5's 0 % Indic EER is the symptom.

IndicSynth's own dataset card states that its bona fide source and target speech
comes from IndicSUPERB, i.e. the Kathbath corpus (AI4Bharat). We verified this
(`tools/kathbath_speaker_check.py`, `analysis/kathbath_check/speaker_check.json`):
IndicSynth's `Target Reference Audio` filenames occur verbatim in Kathbath, with
the speaker ID agreeing on 1,011/1,011 (Gujarati) and 435/435 (Hindi) matches,
and gender agreeing on every row checked.

So each fake here is paired with genuine Kathbath speech from the SAME speaker
the fake was cloned from: same corpus, same recording conditions, same voice.
The remaining difference between the two halves is synthesis itself -- the
same design ASVspoof uses.

Selection
---------
1. Fake candidates: the IndicSynth rows sampled at 40 random offsets by the
   speaker check, restricted to target speakers that have Kathbath clips.
2. Fakes are drawn round-robin across speakers, so no single voice dominates.
3. For each speaker with k fakes, k real Kathbath clips of that speaker are
   drawn. Recordings that IndicSynth used as a cloning reference are excluded,
   so no real clip is the literal source of a fake.
4. Every clip records its speaker, so Phase 4/5 splits can be speaker-disjoint.

Access
------
Metadata comes from the page cache written by the speaker check. Audio is
fetched clip by clip from the HF dataset-viewer (`/rows` -> cached asset URL):
Kathbath is served as lossless FLAC decoded from its original m4a, IndicSynth
as its original WAV. This downloads ~600 small files instead of multi-GB
parquet shards, and needs no FFmpeg. Downloaded bytes are kept under
`cache/raw_audio/` so a re-run does not re-download.

Kathbath is gated (auto-approved): accept the terms on its HF page and log in
with `hf auth login` first.

Known residual difference
-------------------------
Kathbath was recorded and distributed as AAC (m4a); IndicSynth outputs are
uncompressed WAV. The real half therefore carries lossy-codec traces the fake
half lacks. This is inherited from the corpora, and is far smaller than the
FLEURS/IndicSynth mismatch it replaces, but it is reported rather than hidden.
"""
import csv
import json
import os
import random
import time
from collections import Counter, defaultdict

import requests
from huggingface_hub import get_token

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAGES = os.path.join(ROOT, "analysis", "kathbath_check", "pages")
RAW = os.path.join(ROOT, "cache", "raw_audio")
API = "https://datasets-server.huggingface.co"

KATHBATH = "ai4bharat/Kathbath"
INDICSYNTH = "vdivyasharma/IndicSynth"
REAL, FAKE = 0, 1        # matches ASVspoof: 0 = bonafide/real, 1 = spoof/fake

ROWS_GAP = 3.0           # s between /rows calls; HF answers 429 above ~1/s sustained
ASSET_GAP = 0.25         # s between audio downloads

_session = requests.Session()
_last = {"rows": 0.0, "asset": 0.0}


def _headers():
    return {"Authorization": f"Bearer {get_token()}"}


def _throttle(kind, gap):
    time.sleep(max(0.0, _last[kind] + gap - time.time()))
    _last[kind] = time.time()


def _get(url, kind, gap, **params):
    wait = 60
    for _ in range(10):
        _throttle(kind, gap)
        try:
            r = _session.get(url, params=params or None, headers=_headers(), timeout=120)
        except requests.RequestException:
            time.sleep(5)
            continue
        if r.ok:
            return r
        if r.status_code == 429 or r.status_code >= 500:   # rate limit / transient server error
            print(f"    HTTP {r.status_code} from HF, waiting {wait}s", flush=True)
            time.sleep(wait)
            wait = min(wait * 2, 600)
            continue
        r.raise_for_status()
    raise RuntimeError(f"giving up on {url} {params}")


def _cached_pages(ds, cfg, split):
    """{page_offset: [metadata rows]} from the speaker-check page cache."""
    pre = f"{ds.replace('/', '_')}_{cfg}_{split}_"
    out = {}
    for f in os.listdir(PAGES):
        if f.startswith(pre) and f.endswith(".json"):
            with open(os.path.join(PAGES, f), encoding="utf-8") as fh:
                out[int(f[len(pre):-5])] = json.load(fh)
    return out


def _stem(fname):
    return os.path.splitext(fname)[0] if fname else None


def plan(language, n_per_class, seed=42):
    """Choose which clips to use, from metadata alone. No audio is fetched."""
    lang = language.lower()
    rng = random.Random(seed)

    # ---- real candidates: Kathbath rows we have metadata for, by speaker ----
    by_spk = defaultdict(list)
    for split in ("train", "valid"):
        for off, rows in _cached_pages(KATHBATH, lang, split).items():
            for i, r in enumerate(rows):
                by_spk[int(r["speaker_id"])].append(
                    {"split": split, "page": off, "idx": i, "fname": r["fname"],
                     "gender": r["gender"]})

    # ---- fake candidates: sampled IndicSynth rows whose speaker is covered ---
    fakes_by_spk, used_refs = defaultdict(list), set()
    for off, rows in _cached_pages(INDICSYNTH, language.capitalize(), "train").items():
        for i, r in enumerate(rows):
            for col in ("Target Reference Audio", "Source Reference Audio"):
                if r.get(col):
                    used_refs.add(_stem(r[col]))
            spk = r.get("Target Speaker ID")
            if spk is None or int(spk) not in by_spk:
                continue
            fakes_by_spk[int(spk)].append(
                {"split": "train", "page": off, "idx": i, "model": r["Generative Model"],
                 "gender": r["Gender"]})

    # a speaker is usable only if it has enough non-reference real clips
    for spk in list(fakes_by_spk):
        rng.shuffle(fakes_by_spk[spk])
        # real clips: random page order, random order within a page, so a
        # speaker's k clips come from as few pages (= API calls) as possible
        pages = defaultdict(list)
        for c in by_spk[spk]:
            if _stem(c["fname"]) not in used_refs:
                pages[(c["split"], c["page"])].append(c)
        order = list(pages)
        rng.shuffle(order)
        by_spk[spk] = []
        for key in order:
            rng.shuffle(pages[key])
            by_spk[spk] += pages[key]
    speakers = sorted(fakes_by_spk)
    rng.shuffle(speakers)

    # ---- round-robin over speakers, capped by each speaker's real supply ----
    take = Counter()
    while sum(take.values()) < n_per_class:
        progressed = False
        for spk in speakers:
            if sum(take.values()) >= n_per_class:
                break
            k = take[spk]
            if k < len(fakes_by_spk[spk]) and k < len(by_spk[spk]):
                take[spk] += 1
                progressed = True
        if not progressed:
            break

    fake_sel, real_sel = [], []
    for spk, k in take.items():
        for c in fakes_by_spk[spk][:k]:
            fake_sel.append({**c, "speaker": spk, "label": FAKE, "source": f"IndicSynth/{c['model']}"})
        for c in by_spk[spk][:k]:
            real_sel.append({**c, "speaker": spk, "label": REAL, "source": "Kathbath"})

    return {"language": language, "requested_per_class": n_per_class,
            "obtained_per_class": len(fake_sel), "speakers": len(take),
            "per_speaker": dict(take), "fake": fake_sel, "real": real_sel,
            "fake_models": dict(Counter(c["model"] for c in fake_sel))}


def _fetch(ds, cfg, sel, audio_col, ext, out_dir):
    """Download the audio for every selected clip; returns {key: bytes}."""
    os.makedirs(out_dir, exist_ok=True)
    pages = defaultdict(list)
    for c in sel:
        pages[(c["split"], c["page"])].append(c)

    got = {}
    for n, ((split, page), clips) in enumerate(sorted(pages.items()), 1):
        todo = [c for c in clips
                if not os.path.exists(os.path.join(out_dir, f"{split}_{page + c['idx']}{ext}"))]
        if todo:
            r = _get(f"{API}/rows", "rows", ROWS_GAP, dataset=ds, config=cfg,
                     split=split, offset=page, length=100)
            rows = r.json()["rows"]
            for c in todo:
                src = rows[c["idx"]]["row"][audio_col][0]["src"]
                blob = _get(src, "asset", ASSET_GAP).content
                with open(os.path.join(out_dir, f"{split}_{page + c['idx']}{ext}"), "wb") as fh:
                    fh.write(blob)
        for c in clips:
            path = os.path.join(out_dir, f"{split}_{page + c['idx']}{ext}")
            with open(path, "rb") as fh:
                got[id(c)] = fh.read()
        if n % 10 == 0 or n == len(pages):
            print(f"    {cfg}: {n}/{len(pages)} pages", flush=True)
    return got


def load_language(language, n_per_class, seed=42, verbose=True):
    """Speaker-matched real + fake clips for one language.

    Returns (records, plan). Each record is {'audio': bytes, 'label', 'speaker',
    'source'}; the plan describes what was selected.
    """
    lang = language.lower()
    p = plan(language, n_per_class, seed)
    if verbose:
        print(f"  plan: {p['obtained_per_class']} real + {p['obtained_per_class']} fake "
              f"from {p['speakers']} speakers; fake models {p['fake_models']}", flush=True)
    if p["obtained_per_class"] < n_per_class:
        print(f"  NOTE: only {p['obtained_per_class']} per class available from the "
              f"cached metadata (requested {n_per_class})", flush=True)

    t0 = time.time()
    fake_bytes = _fetch(INDICSYNTH, language.capitalize(), p["fake"], "audio", ".wav",
                        os.path.join(RAW, lang, "indicsynth"))
    real_bytes = _fetch(KATHBATH, lang, p["real"], "audio_filepath", ".flac",
                        os.path.join(RAW, lang, "kathbath"))
    if verbose:
        print(f"  audio fetched in {time.time() - t0:.0f}s", flush=True)

    records = [{"audio": real_bytes[id(c)], "label": REAL, "speaker": c["speaker"],
                "source": c["source"]} for c in p["real"]]
    records += [{"audio": fake_bytes[id(c)], "label": FAKE, "speaker": c["speaker"],
                 "source": c["source"]} for c in p["fake"]]

    # provenance: exactly which clips went into the set
    os.makedirs(os.path.join(RAW, lang), exist_ok=True)
    with open(os.path.join(RAW, lang, "selection.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "speaker", "source", "split", "row", "kathbath_fname"])
        for c in p["real"] + p["fake"]:
            w.writerow([c["label"], c["speaker"], c["source"], c["split"],
                        c["page"] + c["idx"], c.get("fname", "")])
    return records, p
