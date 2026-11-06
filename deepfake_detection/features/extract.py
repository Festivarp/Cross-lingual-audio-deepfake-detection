# -*- coding: utf-8 -*-
"""MFCC / LFCC / CQCC extraction.

Transform definitions and the CQCC computation (CQT -> dB -> DCT) are reused
verbatim from `mentor_demo_features.ipynb`, which in turn reuses them from the
original exploratory notebook. Parameters: 16 kHz, n_fft=400 (25 ms),
hop_length=160 (10 ms), 23 mel filters, 13 coefficients per family,
center=False.

Provenance of the parameters
----------------------------
The five-step cepstral pipeline (framing/windowing -> filterbank -> log -> DCT)
follows Tahaoglu et al. [11], PDF pp. 4-6. The *specific* values (400/160/23/13)
are NOT stated in [11]; they are standard speech-processing defaults carried
over from this project's own notebooks, and are documented as such in the paper
(Section II-C, "Status of the 25 ms / 10 ms framing: PARTIALLY SOURCED").
"""
import io

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy.fftpack import dct

TARGET_SR = 16000
N_FFT = 400          # 25 ms
HOP_LENGTH = 160     # 10 ms
N_MELS = 23
N_COEFF = 13

CLIP_SECONDS = 4.0
TARGET_LEN = int(CLIP_SECONDS * TARGET_SR)     # 64,000 samples

# With center=False: n_frames = floor((64000 - 400)/160) + 1 = 398
N_FRAMES = 398

mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=TARGET_SR,
    n_mfcc=N_COEFF,
    melkwargs={"n_fft": N_FFT, "hop_length": HOP_LENGTH, "n_mels": N_MELS,
               "center": False},
)

lfcc_transform = torchaudio.transforms.LFCC(
    sample_rate=TARGET_SR,
    n_lfcc=N_COEFF,
    speckwargs={"n_fft": N_FFT, "hop_length": HOP_LENGTH, "center": False},
)


def _sf_read(source):
    """Decode a path or in-memory buffer with soundfile -> (waveform[C, N], sr)."""
    y, sr = sf.read(source, dtype="float32", always_2d=True)
    return torch.tensor(y.T), sr


def load_waveform(item, target_sr=TARGET_SR):
    """Accepts a filepath (str), an undecoded HF audio dict {'bytes','path'},
    a legacy HF audio dict {'array','sampling_rate'}, raw bytes, or a torchcodec
    AudioDecoder. Returns a mono waveform at `target_sr`.

    The soundfile branches exist because `datasets` >= 4.0 and `torchaudio` >= 2.9
    both route audio decoding through torchcodec, which needs FFmpeg shared
    libraries that are not present in every environment.
    """
    if isinstance(item, (bytes, bytearray)):
        waveform, sr = _sf_read(io.BytesIO(item))
    elif isinstance(item, str):
        try:
            waveform, sr = torchaudio.load(item)
        except Exception:
            waveform, sr = _sf_read(item)
    elif isinstance(item, dict) and item.get("bytes") is not None:
        waveform, sr = _sf_read(io.BytesIO(item["bytes"]))
    elif isinstance(item, dict) and "array" in item:
        waveform = torch.tensor(np.asarray(item["array"]), dtype=torch.float32).unsqueeze(0)
        sr = item["sampling_rate"]
    elif hasattr(item, "get_all_samples"):
        samples = item.get_all_samples()
        waveform = samples.data
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(np.asarray(waveform), dtype=torch.float32)
        waveform = waveform.float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        sr = samples.sample_rate
    else:
        raise TypeError(f"Unsupported clip item type: {type(item)}")

    if waveform.shape[0] > 1:                      # stereo -> mono
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
    return waveform


def fix_duration(waveform, target_len=TARGET_LEN):
    """Crop from the start, or zero-pad at the end, to exactly `target_len`."""
    n = waveform.shape[-1]
    if n > target_len:
        return waveform[..., :target_len]
    if n < target_len:
        return torch.nn.functional.pad(waveform, (0, target_len - n))
    return waveform


def extract_one(family, item):
    """Extract a SINGLE family: 'mfcc', 'lfcc' or 'cqcc'. Returns float32 (13, 398).

    `extract_features` computes all three together, which is the right thing when
    every branch will be used. The cascade (deepfake_detection/cascade.py) does
    not know in advance which families it will need, and CQCC costs ~42x MFCC to
    compute, so it extracts them one at a time and stops as soon as a verdict is
    confident. This function exists for that path.
    """
    waveform = item if isinstance(item, torch.Tensor) else fix_duration(load_waveform(item))
    if waveform.shape[-1] != TARGET_LEN:
        waveform = fix_duration(waveform)

    if family == "mfcc":
        out = mfcc_transform(waveform).squeeze(0).numpy()
    elif family == "lfcc":
        out = lfcc_transform(waveform).squeeze(0).numpy()
    elif family == "cqcc":
        waveform_np = waveform.squeeze().numpy()
        cqt = np.abs(librosa.cqt(waveform_np, sr=TARGET_SR, hop_length=HOP_LENGTH))
        log_cqt = librosa.amplitude_to_db(cqt, ref=np.max)
        out = dct(log_cqt, type=2, axis=0, norm="ortho")[:N_COEFF]
    else:
        raise ValueError(f"unknown family {family!r}")
    return out[:, :N_FRAMES].astype("float32")


def extract_features(item):
    """Returns (mfcc, lfcc, cqcc), each float32 (13, 398).

    CQCC natively yields 401 frames (1 + floor(64000/160)); it is cropped to 398
    so that all three branches receive an identically shaped tensor. That crop
    discards the final 30 ms of the CQCC stream only.
    """
    waveform = item if isinstance(item, torch.Tensor) else fix_duration(load_waveform(item))
    if waveform.shape[-1] != TARGET_LEN:
        waveform = fix_duration(waveform)

    mfcc = mfcc_transform(waveform).squeeze(0).numpy()
    lfcc = lfcc_transform(waveform).squeeze(0).numpy()

    waveform_np = waveform.squeeze().numpy()
    cqt = np.abs(librosa.cqt(waveform_np, sr=TARGET_SR, hop_length=HOP_LENGTH))
    log_cqt = librosa.amplitude_to_db(cqt, ref=np.max)
    cqcc = dct(log_cqt, type=2, axis=0, norm="ortho")[:N_COEFF]

    return (mfcc[:, :N_FRAMES].astype("float32"),
            lfcc[:, :N_FRAMES].astype("float32"),
            cqcc[:, :N_FRAMES].astype("float32"))
