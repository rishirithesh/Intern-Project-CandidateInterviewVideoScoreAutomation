import wave
from typing import Dict

import numpy as np


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _energy_to_score(energy: float) -> float:
    return _clamp((energy - 0.02) / 0.3, 0.0, 1.0)


def _silence_to_score(silence_ratio: float) -> float:
    return _clamp(1.0 - silence_ratio * 1.75, 0.0, 1.0)


def analyze_audio(audio_path: str) -> Dict[str, float]:
    with wave.open(audio_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    dtype = None
    sample_width = wf.getsampwidth()
    if sample_width == 2:
        dtype = np.int16
    elif sample_width == 4:
        dtype = np.int32
    else:
        raise ValueError('Unsupported audio sample width: %s' % sample_width)

    audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    audio_norm = audio / np.max(np.abs(audio) + 1e-9)
    duration = len(audio_norm) / sample_rate
    energy = float(np.sqrt(np.mean(np.square(audio_norm))))

    frame_length = int(sample_rate * 0.1)
    if frame_length < 256:
        frame_length = 256
    silence_frames = 0
    total_windows = 0

    for start in range(0, len(audio_norm), frame_length):
        window = audio_norm[start:start + frame_length]
        if window.size == 0:
            continue
        window_energy = float(np.sqrt(np.mean(np.square(window))))
        if window_energy < 0.02:
            silence_frames += 1
        total_windows += 1

    silence_ratio = float(silence_frames / max(total_windows, 1))
    clarity_score = _energy_to_score(energy) * _silence_to_score(silence_ratio)

    return {
        'duration': round(duration, 1),
        'sample_rate': sample_rate,
        'energy': round(energy, 4),
        'silence_ratio': round(silence_ratio, 4),
        'energy_score': round(_energy_to_score(energy), 3),
        'clarity_score': round(clarity_score, 3),
        'audio_confidence': round((0.7 * _energy_to_score(energy) + 0.3 * _silence_to_score(silence_ratio)), 3),
    }
