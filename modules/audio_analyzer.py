import wave
import numpy as np

def analyze_audio(audio_path):
    with wave.open(audio_path, 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)

    duration = len(audio) / 16000
    energy = float(np.mean(np.abs(audio))) / 10000
    speech_rate = np.mean(np.abs(np.diff(audio))) / 10000

    return {
        "duration": duration,
        "energy": energy,
        "speech_rate": speech_rate
    }