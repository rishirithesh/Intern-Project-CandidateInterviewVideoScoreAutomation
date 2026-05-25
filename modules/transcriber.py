from faster_whisper import WhisperModel
import os

MODEL_NAME = os.environ.get('WHISPER_MODEL', 'base')
MODEL_DEVICE = os.environ.get('WHISPER_DEVICE', 'cpu')
MODEL_COMPUTE_TYPE = os.environ.get('WHISPER_COMPUTE_TYPE', 'int8')

_model = None


def get_whisper_model():
    global _model
    if _model is None:
        print("🔄 Loading deterministic Whisper model...")
        _model = WhisperModel(MODEL_NAME, device=MODEL_DEVICE, compute_type=MODEL_COMPUTE_TYPE)
    return _model


def transcribe_audio(audio_path: str) -> dict:
    model = get_whisper_model()
    print("→ Transcribing audio with deterministic settings...")
    segments, info = model.transcribe(
        audio_path,
        language='en',
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        temperature=0.0,
        best_of=1,
    )

    transcript = ' '.join(segment.text.strip() for segment in segments).strip()
    return {
        'text': transcript,
        'segments': [
            {
                'start': segment.start,
                'end': segment.end,
                'text': segment.text,
            }
            for segment in segments
        ],
        'duration': info.duration if hasattr(info, 'duration') else None,
    }
