from faster_whisper import WhisperModel

model = None

def get_whisper_model():
    global model
    if model is None:
        print("🔄 Loading lightweight 'base' model for office laptops...")
        model = WhisperModel(
            "base",              # Good accuracy + reasonable speed
            device="cpu",
            compute_type="int8"  # Reduces RAM usage significantly
        )
    return model


def transcribe_audio(audio_path: str) -> dict:
    """Lightweight transcription"""
    whisper_model = get_whisper_model()
    
    print("→ Starting transcription...")
    segments, info = whisper_model.transcribe(
        audio_path,
        language="en",
        beam_size=5,
        word_timestamps=True,
        vad_filter=True
    )
    
    full_text = ""
    segment_list = []
    
    for segment in segments:
        full_text += segment.text.strip() + " "
        segment_list.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text
        })
    
    print(f"→ Transcription completed: {len(full_text.split())} words")
    return {
        "text": full_text.strip(),
        "segments": segment_list
    }