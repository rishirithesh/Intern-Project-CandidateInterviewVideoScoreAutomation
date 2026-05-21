from moviepy.editor import VideoFileClip

def extract_audio(video_path: str, audio_path: str = None) -> str:
    """
    Extract audio from video file and save as WAV (16kHz)
    """
    if audio_path is None:
        audio_path = video_path.rsplit('.', 1)[0] + '.wav'

    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(audio_path, fps=16000, logger=None)
        video.close()
        return audio_path
    except Exception as e:
        raise Exception(f"Failed to extract audio: {str(e)}")