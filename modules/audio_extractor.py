from moviepy.editor import VideoFileClip
import os

def extract_audio(video_path: str, audio_path: str = None) -> str:
    """
    Extract audio safely with validation
    """

    if not os.path.exists(video_path):
        raise Exception("Video file not found")

    if audio_path is None:
        audio_path = video_path.rsplit('.', 1)[0] + '.wav'

    try:
        video = VideoFileClip(video_path)

        if video.audio is None:
            raise Exception("No audio track found in video")

        audio = video.audio
        audio.write_audiofile(audio_path, fps=16000, logger=None)

        video.close()

        return audio_path

    except Exception as e:
        raise Exception(f"Audio extraction failed: {str(e)}")