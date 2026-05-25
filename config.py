import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'super-secret-key-for-development-only')
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm'}
    MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200 MB

    LLM_SERVER_URL = os.environ.get('LLM_SERVER_URL', 'http://localhost:11434')
    LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', 'phi3:latest')
    LLM_TIMEOUT = int(os.environ.get('LLM_TIMEOUT', '180'))
    LLM_GENERATION_TIMEOUT = int(os.environ.get('LLM_GENERATION_TIMEOUT', '180'))

    VIDEO_SAMPLE_COUNT = int(os.environ.get('VIDEO_SAMPLE_COUNT', '15'))
    VIDEO_MIN_SAMPLE_SECONDS = float(os.environ.get('VIDEO_MIN_SAMPLE_SECONDS', '0.5'))
    AUDIO_TARGET_SAMPLE_RATE = int(os.environ.get('AUDIO_TARGET_SAMPLE_RATE', '16000'))
