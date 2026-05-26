import os


DEFAULT_LLM_MODEL = 'qwen2.5:3b'


def _llm_model_name() -> str:
    if os.environ.get('ALLOW_LLM_MODEL_OVERRIDE', '').strip().lower() not in {'1', 'true', 'yes'}:
        return DEFAULT_LLM_MODEL
    configured = os.environ.get('LLM_MODEL_NAME', '').strip()
    return configured or DEFAULT_LLM_MODEL


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'super-secret-key-for-development-only')
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm'}
    MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200 MB

    LLM_SERVER_URL = os.environ.get('LLM_SERVER_URL', 'http://localhost:11434')
    LLM_MODEL_NAME = _llm_model_name()
    LLM_TIMEOUT = int(os.environ.get('LLM_TIMEOUT', '180'))
    LLM_GENERATION_TIMEOUT = int(os.environ.get('LLM_GENERATION_TIMEOUT', '180'))

    VIDEO_SAMPLE_COUNT = int(os.environ.get('VIDEO_SAMPLE_COUNT', '15'))
    VIDEO_MIN_SAMPLE_SECONDS = float(os.environ.get('VIDEO_MIN_SAMPLE_SECONDS', '0.5'))
    AUDIO_TARGET_SAMPLE_RATE = int(os.environ.get('AUDIO_TARGET_SAMPLE_RATE', '16000'))
