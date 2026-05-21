import os

class Config:
    SECRET_KEY = 'super-secret-key-for-development-only'
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm'}
    MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200 MB