import os
import tempfile

DEFAULT_DOWNLOAD_CACHE_ROOT = os.path.join(tempfile.gettempdir(), "openpilot_cache")

class Paths:
    @staticmethod
    def shm_path():
        return tempfile.gettempdir()
    
    @staticmethod
    def log_root():
        return os.path.join(tempfile.gettempdir(), "openpilot_logs")
    
    @staticmethod
    def download_cache_root():
        return DEFAULT_DOWNLOAD_CACHE_ROOT
    
    @staticmethod
    def comma_home():
        return os.path.join(tempfile.gettempdir(), "comma_home")
