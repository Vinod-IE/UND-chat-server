import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    DB_FOLDER = os.getenv('DB_FOLDER', './vector_store')
    # GROQ_KEY = os.getenv('GROQ_KEY')
    # GROQ_MODEL = os.getenv('GROQ_MODEL')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 500))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 100))

    @classmethod
    def check(cls):
        required = ['OPENAI_API_KEY']  #'GROQ_KEY', 'GROQ_MODEL',
        for var in required:
            if not getattr(cls, var):
                raise ValueError(f"Missing {var} in environment")
