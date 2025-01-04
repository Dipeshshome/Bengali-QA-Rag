from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    CHROMA_DB_DIR = "chroma_db"
    MODEL_NAME = "gpt-3.5-turbo"