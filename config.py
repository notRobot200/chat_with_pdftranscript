import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Directory Configuration
CHROMA_DB_DIR = "docs/chroma_db"
TEMP_DIR = "docs/temp"
DEFAULT_DOCS_DIR = "docs"

# Embeddings Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM Configuration
LLM_MODEL = "meta-llama/Llama-2-7b-chat-hf"
LLM_TEMPERATURE = 0
LLM_MAX_TOKENS = 2000
LLM_TOP_P = 0.9
LLM_CONTEXT_WINDOW = 4096

# PDF Processing Configuration
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Retrieval Configuration
TOP_K_CHUNKS = 3
