import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# Models
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama3"

# Ingestion Settings
WIKI_LANG = "es"  # Wikipedia language (can be 'en' or 'es' depending on preference, defaulting to user language implication or 'en' if standard. User asked in Spanish, let's use 'es' if possible, or verify what they want. 'wikipedia-api' supports language code. The prompt requested 'Wikipedia' without specifying lang, but the user speaks Spanish. I will default to Spanish for content relevance.)
WIKI_TOPICS = [
    "Cambio climático",
    "Inteligencia artificial",
    "Vacunación",
    "Energía nuclear"
]

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50  # 10% of 500
