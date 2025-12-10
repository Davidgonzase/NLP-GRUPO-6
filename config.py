import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# Models
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
LLM_MODEL_NAME = "llama3"

# Ingestion Settings
WIKI_LANG = "en"  
WIKI_TOPICS = [
    "Artificial Intelligence",
    "ChatGPT",
    "Gemini",
    "Hallucination (artificial intelligence)",
    "BERT (language model)",
    "Transformer (deep learning)",
    "Generative pre-trained transformer"
]

CHUNK_SIZE = 150 
CHUNK_OVERLAP = 15 
