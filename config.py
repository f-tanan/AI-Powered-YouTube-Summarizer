"""Configuration settings for the Icebreaker Bot."""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Local embeddings (sentence-transformers, runs on CPU, no API quota)
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_EMBED_KWARGS = {"device": "cpu"}
LOCAL_ENCODE_KWARGS = {"normalize_embeddings": True}

# Google Gemini settings
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_ID = "gemini-flash-latest"

# Generation parameters, shared by whichever LLM backend is active
LLM_TEMPERATURE = 0.2
LLM_MAX_NEW_TOKENS = 2048



