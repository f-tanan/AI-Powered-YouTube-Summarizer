"""Configuration settings for the YouTube Summarizer."""

import os

from dotenv import load_dotenv

load_dotenv()

# Local embeddings (sentence-transformers, runs on CPU, no API quota)
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_EMBED_KWARGS = {"device": "cpu"}
LOCAL_ENCODE_KWARGS = {"normalize_embeddings": True}

# Google Gemini settings
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_ID = "gemini-flash-latest"

# Generation parameters
LLM_TEMPERATURE = 0.2
LLM_MAX_NEW_TOKENS = 2048

# Transcript chunking. 1000 chars keeps each chunk self-contained enough to
# stand alone as retrieval context; the overlap avoids splitting mid-thought.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Retrieval. MMR trades a little relevance for diversity, so repeated phrasing
# in a transcript does not crowd out distinct content.
RETRIEVAL_K = 4
RETRIEVAL_FETCH_K = 20

# Gradio server settings
GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
