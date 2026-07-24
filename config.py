"""Configuration settings for the YouTube Summarizer."""

import os

from dotenv import load_dotenv

load_dotenv()

# Local embeddings (sentence-transformers, runs on CPU, no API quota).
# The multilingual model is used instead of all-MiniLM-L6-v2 because the latter
# is English-only: Arabic text embeds into meaningless vectors there, so
# retrieval returns arbitrary chunks. This one costs a larger download (~470MB
# vs ~90MB) but maps Arabic and English into the same 384-dim space.
LOCAL_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LOCAL_EMBED_KWARGS = {"device": "cpu"}
LOCAL_ENCODE_KWARGS = {"normalize_embeddings": True}

# Google Gemini settings
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_ID = "gemini-flash-latest"

# Generation parameters
LLM_TEMPERATURE = 0.2
LLM_MAX_NEW_TOKENS = 2048

# Transcript languages, most preferred first. Region variants (ar-EG, en-US)
# match on their base code, so "ar" covers every Arabic track YouTube offers.
TRANSCRIPT_LANGUAGES = ("en", "ar")

LANGUAGE_NAMES = {"en": "English", "ar": "Arabic"}

# Transcript chunking. 1000 chars keeps each chunk self-contained enough to
# stand alone as retrieval context; the overlap avoids splitting mid-thought.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Retrieval. MMR trades a little relevance for diversity, so repeated phrasing
# in a transcript does not crowd out distinct content.
RETRIEVAL_K = 4
RETRIEVAL_FETCH_K = 20

# Gradio server settings. Leaving the port unset lets Gradio scan 7860-7959
# for a free one, so a second instance does not fail outright on a taken port.
GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
_port = os.getenv("GRADIO_SERVER_PORT")
GRADIO_SERVER_PORT = int(_port) if _port else None
