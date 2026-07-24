"""Factories for the embedding and LLM models.

Both factories are cached: loading the sentence-transformers model reads
several hundred megabytes off disk, so it must happen once per process rather
than once per request.
"""

from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

import config


@lru_cache(maxsize=None)
def get_embeddings(model_id=None):
    """Return a local sentence-transformers embeddings model."""
    return HuggingFaceEmbeddings(
        model_name=model_id or config.LOCAL_EMBEDDING_MODEL,
        model_kwargs=config.LOCAL_EMBED_KWARGS,
        encode_kwargs=config.LOCAL_ENCODE_KWARGS,
    )


@lru_cache(maxsize=None)
def get_llm(model_id=None):
    """Return a Gemini chat model using the configured API key."""
    if not config.GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Create a free key at "
            "https://aistudio.google.com/apikey and add it to your .env file."
        )

    return ChatGoogleGenerativeAI(
        model=model_id or config.GEMINI_MODEL_ID,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=config.LLM_TEMPERATURE,
        max_output_tokens=config.LLM_MAX_NEW_TOKENS,
    )
