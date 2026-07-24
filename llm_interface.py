"""Module for interfacing with the embedding and LLM backends."""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

import config


def get_embeddings(model_id=None):
    """Create a local sentence-transformers embeddings instance."""
    return HuggingFaceEmbeddings(
        model_name=model_id or config.LOCAL_EMBEDDING_MODEL,
        model_kwargs=config.LOCAL_EMBED_KWARGS,
        encode_kwargs=config.LOCAL_ENCODE_KWARGS,
    )


def get_llm(model_id=None):
    """Create a Gemini chat model instance using the configured API key."""
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
