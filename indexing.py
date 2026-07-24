"""Chunking transcripts and building the retrieval index."""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import config
from llm_interface import get_embeddings

# One FAISS index per video. Building an index re-embeds every chunk, so a
# follow-up question about a video the user already asked about must not pay
# that cost again.
_INDEX_CACHE = {}


def chunk_transcript(snippets, chunk_size=None, chunk_overlap=None):
    """Group transcript snippets into overlapping Documents.

    Snippet boundaries are natural sentence-ish breaks, so chunks are built by
    accumulating whole snippets rather than by splitting raw characters. Each
    chunk carries the start time of its first snippet as metadata, which keeps
    timestamps out of the embedded text while still allowing citations.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    documents = []
    buffer = []
    buffer_len = 0

    for snippet in snippets:
        buffer.append(snippet)
        buffer_len += len(snippet["text"]) + 1

        if buffer_len >= chunk_size:
            documents.append(_to_document(buffer))
            buffer, buffer_len = _carry_over(buffer, chunk_overlap)

    if buffer:
        documents.append(_to_document(buffer))

    return documents


def _to_document(buffer):
    """Build a Document from buffered snippets, tagged with its start time."""
    return Document(
        page_content=" ".join(s["text"] for s in buffer),
        metadata={"start": buffer[0]["start"]},
    )


def _carry_over(buffer, chunk_overlap):
    """Return the trailing snippets that should open the next chunk."""
    carried = []
    carried_len = 0

    for snippet in reversed(buffer):
        length = len(snippet["text"]) + 1
        if carried_len + length > chunk_overlap:
            break
        carried.insert(0, snippet)
        carried_len += length

    return carried, carried_len


def build_index(documents):
    """Embed documents into a FAISS index."""
    return FAISS.from_documents(documents, get_embeddings())


def get_index(video_id, documents):
    """Return the cached index for a video, building it on first use."""
    if video_id not in _INDEX_CACHE:
        _INDEX_CACHE[video_id] = build_index(documents)
    return _INDEX_CACHE[video_id]


def retrieve(index, question, k=None, fetch_k=None):
    """Fetch the chunks most relevant to a question.

    Uses maximal marginal relevance so that repeated phrasing in a transcript
    does not fill every slot with near-duplicates of the same passage.
    """
    return index.max_marginal_relevance_search(
        question,
        k=k or config.RETRIEVAL_K,
        fetch_k=fetch_k or config.RETRIEVAL_FETCH_K,
    )


def clear_cache():
    """Drop every cached index. Used by tests."""
    _INDEX_CACHE.clear()
