"""Tests for chunking and the index cache. No model is loaded here."""

import indexing
from indexing import chunk_transcript, get_index


def test_chunks_carry_start_time_in_metadata(snippets):
    docs = chunk_transcript(snippets, chunk_size=100, chunk_overlap=0)

    assert docs
    assert docs[0].metadata["start"] == 0.0
    # Later chunks start later in the video.
    starts = [d.metadata["start"] for d in docs]
    assert starts == sorted(starts)


def test_chunk_text_excludes_timestamps(snippets):
    docs = chunk_transcript(snippets, chunk_size=100, chunk_overlap=0)

    for doc in docs:
        assert "Start:" not in doc.page_content
        assert "Text:" not in doc.page_content


def test_every_snippet_survives_chunking(snippets):
    docs = chunk_transcript(snippets, chunk_size=100, chunk_overlap=0)
    combined = " ".join(d.page_content for d in docs)

    for snippet in snippets:
        assert snippet["text"] in combined


def test_large_chunk_size_yields_single_document(snippets):
    docs = chunk_transcript(snippets, chunk_size=10_000, chunk_overlap=0)
    assert len(docs) == 1


def test_overlap_repeats_trailing_snippets(snippets):
    without = chunk_transcript(snippets, chunk_size=100, chunk_overlap=0)
    with_overlap = chunk_transcript(snippets, chunk_size=100, chunk_overlap=80)

    # Overlap duplicates content, so it cannot produce fewer chunks.
    assert len(with_overlap) >= len(without)


def test_empty_transcript_yields_no_documents():
    assert chunk_transcript([]) == []


def test_get_index_builds_once_per_video(monkeypatch, snippets):
    indexing.clear_cache()
    calls = []

    def fake_build(documents):
        calls.append(documents)
        return f"index-{len(calls)}"

    monkeypatch.setattr(indexing, "build_index", fake_build)
    docs = chunk_transcript(snippets)

    first = get_index("vid1", docs)
    second = get_index("vid1", docs)
    other = get_index("vid2", docs)

    assert first is second          # cached, not rebuilt
    assert first != other           # a different video gets its own index
    assert len(calls) == 2          # one build per distinct video

    indexing.clear_cache()
