"""Tests for URL parsing and transcript normalization."""

import pytest

from transcript import (
    TranscriptError,
    _normalize,
    format_timestamp,
    get_transcript,
    get_video_id,
    transcript_text,
)

VIDEO_ID = "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "url",
    [
        f"https://www.youtube.com/watch?v={VIDEO_ID}",
        f"https://youtube.com/watch?v={VIDEO_ID}",
        f"https://www.youtube.com/watch?v={VIDEO_ID}&t=42s",
        f"https://www.youtube.com/watch?list=PL123&v={VIDEO_ID}",
        f"https://youtu.be/{VIDEO_ID}",
        f"https://youtu.be/{VIDEO_ID}?t=42",
        f"https://www.youtube.com/shorts/{VIDEO_ID}",
        f"https://www.youtube.com/embed/{VIDEO_ID}",
        f"https://www.youtube.com/live/{VIDEO_ID}",
    ],
)
def test_get_video_id_accepts_common_url_shapes(url):
    assert get_video_id(url) == VIDEO_ID


@pytest.mark.parametrize(
    "url",
    ["", None, "https://vimeo.com/12345", "not a url", "https://www.youtube.com/"],
)
def test_get_video_id_rejects_non_youtube_urls(url):
    assert get_video_id(url) is None


def test_get_transcript_rejects_bad_url_before_any_network_call():
    with pytest.raises(TranscriptError, match="does not look like a YouTube"):
        get_transcript("https://vimeo.com/12345")


def test_normalize_extracts_text_and_start(fake_fetched, snippets):
    assert _normalize(fake_fetched) == snippets


def test_normalize_skips_malformed_entries(fake_fetched):
    class Broken:
        """No text or start attribute at all."""

    fake_fetched.insert(2, Broken())
    result = _normalize(fake_fetched)

    # The broken entry is dropped rather than raising.
    assert len(result) == len(fake_fetched) - 1
    assert all("text" in r and "start" in r for r in result)


def test_normalize_drops_empty_text(fake_fetched):
    fake_fetched[0].text = "   "
    assert len(_normalize(fake_fetched)) == len(fake_fetched) - 1


def test_transcript_text_has_no_scaffolding(snippets):
    text = transcript_text(snippets)

    assert "Text:" not in text
    assert "Start:" not in text
    assert text.startswith("Welcome to this talk")


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "0:00"), (9.5, "0:09"), (65, "1:05"), (600, "10:00"), (3661, "1:01:01")],
)
def test_format_timestamp(seconds, expected):
    assert format_timestamp(seconds) == expected
