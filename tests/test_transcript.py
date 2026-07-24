"""Tests for URL parsing and transcript normalization."""

import pytest
from conftest import FakeTrack

from transcript import (
    TranscriptError,
    _base_language,
    _normalize,
    _select,
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


@pytest.mark.parametrize(
    ("code", "expected"),
    [("ar", "ar"), ("ar-EG", "ar"), ("AR-SA", "ar"), ("en-US", "en"), ("zh_CN", "zh")],
)
def test_base_language_strips_region(code, expected):
    assert _base_language(code) == expected


def test_select_picks_arabic_when_no_english_exists():
    tracks = [FakeTrack("fr", False), FakeTrack("ar", True)]
    assert _select(tracks).language_code == "ar"


def test_select_matches_arabic_region_variants():
    assert _select([FakeTrack("ar-EG", True)]).language_code == "ar-EG"


def test_select_prefers_english_over_arabic():
    tracks = [FakeTrack("ar", False), FakeTrack("en", False)]
    assert _select(tracks).language_code == "en"


def test_select_prefers_manual_over_generated_within_a_language():
    # The generated track comes first, so a first-match scan would take it.
    tracks = [FakeTrack("ar", True), FakeTrack("ar", False)]
    assert _select(tracks).is_generated is False


def test_select_prefers_language_rank_over_manual():
    # A generated English track still beats a manual Arabic one.
    tracks = [FakeTrack("ar", False), FakeTrack("en", True)]
    assert _select(tracks).language_code == "en"


def test_select_returns_none_for_unsupported_languages():
    assert _select([FakeTrack("fr", False), FakeTrack("de", True)]) is None
