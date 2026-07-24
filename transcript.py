"""Fetching and normalizing YouTube transcripts."""

import re

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import CouldNotRetrieveTranscript

import config

# Accepts the common YouTube link shapes: watch?v=, youtu.be/, /shorts/,
# /embed/ and /live/, with or without extra query parameters.
_VIDEO_ID_PATTERNS = (
    r"(?:youtube\.com|youtube-nocookie\.com)/watch\?(?:.*&)?v=([a-zA-Z0-9_-]{11})",
    r"youtu\.be/([a-zA-Z0-9_-]{11})",
    r"(?:youtube\.com|youtube-nocookie\.com)/(?:shorts|embed|live|v)/([a-zA-Z0-9_-]{11})",
)


class TranscriptError(Exception):
    """Raised when a transcript cannot be retrieved for a URL."""


def get_video_id(url):
    """Extract the 11-character video ID from a YouTube URL.

    Returns None when the URL is not a recognizable YouTube link.
    """
    if not url:
        return None

    for pattern in _VIDEO_ID_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def get_transcript(url):
    """Fetch a transcript for a YouTube URL in one of the supported languages.

    Returns a list of {"text": str, "start": float} entries. Languages are
    tried in config.TRANSCRIPT_LANGUAGES order, and within a language a
    manually created transcript beats an auto-generated one.

    Raises TranscriptError if the URL is not a YouTube link, or if the video
    has no transcript in a supported language.
    """
    video_id = get_video_id(url)
    if video_id is None:
        raise TranscriptError(
            "That does not look like a YouTube video link. Paste a URL such as "
            "https://www.youtube.com/watch?v=VIDEO_ID"
        )

    try:
        transcripts = YouTubeTranscriptApi().list(video_id)
    except CouldNotRetrieveTranscript as exc:
        raise TranscriptError(
            f"No transcript is available for this video ({type(exc).__name__}). "
            "Subtitles may be disabled, or the video may be private or age-restricted."
        ) from exc

    chosen = _select(transcripts)
    if chosen is None:
        raise TranscriptError(
            f"This video has no transcript in {_supported_languages()}."
        )

    return _normalize(chosen.fetch())


def _base_language(language_code):
    """Reduce a track's language code to its base tag: "ar-EG" -> "ar"."""
    return language_code.lower().replace("_", "-").split("-")[0]


def _select(transcripts, languages=None):
    """Pick the best available transcript, or None if none is usable.

    Ranks candidates by (language preference, auto-generated) and takes the
    smallest. Sorting rather than short-circuiting on the first match matters
    because the API yields tracks in arbitrary order, so an auto-generated
    English track can appear before a manual one.
    """
    languages = languages or config.TRANSCRIPT_LANGUAGES

    best = None
    best_rank = None

    for t in transcripts:
        base = _base_language(t.language_code)
        if base not in languages:
            continue

        rank = (languages.index(base), t.is_generated)
        if best_rank is None or rank < best_rank:
            best, best_rank = t, rank

    return best


def _supported_languages():
    """Render the configured languages for an error message."""
    names = [
        config.LANGUAGE_NAMES.get(code, code) for code in config.TRANSCRIPT_LANGUAGES
    ]
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} or {names[-1]}"


def _normalize(fetched):
    """Convert API snippet objects into plain dicts, skipping malformed entries."""
    snippets = []
    for snippet in fetched:
        try:
            text = snippet.text.strip()
            start = float(snippet.start)
        except (AttributeError, TypeError, ValueError):
            # A snippet missing text or start is unusable; drop it rather than
            # letting it break the whole transcript.
            continue
        if text:
            snippets.append({"text": text, "start": start})
    return snippets


def transcript_text(snippets):
    """Join snippets into one plain-text transcript for summarization."""
    return " ".join(s["text"] for s in snippets)


def format_timestamp(seconds):
    """Render a start time in seconds as M:SS (or H:MM:SS past an hour)."""
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
