"""Tests that UI handlers surface errors instead of raising."""

import app
from transcript import TranscriptError


def test_summarize_video_reports_bad_url_without_raising():
    result = app.summarize_video("https://vimeo.com/12345")

    assert result.startswith("⚠️")
    assert "YouTube" in result


def test_ask_question_requires_a_question():
    result = app.ask_question("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "   ")
    assert result.startswith("⚠️")


def test_ask_question_reports_missing_transcript(monkeypatch):
    def boom(url):
        raise TranscriptError("This video has no English transcript.")

    monkeypatch.setattr(app, "get_transcript", boom)

    result = app.ask_question("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "why?")

    assert result.startswith("⚠️")
    assert "no English transcript" in result


def test_summarize_video_reports_missing_api_key(monkeypatch, snippets):
    monkeypatch.setattr(app, "get_transcript", lambda url: snippets)
    monkeypatch.setattr(app, "get_index", lambda video_id, docs: object())

    def no_key(text):
        raise ValueError("GOOGLE_API_KEY is not set.")

    monkeypatch.setattr(app, "summarize", no_key)

    result = app.summarize_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    assert result.startswith("⚠️")
    assert "GOOGLE_API_KEY" in result
