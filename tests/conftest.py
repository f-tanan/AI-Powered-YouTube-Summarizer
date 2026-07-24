"""Shared fixtures. Everything here runs offline."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def snippets():
    """A short transcript in the shape transcript.get_transcript returns."""
    return [
        {"text": "Welcome to this talk about neural networks.", "start": 0.0},
        {"text": "We will start with the perceptron.", "start": 4.5},
        {"text": "A perceptron takes weighted inputs and applies a step function.", "start": 9.0},
        {"text": "Stacking perceptrons gives us a multilayer network.", "start": 15.2},
        {"text": "Training uses backpropagation to adjust the weights.", "start": 21.7},
        {"text": "Backpropagation applies the chain rule from calculus.", "start": 28.3},
        {"text": "That is everything for today, thanks for watching.", "start": 35.0},
    ]


class FakeSnippet:
    """Mimics a youtube_transcript_api snippet object."""

    def __init__(self, text, start):
        self.text = text
        self.start = start


@pytest.fixture
def fake_fetched(snippets):
    return [FakeSnippet(s["text"], s["start"]) for s in snippets]


class FakeTrack:
    """Mimics one entry of youtube_transcript_api's transcript list."""

    def __init__(self, language_code, is_generated):
        self.language_code = language_code
        self.is_generated = is_generated

    def __repr__(self):
        kind = "auto" if self.is_generated else "manual"
        return f"<{self.language_code} {kind}>"
