"""Live end-to-end check: fetch a real video, index it, summarize and ask.

Unlike the pytest suite this hits the network and the Gemini API. Run it with
    python smoke_test.py [YOUTUBE_URL]
"""

import sys

from indexing import chunk_transcript, get_index
from summarizer import answer_question, summarize
from transcript import (
    TranscriptError,
    format_timestamp,
    get_transcript,
    get_video_id,
    transcript_text,
)

DEFAULT_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
DEFAULT_QUESTION = "What does the singer promise never to do?"


def main(url, question):
    try:
        snippets = get_transcript(url)
    except TranscriptError as exc:
        print(f"Transcript unavailable: {exc}")
        return 1

    text = transcript_text(snippets)
    documents = chunk_transcript(snippets)
    print(f"{len(snippets)} snippets, {len(text)} chars -> {len(documents)} chunks")

    index = get_index(get_video_id(url), documents)
    print(f"Index: {index.index.ntotal} vectors, dimension {index.index.d}")

    print(f"\nSummary:\n{summarize(text)}")

    answer, sources = answer_question(index, question)
    cited = ", ".join(format_timestamp(d.metadata["start"]) for d in sources)
    print(f"\nQ: {question}\nA: {answer}\nSources: {cited}")
    return 0


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    question = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_QUESTION
    sys.exit(main(url, question))
