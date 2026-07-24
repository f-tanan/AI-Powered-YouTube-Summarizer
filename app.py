"""Gradio UI for the YouTube summarizer."""

import gradio as gr

import config
from indexing import chunk_transcript, get_index
from summarizer import answer_question, summarize
from transcript import (
    TranscriptError,
    format_timestamp,
    get_transcript,
    get_video_id,
    transcript_text,
)


def _load(url):
    """Fetch and index a video, returning (index, transcript text).

    get_index caches by video ID, so asking a second question about the same
    video reuses the embeddings computed for the first.
    """
    snippets = get_transcript(url)
    video_id = get_video_id(url)
    index = get_index(video_id, chunk_transcript(snippets))
    return index, transcript_text(snippets)


def summarize_video(url):
    """Handler for the Summarize button."""
    try:
        _, text = _load(url)
        return summarize(text)
    except (TranscriptError, ValueError) as exc:
        return f"⚠️ {exc}"


def ask_question(url, question):
    """Handler for the Ask button."""
    if not question.strip():
        return "⚠️ Enter a question first."

    try:
        index, _ = _load(url)
        answer, sources = answer_question(index, question)
    except (TranscriptError, ValueError) as exc:
        return f"⚠️ {exc}"

    cited = ", ".join(format_timestamp(doc.metadata["start"]) for doc in sources)
    return f"{answer}\n\n*Based on excerpts at: {cited}*"


def build_ui():
    """Assemble the Gradio interface."""
    with gr.Blocks(title="YouTube Summarizer") as demo:
        gr.Markdown(
            "# YouTube Summarizer\nSummarize a video and ask questions about it. "
            "English and Arabic videos are supported — answers come back in the "
            "language you ask in."
        )

        url = gr.Textbox(
            label="YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
        )

        with gr.Tab("Summary"):
            summarize_btn = gr.Button("Summarize", variant="primary")
            summary_out = gr.Markdown()
            summarize_btn.click(summarize_video, inputs=url, outputs=summary_out)

        with gr.Tab("Q&A"):
            question = gr.Textbox(
                label="Question",
                placeholder="What does the speaker say about ...?",
            )
            ask_btn = gr.Button("Ask", variant="primary")
            answer_out = gr.Markdown()
            ask_btn.click(ask_question, inputs=[url, question], outputs=answer_out)

    return demo


def main():
    build_ui().launch(
        server_name=config.GRADIO_SERVER_NAME,
        server_port=config.GRADIO_SERVER_PORT,
    )


if __name__ == "__main__":
    main()
