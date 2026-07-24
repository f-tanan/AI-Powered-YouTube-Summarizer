"""Tests for prompt assembly, summarization and QA, using a fake LLM."""

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from summarizer import (
    QA_PROMPT,
    SUMMARY_PROMPT,
    answer_question,
    format_context,
    response_text,
    summarize,
)


class FakeResponse:
    def __init__(self, content):
        self.content = content


def test_response_text_passes_through_plain_strings():
    assert response_text(FakeResponse("hello")) == "hello"


def test_response_text_extracts_gemini_content_blocks():
    # Gemini 3.x returns text blocks alongside reasoning signature blocks.
    response = FakeResponse(
        [
            {"type": "text", "text": "the answer"},
            {"type": "thought_signature", "extras": {"signature": "abc123"}},
        ]
    )

    assert response_text(response) == "the answer"


def test_response_text_joins_multiple_text_blocks():
    response = FakeResponse(
        [{"type": "text", "text": "part one "}, {"type": "text", "text": "part two"}]
    )

    assert response_text(response) == "part one part two"


def fake_llm(reply="canned reply"):
    return FakeListChatModel(responses=[reply])


def test_summarize_returns_model_text():
    assert summarize("a transcript", llm=fake_llm("• point one")) == "• point one"


def test_summary_prompt_embeds_the_transcript():
    rendered = SUMMARY_PROMPT.format(transcript="the quick brown fox")
    assert "the quick brown fox" in rendered


def test_format_context_labels_each_excerpt_with_a_timestamp():
    docs = [
        Document(page_content="first passage", metadata={"start": 5.0}),
        Document(page_content="second passage", metadata={"start": 125.0}),
    ]

    context = format_context(docs)

    assert "[0:05] first passage" in context
    assert "[2:05] second passage" in context


def test_qa_prompt_includes_context_and_question():
    rendered = QA_PROMPT.format(context="[0:05] some excerpt", question="why?")

    assert "[0:05] some excerpt" in rendered
    assert "why?" in rendered


class StubIndex:
    """Stands in for a FAISS index, recording the retrieval arguments."""

    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    def max_marginal_relevance_search(self, question, k, fetch_k):
        self.calls.append((question, k, fetch_k))
        return self.documents


def test_answer_question_returns_answer_and_its_sources():
    docs = [Document(page_content="backpropagation", metadata={"start": 21.7})]
    index = StubIndex(docs)

    answer, sources = answer_question(index, "how does training work?", llm=fake_llm("it uses backprop"))

    assert answer == "it uses backprop"
    assert sources == docs


def test_answer_question_uses_mmr_retrieval():
    index = StubIndex([Document(page_content="x", metadata={"start": 0.0})])

    answer_question(index, "a question", llm=fake_llm())

    question, k, fetch_k = index.calls[0]
    assert question == "a question"
    # fetch_k must exceed k, or MMR has no candidates to diversify among.
    assert fetch_k > k
