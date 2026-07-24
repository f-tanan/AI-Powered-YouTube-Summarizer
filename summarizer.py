"""Summarization and question answering over a transcript."""

from langchain_core.prompts import PromptTemplate

from indexing import retrieve
from llm_interface import get_llm
from transcript import format_timestamp

# Both prompts pin the output language explicitly. Without that instruction the
# model answers an Arabic transcript in English, since the instructions
# themselves are English.
SUMMARY_PROMPT = PromptTemplate.from_template(
    "Summarize the following YouTube transcript as 3-5 concise bullet points "
    "covering its main ideas. Do not add information that is not in the "
    "transcript. Write the summary in the same language as the transcript.\n\n"
    "Transcript:\n{transcript}\n\nSummary:"
)

QA_PROMPT = PromptTemplate.from_template(
    "Answer the question using only the transcript excerpts below. Each "
    "excerpt is labelled with its timestamp. If the excerpts do not contain "
    "the answer, say that the video does not cover it. Write the answer in the "
    "same language as the question.\n\n"
    "Excerpts:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)


def response_text(response):
    """Extract plain text from a chat response.

    Newer Gemini models return `content` as a list of content blocks (text
    alongside reasoning signatures) rather than a single string, so the text
    blocks have to be pulled out and joined.
    """
    content = response.content

    if isinstance(content, str):
        return content

    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))

    return "".join(parts).strip()


def summarize(transcript_text, llm=None):
    """Summarize a full transcript into bullet points."""
    llm = llm or get_llm()
    response = (SUMMARY_PROMPT | llm).invoke({"transcript": transcript_text})
    return response_text(response)


def format_context(documents):
    """Render retrieved chunks as timestamped excerpts for the prompt."""
    return "\n\n".join(
        f"[{format_timestamp(doc.metadata['start'])}] {doc.page_content}"
        for doc in documents
    )


def answer_question(index, question, llm=None):
    """Answer a question about a video from its indexed transcript.

    Returns (answer, source_documents) so callers can show what was cited.
    """
    llm = llm or get_llm()
    documents = retrieve(index, question)
    context = format_context(documents)
    response = (QA_PROMPT | llm).invoke({"context": context, "question": question})
    return response_text(response), documents
