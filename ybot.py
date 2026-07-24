# Import necessary libraries for the YouTube bot
import gradio as gr
import re  #For extracting video id 
from youtube_transcript_api import YouTubeTranscriptApi  # For extracting transcripts from YouTube videos
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into manageable segments
from langchain_community.vectorstores import FAISS  # For efficient vector storage and similarity search
from langchain.chains import LLMChain  # For creating chains of operations with LLMs
from langchain.prompts import PromptTemplate  # For defining prompt templates
from llm_interface import get_embeddings, get_llm  # For creating the embedding and LLM models


def get_video_id(url):    
    # Regex pattern to match YouTube video URLs
    pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None




def get_transcript(url):
    # Extracts the video ID from the URL
    video_id = get_video_id(url)
    
    # Create a YouTubeTranscriptApi() object
    ytt_api = YouTubeTranscriptApi()
    
    # Fetch the list of available transcripts for the given YouTube video
    transcripts = ytt_api.list(video_id)
    
    transcript = ""
    for t in transcripts:
        # Check if the transcript's language is English
        if t.language_code == 'en':
            if t.is_generated:
                # If no transcript has been set yet, use the auto-generated one
                if len(transcript) == 0:
                    transcript = t.fetch()
            else:
                # If a manually created transcript is found, use it (overrides auto-generated)
                transcript = t.fetch()
                break  # Prioritize the manually created transcript, exit the loop
    
    return transcript if transcript else None


# Sample YouTube URL
url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Fetching the transcript
transcript = get_transcript(url)



def process(transcript):
    # Initialize an empty string to hold the formatted transcript
    txt = ""
    
    # Loop through each entry in the transcript
    for i in transcript:
        try:
            # Append the text and its start time to the output string
            txt += f"Text: {i.text} Start: {i.start}\n"
        except KeyError:
            # If there is an issue accessing 'text' or 'start', skip this entry
            pass
            
    # Return the processed transcript as a single string
    return txt




def chunk_transcript(processed_transcript, chunk_size=200, chunk_overlap=20):
    # Initialize the RecursiveCharacterTextSplitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Split the transcript into chunks
    chunks = text_splitter.split_text(processed_transcript)
    return chunks


SUMMARY_PROMPT = PromptTemplate(
    input_variables=["transcript"],
    template=(
        "Summarize the following YouTube transcript in three short bullet points.\n\n"
        "{transcript}"
    ),
)

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Answer the question using only the transcript excerpts below. "
        "If the excerpts do not contain the answer, say so.\n\n"
        "Excerpts:\n{context}\n\nQuestion: {question}\nAnswer:"
    ),
)


if __name__ == "__main__":
    processed = process(transcript)
    chunks = chunk_transcript(processed)
    print(f"Transcript: {len(processed)} chars -> {len(chunks)} chunks")

    # --- Embedding model: index every chunk and retrieve against a real query ---
    embeddings = get_embeddings()
    index = FAISS.from_texts(chunks, embeddings)
    print(f"FAISS index built, dimension {index.index.d}, {index.index.ntotal} vectors")

    question = "What does the singer promise never to do?"
    hits = index.similarity_search(question, k=3)
    print(f"\nTop {len(hits)} chunks for: {question!r}")
    for i, hit in enumerate(hits, 1):
        print(f"  {i}. {hit.page_content[:80]!r}")

    # --- LLM: summarize the transcript, then answer using the retrieved chunks ---
    llm = get_llm()

    summary = (SUMMARY_PROMPT | llm).invoke({"transcript": processed})
    print(f"\nSummary:\n{summary.content}")

    context = "\n".join(hit.page_content for hit in hits)
    answer = (QA_PROMPT | llm).invoke({"context": context, "question": question})
    print(f"\nAnswer: {answer.content}")


