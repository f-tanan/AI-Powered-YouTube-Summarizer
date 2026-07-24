# AI-Powered YouTube Summarizer

Paste a YouTube URL to get a bullet-point summary of the video, then ask
follow-up questions and get answers grounded in the transcript with timestamp
citations. English and Arabic videos are both supported.

Summarization and Q&A run on Google Gemini; embeddings run locally on CPU, so
the only API key you need is a free Gemini one.

## How it works

1. **Fetch** — `youtube-transcript-api` pulls the caption track. Languages are
   tried in `config.TRANSCRIPT_LANGUAGES` order (English, then Arabic), and a
   manually written track is preferred over an auto-generated one.
2. **Chunk** — the transcript is grouped into ~1000-character overlapping
   chunks. Chunks are built by accumulating whole caption snippets rather than
   slicing raw characters, so they break at natural pauses. Each chunk keeps
   the start time of its first snippet as metadata.
3. **Index** — chunks are embedded with a local multilingual
   sentence-transformers model into a FAISS index, cached per video ID.
4. **Answer** — a question retrieves the most relevant chunks using maximal
   marginal relevance (which keeps repeated phrasing from filling every slot),
   and Gemini answers from those excerpts alone. The cited timestamps are shown
   alongside the answer.

Summarization skips retrieval entirely and sends the full transcript.

## Setup

Requires Python 3.10+ (developed on 3.11).

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

Then create your `.env`:

```bash
cp .env.example .env          # copy .env.example .env  on Windows
```

and put a [free Gemini API key](https://aistudio.google.com/apikey) in
`GOOGLE_API_KEY`. Nothing else in `.env` is required.

The first run downloads the embedding model (~470MB) from Hugging Face and
caches it under `~/.cache/huggingface`. Later runs are offline for that step.

## Running

```bash
python app.py
```

Gradio serves on `127.0.0.1` and scans ports 7860-7959 for a free one, so a
second instance will not fail on a taken port. Pin a specific port with
`GRADIO_SERVER_PORT` in `.env` if you need one.

## Tests

```bash
python -m pytest tests -q
```

The suite runs fully offline — no network, no API key, no model download. For a
live end-to-end check against a real video and the real Gemini API:

```bash
python smoke_test.py [YOUTUBE_URL] [QUESTION]
```

## Configuration

Everything tunable lives in [`config.py`](config.py):

| Setting | Default | Notes |
| --- | --- | --- |
| `TRANSCRIPT_LANGUAGES` | `("en", "ar")` | Preference order. Region variants (`ar-EG`, `en-US`) match on their base code. |
| `LOCAL_EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Multilingual by necessity — see below. |
| `GEMINI_MODEL_ID` | `gemini-flash-latest` | |
| `LLM_TEMPERATURE` | `0.2` | Low, to keep answers close to the transcript. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Overlap avoids splitting mid-thought. |
| `RETRIEVAL_K` / `RETRIEVAL_FETCH_K` | `4` / `20` | Fetch 20 candidates, return the 4 most relevant-and-diverse. |

Adding another language means adding its code to `TRANSCRIPT_LANGUAGES` and a
display name to `LANGUAGE_NAMES`. The embedding model already covers 50+
languages and the prompts ask the model to reply in the transcript's language,
so no other change is needed.

**On the embedding model:** the obvious choice, `all-MiniLM-L6-v2`, is
English-only. Arabic text embeds into meaningless vectors there and retrieval
returns arbitrary chunks — while summarization still looks fine, because it
sends the whole transcript and never touches the index. The multilingual model
costs a larger download (~470MB vs ~90MB) but maps both languages into the same
384-dimensional space.

## Known limits

- Videos with subtitles disabled, or that are private or age-restricted, have
  no retrievable transcript.
- Arabic auto-generated captions are noticeably rougher than English ones,
  which shows up in both summary and answer quality.
- Code-switched (mixed Arabic/English) videos fall back to whichever single
  track YouTube labeled the video with.
- The FAISS index cache is in-process memory only, so it is empty again after a
  restart.

## Layout

| File | Role |
| --- | --- |
| [`app.py`](app.py) | Gradio UI and request handlers |
| [`transcript.py`](transcript.py) | URL parsing, track selection, normalization |
| [`indexing.py`](indexing.py) | Chunking, FAISS index, retrieval |
| [`summarizer.py`](summarizer.py) | Prompts for summarization and Q&A |
| [`llm_interface.py`](llm_interface.py) | Cached model factories |
| [`config.py`](config.py) | All settings |
