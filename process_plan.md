
# Aura — Academic Retrieval & Analysis (POC Technical Plan)

## Overview

Aura is a local-first academic copilot that helps students turn long lectures into actionable study material. The POC supports multiple ingestion modes (live + recorded), while keeping the *analysis outputs* (summaries/resources/exam-focus) post-lecture:

1. Capture and transcribe lectures into timestamped text (audio stays local):
	- real-time transcription during live, in-person classes (mic capture)
	- real-time transcription for online lectures (system audio capture via loopback device)
	- batch transcription from uploaded recordings (mp3/m4a/wav)
2. Generate:
	 - Short summary (quick revision)
	 - Long summary (topic-by-topic)
	 - Additional resources (curated + real web search with citations)
3. In Phase 2, ingest past-year exam papers (PDFs), extract questions, build an exam-weighted topic model per subject, and then cross-reference each lecture against exam-relevant concepts.

The baseline LLM for reasoning/summarization is the OpenAI API, but only transcript + document text is sent (no raw audio).

## POC Goals

- Reliable lecture transcription for long audio files (batch) and live sessions (real-time).
- Usable summaries that cite where information came from (timestamps and slide/page references).
- A practical way to recover “visual context” by allowing PPT/PDF uploads.
- Exam relevance analysis that highlights high-frequency topics and surfaces similar past questions.

## Key Product Decisions (current)

- Deployment: local-first on macOS (single user).
- Audio input supports three paths:
	- Live (in-person): mic capture with real-time transcription
	- Live (online): system-audio capture (Zoom/Teams) using a macOS loopback device (e.g., BlackHole/Loopback) with real-time transcription
	- Recorded: upload an audio file for batch transcription
- Privacy: audio remains local; only extracted text is sent to OpenAI.
- Resources: combination of curated sources and real web search with citations.

## Security Note (do first)

- Never commit API keys.
- If an OpenAI key exists in a tracked `.env` file, rotate/revoke it immediately and remove it from git history if the repo was ever pushed.

## High-Level Architecture

### Pipeline (per lecture)

1. Ingest audio in one of three modes:
	- Live mic capture
	- Live system-audio capture (loopback input device)
	- Upload recorded file
2. Run local ASR:
	- live mode: streaming transcription into incremental timestamped segments
	- batch mode: full-file transcription into timestamped segments
3. (Optional) Ingest slides/handouts (PPT/PDF) → text chunks per slide/page.
4. Align transcript segments ↔ slide/page chunks via embeddings retrieval.
5. Summarize and structure outputs using OpenAI (text-only):
	 - Short summary
	 - Long summary (topic breakdown)
	 - Additional resources (curated + search + citations)
6. Persist all artifacts locally for later retrieval and “exam focus” analysis.

### Phase 2 (exam papers)

1. Ingest exam PDF(s) → extracted structured text (OCR/layout-aware).
2. Parse into question objects (number, subparts, marks if available, type).
3. Map questions to a controlled topic taxonomy per subject.
4. Compute exam-weighting per topic (frequency, optionally marks-weighted).
5. During lecture summarization, retrieve and display exam-relevant concepts and example questions.

## Recommended Tech Stack (POC)

### App + Orchestration

- Python 3.13
- Streamlit for UI (fast iteration)
- Background jobs: simple local queue (threads/process) or a lightweight task runner pattern (no need for Celery in POC)

Audio capture (local-first):

- Mic capture: `streamlit-webrtc` (browser mic) or `sounddevice`/`pyaudio` (device input)
- System audio (Zoom/Teams): macOS loopback device (e.g., BlackHole/Loopback) routed as an input device

### Speech-to-Text (local)

- POC default on macOS: `faster-whisper` (batch, timestamps, good quality on CPU)
- Optional later “power mode”: NVIDIA Parakeet/NeMo on a CUDA GPU machine

### Documents (slides/PDF)

- `marker` for PDF/PPT extraction into Markdown/JSON (layout + OCR)
- Convert PPT to PDF/images if needed before extraction (as preprocessing)

### Storage

- SQLite for metadata and structured objects
- Local vector store:
	- Chroma or LanceDB

### LLM (text-only)

- OpenAI API for:
	- topic segmentation/labeling
	- summarization (short + long)
	- exam question structuring (from extracted text)
	- topic mapping (question → taxonomy) with confidence

### Web resources

- Curated sources via official APIs where possible (e.g., Wikipedia)
- Real web search via a search API (Brave/Bing/SerpAPI) to avoid scraping
- Store URLs + short rationale + which lecture topic they support

## Data Model (minimal)

### Core

- Subject
	- id, name, optional syllabus outline / taxonomy
- Lecture
	- id, subject_id, date, title
- Artifact
	- lecture_id, type (audio, transcript, slides, exam_paper), path, created_at
- TranscriptSegment
	- lecture_id, start_time, end_time, text, confidence (if available)
- SlideChunk
	- lecture_id, slide_or_page_id, text, source_doc, confidence/quality metadata
- DerivedOutput
	- lecture_id, type (short_summary, long_summary, resources, exam_focus), json/text, created_at

### Phase 2

- ExamPaper
	- subject_id, term/year, source file, created_at
- ExamQuestion
	- exam_paper_id, question_id, text, parts, marks (optional), type, extracted_confidence
- Topic
	- subject_id, topic_id (stable), name, parent_topic_id (optional)
- QuestionTopicMap
	- exam_question_id, topic_id, confidence
- TopicStats
	- topic_id, frequency, marks_total (optional), last_updated

## Handling Your Key Concerns

### 1) Math-heavy subjects (equations, derivations, symbols)

Reality check: audio-only ASR frequently corrupts equations. The POC should optimize for honesty and traceability.

POC strategy:

- Preserve verbatim transcript segments with timestamps.
- If slides/handouts exist, treat them as the source of truth for equations.
- In summaries, never “invent” equations. When uncertain:
	- show the closest transcript quote
	- cite the timestamp range
	- attach any matching slide/page snippet if available
- Add a math glossary output per lecture (variables and definitions) sourced from transcript + slides, and mark low-confidence items.

### 2) Blackboard/PPT visual context

POC strategy:

- Optional “Upload slides/handout PDF” is a first-class input.
- Extract slide/page chunks and align them to transcript segments using embeddings.
- Long summary topics cite both:
	- transcript timestamps
	- slide/page identifiers (when available)

This makes the output auditable and much more useful than audio-only notes.

### 3) Exam paper variety (MSQ/long answer/math/diagrams)

POC strategy:

- Use layout-aware PDF extraction (marker) with OCR for scanned papers.
- Convert extracted text into structured questions with an LLM step (text-only).
- Keep confidence scores and allow “unknown/unparsed” questions rather than forcing bad structure.
- For diagrams, store references to the source page image (even if you cannot interpret it perfectly in POC).

### 4) Real-time ingestion vs post-lecture analysis

Recommended split (fits your requirements and reduces complexity):

- Real-time: capture audio and generate an on-screen rolling transcript (and optionally “live highlights” like key terms).
- Post-lecture: run the full multi-pass pipeline for quality outputs (topic segmentation → short/long summary → resources → exam focus).

This keeps the live experience responsive while preserving accuracy for the final study artifacts.

## Step-by-Step Implementation Guide (POC Milestones)

### Milestone 0 — Repo hygiene and basic scaffolding

- Add secret-safe configuration pattern (env vars, example env file).
- Add a local data folder structure (raw inputs vs derived outputs).
- Add minimal logging and run configuration.

### Milestone 1 — Audio ingestion UI

- Streamlit page: create/select Subject, create Lecture
- Inputs:
	- Live (in-person): select mic input → Start/Stop live session
	- Live (online): select loopback/system-audio input device → Start/Stop live session
	- Recorded: upload audio file (mp3/m4a/wav)

Notes:

- On macOS, “system audio capture” typically requires a virtual/loopback audio driver (e.g., BlackHole) so the app can read system output as an input device.
- For online lectures, keep it simple in POC: instruct users to route Zoom/Teams output to the loopback device and select that device in Aura.

### Milestone 2 — Local transcription (ASR)

- Implement a `Transcriber` interface:
	- batch: `transcribe_file(audio_path) → list[segments]`
	- live: `start_stream(...)`, `feed_audio_chunk(...) → partial_segments`, `finalize_stream() → final_segments`
- Add:
	- chunking + overlap (batch)
	- VAD + short rolling windows (live)
	- transcript export (JSON + readable text)

### Milestone 3 — Slides/PDF ingestion (optional)

- Upload slides/handouts (PDF; PPT via conversion step)
- Run extraction → slide/page chunks
- Persist chunks + metadata

### Milestone 4 — Alignment (transcript ↔ slides)

- Embed transcript segments and slide chunks
- For each topic window (or segment group), retrieve top-k slide chunks
- Store alignment links for citations

### Milestone 5 — Summaries (OpenAI text-only)

- Short summary:
	- 10–15 bullets, focused on quick revision
- Long summary:
	- topic headings
	- key definitions
	- important steps/derivations (only when sourced)
	- citations: timestamp ranges + slide/page ids
- Output format: structured JSON (for UI) plus readable Markdown

### Milestone 6 — Additional resources (curated + search)

- For each topic:
	- generate search queries
	- retrieve results via API
	- filter/rank (duplicates, low-quality domains)
- Output:
	- links
	- 1–2 sentence “why it helps”
	- difficulty level tags (intro/intermediate/advanced)

### Milestone 7 — Phase 2: exam papers ingestion

- Upload exam PDFs
- Extract via marker
- Parse into structured questions with LLM
- Create a subject topic taxonomy (start manually from syllabus)
- Map questions → topics with confidence
- Compute topic frequency / (optional) marks weighting

### Milestone 8 — Exam Focus per lecture

- For each lecture topic:
	- retrieve relevant exam topics/questions
	- compute “exam relevance score” from frequency + similarity
- Output section:
	- “High exam-relevance concepts covered today”
	- representative past questions (with paper/page reference)
	- practice checklist

## Verification Checklist

### Functional

- Run end-to-end on:
	- one math-heavy lecture
	- one slide-heavy lecture
- Confirm outputs appear in the UI without manual intervention.

### Quality gates

- Transcript:
	- correct timestamps
	- reasonable speaker content fidelity
- Summaries:
	- each major claim is traceable to timestamps/slides
	- math content is not hallucinated (uncertain items are flagged)
- Exam focus:
	- top surfaced topics match obvious exam patterns from papers

### Performance

- Measure total runtime for a 60–90 minute recording on your Mac.
- Confirm the pipeline can run asynchronously while you keep using the UI.

## Risks and Mitigations

- Math fidelity from audio-only: mitigate by requiring/encouraging slide uploads and by conservative summarization with citations.
- OCR/layout failures in exam PDFs: mitigate via confidence scoring and allowing partial extraction.
- Topic taxonomy drift: mitigate by using a controlled topic list per subject (syllabus-based) instead of free-form labels.

## Future Extensions (after POC)

- Optional GPU transcription backend (Parakeet) for speed and quality.
- Better slide alignment via timestamped slide-change detection (if you later support screen recording).
- Knowledge graph features: entities and relationships across lectures and exam questions.
- Mobile capture workflow (record on phone, sync to desktop POC).
