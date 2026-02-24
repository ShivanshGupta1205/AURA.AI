"""
Short and long summary generation using OpenAI gpt-4o-mini (text-only).

Design principles
─────────────────
- Single API call returns BOTH summaries in a structured JSON response
  (zero redundant calls, minimal token usage).
- All data pre-processing (segment merging, filtering, formatting) happens
  locally BEFORE the API call so the model only sees clean, compact input.
- OpenAI structured output (JSON schema) is used so the response needs no
  fragile string parsing.
- Later, pass a ``syllabus`` list to lock topic headers to the subject's
  official taxonomy instead of letting the model choose freely.

Usage
─────
  # from another module
  from src.summarization import summarize_transcript
  out = summarize_transcript("data/transcripts/transcript_20260223_224820.json")

  # CLI
  python -m src.summarization --transcript data/transcripts/transcript_20260223_224820.json
  python -m src.summarization --transcript ... --syllabus "Calculus,Linear Algebra,..."
"""

from __future__ import annotations

import os
import json
import argparse
import datetime
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load .env from the project root so OPENAI_API_KEY is available
load_dotenv()


# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────

MODEL = "gpt-4o-mini"

# Segments with avg_logprob below this are considered low-confidence and
# are kept but flagged in the prompt so the model can handle them carefully.
CONFIDENCE_WARN_THRESHOLD = -0.6

# If the full transcript exceeds this word count we trim redundant filler
# before sending (keeps prompt lean without losing content).
MAX_WORDS_BEFORE_TRIM = 12_000


# ────────────────────────────────────────────────────────────────────────────
# JSON schema for structured output
# ────────────────────────────────────────────────────────────────────────────

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "topics": {
            "type": "array",
            "description": "Topics covered in the lecture, in the order they appeared.",
            "items": {
                "type": "object",
                "properties": {
                    "header": {
                        "type": "string",
                        "description": "A concise topic name used as the section header.",
                    },
                    "start_time": {
                        "type": "number",
                        "description": "Approx lecture time (seconds) when this topic began.",
                    },
                    "end_time": {
                        "type": "number",
                        "description": "Approx lecture time (seconds) when this topic ended.",
                    },
                    "short_bullets": {
                        "type": "array",
                        "description": "3-6 crisp bullet points for quick revision.",
                        "items": {"type": "string"},
                    },
                    "long_paragraphs": {
                        "type": "array",
                        "description": (
                            "2-4 detailed paragraphs covering key explanations, "
                            "definitions, derivations, and examples for this topic."
                        ),
                        "items": {"type": "string"},
                    },
                    "key_terms": {
                        "type": "array",
                        "description": "Important vocabulary or formulae introduced.",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "header",
                    "start_time",
                    "end_time",
                    "short_bullets",
                    "long_paragraphs",
                    "key_terms",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["topics"],
    "additionalProperties": False,
}


# ────────────────────────────────────────────────────────────────────────────
# Prompts
# ────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert academic note-taker assistant.
    Your job is to analyse a university-level lecture transcript and produce
    structured, high-quality study material for a student.

    You will output a JSON object matching the provided schema exactly.
    No prose outside the JSON.

    TOPIC HEADERS
    ─────────────
    {topic_instruction}

    CONTENT RULES
    ─────────────
    • Preserve technical accuracy above all else.
    • For mathematical or scientific content: reproduce symbols, formulae, and
      variable names exactly as they appear in the transcript. If an equation
      looks corrupted or incomplete in the transcript, mark it with [UNCLEAR]
      and show the verbatim phrase — never invent or guess an equation.
    • Segments marked [LOW CONFIDENCE] may contain transcription errors.
      Handle them cautiously; flag genuinely ambiguous content with [UNCLEAR].
    • Do not pad with filler phrases like "In this lecture we learned…".
    • Each short_bullet must be a standalone, self-contained fact or insight.
    • long_paragraphs should read like polished academic notes — explain the
      "why" and "how", not just the "what".
    • key_terms should list terms in the form  "term — definition"  or just
      the term if no definition was given in the lecture.

    COVERAGE
    ────────
    • Cover the full lecture, not just the opening minutes.
    • If the lecture had only one clear theme, produce one topic block.
    • If transitions between topics are unclear, err on the side of fewer,
      broader topics rather than many thin ones.
""").strip()

USER_PROMPT_TEMPLATE = textwrap.dedent("""
    LECTURE METADATA
    ────────────────
    Duration : {duration}
    Segments : {n_segments}  ({n_flagged} low-confidence)
    Word count: ~{word_count}

    TRANSCRIPT (format: [HH:MM:SS] text)
    ─────────────────────────────────────
    {transcript_text}
""").strip()


# ────────────────────────────────────────────────────────────────────────────
# Pre-processing helpers
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class PreparedTranscript:
    """Everything the caller needs to build the API request."""
    formatted_text: str       # clean, time-annotated transcript string
    duration_str: str         # e.g. "01:12:45"
    n_segments: int
    n_flagged: int            # low-confidence segment count
    word_count: int
    segment_times: list[tuple[float, float]]  # [(start, end), …]


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _merge_short_segments(
    segments: list[dict], gap_seconds: float = 1.5
) -> list[dict]:
    """
    Merge consecutive segments that are very short (< 3 words) or separated
    by tiny gaps into one segment.  Reduces fragmentation without data loss.
    """
    if not segments:
        return []
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg["start_time"] - prev["end_time"]
        prev_words = len(prev["text"].split())
        if gap <= gap_seconds and prev_words < 6:
            prev["text"] += " " + seg["text"]
            prev["end_time"] = seg["end_time"]
            # keep the lower (worse) confidence so flagging stays conservative
            if seg.get("confidence") is not None and prev.get("confidence") is not None:
                prev["confidence"] = min(prev["confidence"], seg["confidence"])
        else:
            merged.append(seg.copy())
    return merged


def prepare_transcript(transcript_path: str | Path) -> PreparedTranscript:
    """
    Load a transcript JSON, clean and compact it, and return a
    ``PreparedTranscript`` ready to be injected into the prompt.

    All heavy lifting is done here — nothing is computed inside the prompt
    builder or after the API call.
    """
    with open(transcript_path) as f:
        data = json.load(f)

    raw_segments: list[dict] = data.get("segments", [])
    duration_seconds: float = data.get("duration_seconds", 0.0)

    if not raw_segments:
        raise ValueError(f"No segments found in {transcript_path}")

    # 1. Merge short/fragmented segments
    segments = _merge_short_segments(raw_segments, gap_seconds=1.5)

    # 2. Identify low-confidence segments
    flagged = {
        i for i, s in enumerate(segments)
        if (s.get("confidence") or 0.0) < CONFIDENCE_WARN_THRESHOLD
    }

    # 3. Build formatted transcript string
    lines: list[str] = []
    total_words = 0
    for i, seg in enumerate(segments):
        ts = _fmt_time(seg["start_time"])
        flag = " [LOW CONFIDENCE]" if i in flagged else ""
        text = seg["text"].strip()
        total_words += len(text.split())
        lines.append(f"[{ts}]{flag} {text}")

    # 4. Trim if very long (lectures > ~12k words are rare but possible)
    #    Strategy: thin out LOw-confidence lines first, then uniformly sample.
    if total_words > MAX_WORDS_BEFORE_TRIM:
        # Remove pure low-confidence single-word segments
        lines = [
            l for i, l in enumerate(lines)
            if i not in flagged or len(segments[i]["text"].split()) > 2
        ]

    formatted_text = "\n".join(lines)

    # 5. Duration string from actual segment data if metadata missing
    if duration_seconds == 0 and segments:
        duration_seconds = segments[-1]["end_time"]

    return PreparedTranscript(
        formatted_text=formatted_text,
        duration_str=_fmt_time(duration_seconds),
        n_segments=len(segments),
        n_flagged=len(flagged),
        word_count=total_words,
        segment_times=[(s["start_time"], s["end_time"]) for s in segments],
    )


# ────────────────────────────────────────────────────────────────────────────
# Prompt assembly
# ────────────────────────────────────────────────────────────────────────────

def _build_topic_instruction(syllabus: list[str] | None) -> str:
    if syllabus:
        topics_str = "\n    ".join(f"• {t}" for t in syllabus)
        return (
            "You MUST use only the following pre-defined topic names as headers.\n"
            "    Pick the best matching topic from the list for each section.\n"
            "    If content does not fit any topic, use 'General / Miscellaneous'.\n\n"
            f"    Allowed topics:\n    {topics_str}"
        )
    return (
        "Choose concise, descriptive topic names that honestly reflect the\n"
        "    lecture content (e.g. 'Newton's Laws of Motion', 'Binary Search Trees').\n"
        "    Later these will be locked to the subject syllabus."
    )


def build_messages(
    prepared: PreparedTranscript,
    syllabus: list[str] | None = None,
) -> list[dict]:
    """Assemble the messages list for the OpenAI chat completions call."""
    system = SYSTEM_PROMPT.format(
        topic_instruction=_build_topic_instruction(syllabus)
    )
    user = USER_PROMPT_TEMPLATE.format(
        duration=prepared.duration_str,
        n_segments=prepared.n_segments,
        n_flagged=prepared.n_flagged,
        word_count=prepared.word_count,
        transcript_text=prepared.formatted_text,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ────────────────────────────────────────────────────────────────────────────
# API call
# ────────────────────────────────────────────────────────────────────────────

def call_openai(messages: list[dict], api_key: str | None = None) -> tuple[dict, dict]:
    """
    Single structured-output API call → returns (parsed_result, usage_info).

    Using ``response_format`` with a JSON schema guarantees the model outputs
    valid JSON that matches our schema — no retry/parse loops needed.

    usage_info keys: prompt_tokens, completion_tokens, total_tokens, model
    """
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "lecture_summary",
                "strict": True,
                "schema": RESPONSE_SCHEMA,
            },
        },
        temperature=0.3,      # low temp = consistent, factual output
        max_tokens=4096,       # enough for a full lecture; raise if needed
    )

    raw = response.choices[0].message.content
    usage = response.usage
    usage_info = {
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "total_tokens": usage.total_tokens if usage else 0,
        "model": MODEL,
    }
    return json.loads(raw), usage_info


# ────────────────────────────────────────────────────────────────────────────
# Markdown rendering
# ────────────────────────────────────────────────────────────────────────────

def render_short_summary(topics: list[dict], meta: PreparedTranscript) -> str:
    """Render the short (bullet-point) summary as Markdown."""
    lines = [
        "# Lecture Summary — Quick Revision",
        f"> Duration: {meta.duration_str}  |  Topics: {len(topics)}",
        "",
    ]
    for topic in topics:
        start = _fmt_time(topic["start_time"])
        end   = _fmt_time(topic["end_time"])
        lines.append(f"## {topic['header']}")
        lines.append(f"*[{start} → {end}]*")
        lines.append("")
        for bullet in topic["short_bullets"]:
            lines.append(f"- {bullet}")
        if topic.get("key_terms"):
            lines.append("")
            lines.append("**Key terms:** " + " · ".join(topic["key_terms"]))
        lines.append("")
    return "\n".join(lines)


def render_long_summary(topics: list[dict], meta: PreparedTranscript) -> str:
    """Render the detailed (paragraph) summary as Markdown."""
    lines = [
        "# Lecture Summary — Detailed Notes",
        f"> Duration: {meta.duration_str}  |  Topics: {len(topics)}",
        "",
        "---",
        "",
    ]
    for topic in topics:
        start = _fmt_time(topic["start_time"])
        end   = _fmt_time(topic["end_time"])
        lines.append(f"## {topic['header']}")
        lines.append(f"*[{start} → {end}]*")
        lines.append("")
        for para in topic["long_paragraphs"]:
            lines.append(para)
            lines.append("")
        if topic.get("key_terms"):
            lines.append("### Key Terms & Definitions")
            for term in topic["key_terms"]:
                lines.append(f"- {term}")
            lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
# Persistence
# ────────────────────────────────────────────────────────────────────────────

def save_summaries(
    short_md: str,
    long_md: str,
    output_dir: str | Path,
    transcript_path: str | Path,
) -> tuple[Path, Path]:
    """
    Save short and long summaries to Markdown files.
    File names echo the source transcript timestamp for easy traceability.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Derive a base name from the source transcript filename
    src_stem = Path(transcript_path).stem          # e.g. transcript_20260223_224820
    ts_part = src_stem.replace("transcript_", "")  # e.g. 20260223_224820

    short_path = out / f"short_summary_{ts_part}.md"
    long_path  = out / f"long_summary_{ts_part}.md"

    short_path.write_text(short_md, encoding="utf-8")
    long_path.write_text(long_md,   encoding="utf-8")

    return short_path, long_path


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def summarize_transcript(
    transcript_path: str | Path,
    output_dir: str | Path = "data/outputs",
    syllabus: list[str] | None = None,
    api_key: str | None = None,
) -> dict[str, Path]:
    """
    Main entry point.

    Parameters
    ──────────
    transcript_path : path to the JSON transcript produced by transcription.py
    output_dir      : where to save the two Markdown files
    syllabus        : optional list of topic names to pin headers to
    api_key         : OpenAI API key (falls back to OPENAI_API_KEY env var)

    Returns
    ───────
    {"short": Path, "long": Path, "usage": dict}
    """

    print(f"[SUMM] Loading transcript: {transcript_path}")

    # ── 1. Pre-process (all CPU work, no API) ─────────────────────────────
    prepared = prepare_transcript(transcript_path)
    print(
        f"[SUMM] Prepared  segments={prepared.n_segments} "
        f"words≈{prepared.word_count}  flagged={prepared.n_flagged}"
    )

    # ── 2. Build messages (prompt assembly) ───────────────────────────────
    messages = build_messages(prepared, syllabus=syllabus)

    # ── 3. One API call ───────────────────────────────────────────────────
    print(f"[SUMM] Calling {MODEL}…")
    result, usage_info = call_openai(messages, api_key=api_key)
    topics: list[dict] = result["topics"]
    print(f"[SUMM] Received {len(topics)} topic(s) from model.")
    print(f"[SUMM] Tokens — prompt={usage_info['prompt_tokens']}  "
          f"completion={usage_info['completion_tokens']}  "
          f"total={usage_info['total_tokens']}")

    # ── 4. Render ─────────────────────────────────────────────────────────
    short_md = render_short_summary(topics, prepared)
    long_md  = render_long_summary(topics, prepared)

    # ── 5. Save ───────────────────────────────────────────────────────────
    short_path, long_path = save_summaries(
        short_md, long_md, output_dir, transcript_path
    )
    print(f"[SUMM] ✓ Short summary → {short_path}")
    print(f"[SUMM] ✓ Long  summary → {long_path}")

    return {"short": short_path, "long": long_path, "usage": usage_info}


# ────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AURA — Generate short & long lecture summaries from a transcript"
    )
    parser.add_argument(
        "--transcript",
        required=True,
        help="Path to transcript JSON (produced by transcription.py)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/outputs",
        help="Directory to save summaries (default: data/outputs)",
    )
    parser.add_argument(
        "--syllabus",
        default=None,
        help=(
            "Comma-separated list of topic names to use as locked headers. "
            "Example: 'Calculus,Linear Algebra,Probability'"
        ),
    )
    args = parser.parse_args()

    syllabus = (
        [t.strip() for t in args.syllabus.split(",") if t.strip()]
        if args.syllabus else None
    )

    summarize_transcript(
        transcript_path=args.transcript,
        output_dir=args.output_dir,
        syllabus=syllabus,
    )


if __name__ == "__main__":
    main()
