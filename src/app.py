"""
AURA — Academic Retrieval & Analysis
Streamlit frontend connecting transcription + summarization.

Run:  streamlit run src/app.py
"""

from __future__ import annotations

import json
import time
import datetime
import threading
from pathlib import Path
import sys

import numpy as np
import streamlit as st

# ── Make sure project root is on sys.path when Streamlit runs this file ────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Local imports ───────────────────────────────────────────────────────────
from src import config  # noqa: F401  — loads .env
from src.database import (
    init_db,
    create_user,
    authenticate_user,
    create_subject,
    get_subjects,
    delete_subject,
    create_lecture,
    update_lecture,
    get_lectures,
    get_lecture,
    add_exam_paper,
    get_exam_papers,
    log_usage,
    get_usage_stats,
)
from src.transcription import (
    FasterWhisperTranscriber,
    AudioCapture,
    TranscriptSegment,
)
from src.summarization import (
    summarize_transcript,
    prepare_transcript,
    _fmt_time,
)

# ── Ensure DB exists ───────────────────────────────────────────────────────
init_db()

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AURA — Academic Retrieval & Analysis",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────────────────
# Custom CSS — student-friendly, modern look
# ────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Global ──────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ─────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}
section[data-testid="stSidebar"] * {
    color: #e8e8e8 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label {
    color: #a0a0c0 !important;
    font-weight: 500;
}

/* ── Header banner ───────────────────────────────────────── */
.aura-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    color: white;
}
.aura-header h1 {
    margin: 0; font-size: 1.8rem; font-weight: 700;
}
.aura-header p {
    margin: 0.3rem 0 0 0; opacity: 0.85; font-size: 0.95rem;
}

/* ── Stat cards ──────────────────────────────────────────── */
.stat-card {
    background: #f8f9fc;
    border: 1px solid #e2e6f0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.stat-card .label {
    font-size: 0.75rem; color: #6b7280;
    text-transform: uppercase; letter-spacing: 0.5px;
    font-weight: 600;
}
.stat-card .value {
    font-size: 1.5rem; font-weight: 700; color: #1a1a2e;
    margin-top: 0.2rem;
}

/* ── Recording pulse ─────────────────────────────────────── */
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.5); }
    70%  { box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }
    100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
}
.rec-dot {
    display: inline-block;
    width: 12px; height: 12px;
    background: #ef4444;
    border-radius: 50%;
    animation: pulse 1.5s infinite;
    vertical-align: middle;
    margin-right: 0.5rem;
}

/* ── Cards / containers ──────────────────────────────────── */
.section-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s;
}

/* ── Login page ──────────────────────────────────────────── */
.login-box {
    max-width: 380px;
    margin: 4rem auto;
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 2.5rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.06);
}
.login-box h2 {
    text-align: center; margin-bottom: 0.3rem;
}
.login-box .subtitle {
    text-align: center; color: #6b7280; font-size: 0.9rem;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# Session-state defaults
# ────────────────────────────────────────────────────────────────────────────

_DEFAULTS = {
    "user": None,
    "page": "dashboard",
    "recording": False,
    "rec_segments": [],
    "rec_buffer": [],
    "rec_elapsed": 0.0,
    "rec_capture": None,
    "rec_transcriber": None,
    "rec_stop_flag": None,
}

for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ────────────────────────────────────────────────────────────────────────────
# LOGIN / SIGNUP
# ────────────────────────────────────────────────────────────────────────────

def show_login():
    st.markdown("""
    <div style="text-align:center; margin-top:2rem;">
        <h1 style="font-size:2.5rem; margin-bottom:0;">🎓 AURA</h1>
        <p style="color:#6b7280; font-size:1.1rem;">Academic Retrieval & Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        tab_login, tab_signup = st.tabs(["🔑 Login", "✨ Create Account"])

        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter username")
                password = st.text_input("Password", type="password", placeholder="Enter password")
                submitted = st.form_submit_button("Login", use_container_width=True)
                if submitted:
                    if not username or not password:
                        st.error("Please fill in all fields.")
                    else:
                        user = authenticate_user(username, password)
                        if user:
                            st.session_state.user = user
                            st.rerun()
                        else:
                            st.error("Invalid username or password.")

        with tab_signup:
            with st.form("signup_form"):
                new_user = st.text_input("Choose a username", placeholder="Username")
                new_pass = st.text_input("Choose a password", type="password", placeholder="Password")
                confirm  = st.text_input("Confirm password", type="password", placeholder="Confirm")
                signed = st.form_submit_button("Create Account", use_container_width=True)
                if signed:
                    if not new_user or not new_pass:
                        st.error("Please fill in all fields.")
                    elif new_pass != confirm:
                        st.error("Passwords do not match.")
                    else:
                        uid = create_user(new_user, new_pass)
                        if uid:
                            st.success("Account created! You can now log in.")
                        else:
                            st.error("Username already taken.")


# ────────────────────────────────────────────────────────────────────────────
# HEADER + STATS BAR
# ────────────────────────────────────────────────────────────────────────────

def show_header():
    user = st.session_state.user
    st.markdown(f"""
    <div class="aura-header">
        <h1>🎓 AURA</h1>
        <p>Academic Retrieval & Analysis — Welcome back, <strong>{user['username']}</strong></p>
    </div>
    """, unsafe_allow_html=True)

    stats = get_usage_stats(user["id"])

    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        ("Lectures", stats.get("total_lectures", 0)),
        ("Transcriptions", stats.get("total_transcriptions", 0)),
        ("Summaries", stats.get("total_summaries", 0)),
        ("Tokens Used", f"{stats.get('total_tokens', 0):,}"),
        ("API Cost", f"${stats.get('total_cost', 0):.4f}"),
    ]
    for col, (label, value) in zip([c1, c2, c3, c4, c5], cards):
        col.markdown(f"""
        <div class="stat-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────────────────────────────────

def show_sidebar():
    user = st.session_state.user
    with st.sidebar:
        st.markdown("### 📚 Navigation")
        page = st.radio(
            "Go to",
            ["Dashboard", "Record Lecture", "Exam Papers & Syllabus"],
            label_visibility="collapsed",
        )
        st.session_state.page = page.lower().replace(" ", "_").replace("&", "and")

        st.markdown("---")
        st.markdown("### 📖 Your Subjects")

        subjects = get_subjects(user["id"])
        if subjects:
            for subj in subjects:
                st.markdown(f"• **{subj['name']}**")
        else:
            st.caption("No subjects yet — add one below.")

        st.markdown("---")
        with st.form("add_subject_form", clear_on_submit=True):
            st.markdown("##### ➕ Add Subject")
            name = st.text_input("Subject name", placeholder="e.g. Linear Algebra", label_visibility="collapsed")
            if st.form_submit_button("Add", use_container_width=True):
                if name.strip():
                    result = create_subject(user["id"], name.strip())
                    if result:
                        st.success(f"Added **{name.strip()}**")
                        st.rerun()
                    else:
                        st.warning("Subject already exists.")

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


# ────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ────────────────────────────────────────────────────────────────────────────

def show_dashboard():
    user = st.session_state.user
    subjects = get_subjects(user["id"])

    if not subjects:
        st.info("👋 Welcome to AURA! Start by adding a subject in the sidebar, then record your first lecture.")
        return

    st.markdown("### 📋 Recent Lectures by Subject")

    for subj in subjects:
        with st.expander(f"📘 **{subj['name']}**", expanded=False):
            lectures = get_lectures(subj["id"])
            if not lectures:
                st.caption("No lectures recorded yet.")
                continue

            for lec in lectures:
                col_t, col_d, col_a = st.columns([3, 2, 3])
                col_t.markdown(f"**{lec['title']}**")
                col_d.caption(lec.get("date", ""))

                btns = []
                if lec.get("short_summary_path") and Path(lec["short_summary_path"]).exists():
                    btns.append(("Short", lec["short_summary_path"]))
                if lec.get("long_summary_path") and Path(lec["long_summary_path"]).exists():
                    btns.append(("Detailed", lec["long_summary_path"]))
                if lec.get("transcript_path") and Path(lec["transcript_path"]).exists():
                    btns.append(("Transcript", lec["transcript_path"]))

                for label, fpath in btns:
                    text = Path(fpath).read_text(encoding="utf-8")
                    col_a.download_button(
                        f"📥 {label}",
                        data=text,
                        file_name=Path(fpath).name,
                        key=f"dl_{lec['id']}_{label}",
                    )

            # Delete subject (at bottom of expander)
            if st.button(f"Delete subject '{subj['name']}'", key=f"del_subj_{subj['id']}"):
                delete_subject(subj["id"])
                st.rerun()


# ────────────────────────────────────────────────────────────────────────────
# RECORD LECTURE
# ────────────────────────────────────────────────────────────────────────────

def _get_transcriber():
    """Lazy-load the transcriber once per session."""
    if st.session_state.rec_transcriber is None:
        with st.spinner("Loading ASR model (first time takes a moment)…"):
            st.session_state.rec_transcriber = FasterWhisperTranscriber(
                model_size="base.en", device="auto", compute_type="int8"
            )
    return st.session_state.rec_transcriber


def _recording_worker(
    capture: AudioCapture,
    transcriber: FasterWhisperTranscriber,
    stop_flag: threading.Event,
    segments_out: list,
    elapsed_out: list,
):
    """Background thread: capture audio → buffer → transcribe on silence/max."""
    SAMPLE_RATE = AudioCapture.SAMPLE_RATE
    BLOCK_SEC = AudioCapture.BLOCK_SECONDS
    SEGMENT_SEC = 10.0
    SILENCE_THRESH = 0.01
    SILENCE_DUR = 1.5

    buffer = []
    buffer_dur = 0.0
    elapsed = 0.0
    silent_chunks = 0
    max_silent = int(SILENCE_DUR / BLOCK_SEC)

    capture.start()
    try:
        while not stop_flag.is_set():
            chunk = capture.read_chunk(timeout=0.5)
            if chunk is None:
                continue

            chunk_dur = len(chunk) / SAMPLE_RATE
            buffer.append(chunk)
            buffer_dur += chunk_dur
            elapsed += chunk_dur
            elapsed_out.clear()
            elapsed_out.append(elapsed)

            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if rms < SILENCE_THRESH:
                silent_chunks += 1
            else:
                silent_chunks = 0

            should_flush = (
                (silent_chunks >= max_silent and buffer_dur > 2.0)
                or buffer_dur >= SEGMENT_SEC
            )
            if should_flush and buffer:
                audio = np.concatenate(buffer)
                buf_start = elapsed - buffer_dur
                segs = transcriber.transcribe_array(audio, SAMPLE_RATE)
                for seg in segs:
                    seg.start_time = round(buf_start + seg.start_time, 2)
                    seg.end_time = round(buf_start + seg.end_time, 2)
                    if seg.text.strip():
                        segments_out.append(seg)
                buffer.clear()
                buffer_dur = 0.0
                silent_chunks = 0

        # Flush remaining
        if buffer:
            audio = np.concatenate(buffer)
            buf_start = elapsed - buffer_dur
            segs = transcriber.transcribe_array(audio, SAMPLE_RATE)
            for seg in segs:
                seg.start_time = round(buf_start + seg.start_time, 2)
                seg.end_time = round(buf_start + seg.end_time, 2)
                if seg.text.strip():
                    segments_out.append(seg)
    finally:
        capture.stop()


def show_record_lecture():
    user = st.session_state.user
    subjects = get_subjects(user["id"])

    if not subjects:
        st.warning("Please add at least one subject in the sidebar before recording.")
        return

    st.markdown("### 🎙️ Record a Lecture")

    col_subj, col_title = st.columns(2)
    with col_subj:
        subject_names = [s["name"] for s in subjects]
        selected_subject = st.selectbox("Subject", subject_names)
    with col_title:
        lecture_title = st.text_input("Lecture title", placeholder="e.g. Week 5 — Eigenvalues")

    subject = next(s for s in subjects if s["name"] == selected_subject)

    # ── Upload tab vs Live tab ──────────────────────────────────────────
    tab_live, tab_upload = st.tabs(["🎤 Live Recording", "📁 Upload Audio File"])

    # ── LIVE RECORDING ──────────────────────────────────────────────────
    with tab_live:
        if not st.session_state.recording:
            if st.button("🔴 Start Recording", use_container_width=True, type="primary"):
                if not lecture_title.strip():
                    st.error("Please enter a lecture title first.")
                else:
                    transcriber = _get_transcriber()
                    capture = AudioCapture(device=None)
                    stop_flag = threading.Event()

                    # Shared mutable lists (thread-safe for append/read)
                    segments_out = []
                    elapsed_out = [0.0]

                    worker = threading.Thread(
                        target=_recording_worker,
                        args=(capture, transcriber, stop_flag, segments_out, elapsed_out),
                        daemon=True,
                    )
                    worker.start()

                    st.session_state.recording = True
                    st.session_state.rec_stop_flag = stop_flag
                    st.session_state.rec_segments = segments_out
                    st.session_state.rec_elapsed = elapsed_out
                    st.session_state.rec_capture = capture
                    st.session_state._rec_worker = worker
                    st.rerun()
        else:
            # ── Currently recording ─────────────────────────────────────
            st.markdown('<p><span class="rec-dot"></span> <strong>Recording in progress…</strong></p>',
                        unsafe_allow_html=True)

            segments = st.session_state.rec_segments
            elapsed = st.session_state.rec_elapsed

            # Show live stats
            el_val = elapsed[0] if elapsed else 0.0
            m, s = divmod(int(el_val), 60)
            h, m = divmod(m, 60)
            st.caption(f"⏱️ Elapsed: {h:02d}:{m:02d}:{s:02d}  •  Segments: {len(segments)}")

            # Show latest transcript lines
            if segments:
                st.markdown("**Live transcript (latest):**")
                for seg in segments[-5:]:
                    st.caption(f"`[{_fmt_time(seg.start_time)}]` {seg.text}")

            col_stop, col_refresh = st.columns(2)
            with col_refresh:
                st.button("🔄 Refresh", use_container_width=True)
            with col_stop:
                if st.button("⬛ Stop & Summarize", use_container_width=True, type="primary"):
                    # Signal the worker thread to stop
                    st.session_state.rec_stop_flag.set()
                    st.session_state._rec_worker.join(timeout=10)

                    segments = list(st.session_state.rec_segments)
                    el_val = st.session_state.rec_elapsed[0] if st.session_state.rec_elapsed else 0.0

                    st.session_state.recording = False

                    if not segments:
                        st.warning("No speech detected.")
                        return

                    _save_and_summarize(
                        segments, el_val, subject, lecture_title.strip(), user
                    )

    # ── UPLOAD ──────────────────────────────────────────────────────────
    with tab_upload:
        uploaded = st.file_uploader(
            "Upload a lecture recording",
            type=["mp3", "m4a", "wav", "ogg", "flac"],
        )
        if uploaded and st.button("▶️ Transcribe & Summarize", use_container_width=True, type="primary"):
            if not lecture_title.strip():
                st.error("Please enter a lecture title first.")
            else:
                _handle_uploaded_audio(uploaded, subject, lecture_title.strip(), user)


def _save_and_summarize(
    segments: list[TranscriptSegment],
    elapsed: float,
    subject: dict,
    title: str,
    user: dict,
):
    """Persist transcript → run summarization → save to DB."""
    from dataclasses import asdict

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("data/transcripts")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save transcript JSON
    transcript_path = out_dir / f"transcript_{ts}.json"
    with open(transcript_path, "w") as f:
        json.dump(
            {
                "created": ts,
                "duration_seconds": round(elapsed, 2),
                "segments": [asdict(s) for s in segments],
            },
            f, indent=2,
        )

    st.success(f"✅ Transcript saved — {len(segments)} segments, {_fmt_time(elapsed)}")

    # Run summarization
    with st.spinner("✨ Generating summaries…"):
        result = summarize_transcript(
            transcript_path=str(transcript_path),
            output_dir="data/outputs",
        )

    # Log API usage
    usage = result.get("usage", {})
    if usage:
        log_usage(
            user_id=user["id"],
            action="summarize",
            model=usage.get("model", "gpt-4o-mini"),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    # Create DB lecture record
    lecture_id = create_lecture(
        subject_id=subject["id"],
        title=title,
        duration_seconds=elapsed,
        transcript_path=str(transcript_path),
        short_summary_path=str(result["short"]),
        long_summary_path=str(result["long"]),
    )

    st.success("✅ Summaries generated!")

    # Show results inline
    _display_results(result)


def _handle_uploaded_audio(uploaded_file, subject: dict, title: str, user: dict):
    """Save uploaded file → batch transcribe → summarize."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_dir = Path("data/audio")
    audio_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(uploaded_file.name).suffix
    audio_path = audio_dir / f"upload_{ts}{suffix}"
    audio_path.write_bytes(uploaded_file.getvalue())

    # Transcribe
    with st.spinner("🔊 Transcribing audio (this may take a while for long files)…"):
        transcriber = _get_transcriber()
        segments = transcriber.transcribe_file(str(audio_path))

    if not segments:
        st.warning("No speech detected in the file.")
        return

    # Compute duration from segments
    elapsed = segments[-1].end_time if segments else 0.0

    _save_and_summarize(segments, elapsed, subject, title, user)


def _display_results(result: dict):
    """Show short + long summaries in expandable sections."""
    tab_short, tab_long = st.tabs(["📝 Short Summary", "📖 Detailed Summary"])

    short_path = result.get("short")
    long_path = result.get("long")

    with tab_short:
        if short_path and Path(short_path).exists():
            st.markdown(Path(short_path).read_text(encoding="utf-8"))
        else:
            st.info("Short summary not available.")

    with tab_long:
        if long_path and Path(long_path).exists():
            st.markdown(Path(long_path).read_text(encoding="utf-8"))
        else:
            st.info("Detailed summary not available.")


# ────────────────────────────────────────────────────────────────────────────
# EXAM PAPERS & SYLLABUS  (template — non-functional backend)
# ────────────────────────────────────────────────────────────────────────────

def show_exam_papers_and_syllabus():
    user = st.session_state.user
    subjects = get_subjects(user["id"])

    st.markdown("### 📄 Exam Papers & Syllabus")

    if not subjects:
        st.warning("Add a subject first to upload materials here.")
        return

    selected = st.selectbox("Select Subject", [s["name"] for s in subjects], key="ep_subj")
    subject = next(s for s in subjects if s["name"] == selected)

    tab_syllabus, tab_exams = st.tabs(["📗 Course Syllabus", "📝 Past Year Papers"])

    with tab_syllabus:
        st.markdown("""
        <div class="section-card">
            <h4>📗 Upload Course Syllabus</h4>
            <p style="color:#6b7280; font-size:0.9rem;">
                Upload your course syllabus (PDF or text). AURA will use it to map lecture
                topics to official course topics for better-organised summaries.
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_syl = st.file_uploader(
            "Upload syllabus",
            type=["pdf", "txt", "docx"],
            key="syl_upload",
        )
        if uploaded_syl:
            if st.button("📤 Upload Syllabus", key="btn_syl"):
                st.info("🚧 Syllabus parsing is coming in a future update. The file has been noted.")

        if subject.get("syllabus_path"):
            st.success(f"Current syllabus: `{subject['syllabus_path']}`")
        else:
            st.caption("No syllabus uploaded yet.")

    with tab_exams:
        st.markdown("""
        <div class="section-card">
            <h4>📝 Upload Past Year Question Papers</h4>
            <p style="color:#6b7280; font-size:0.9rem;">
                Upload exam PDFs and AURA will parse questions, identify topic patterns,
                and highlight exam-relevant concepts after each lecture.
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_exams = st.file_uploader(
            "Upload question papers",
            type=["pdf"],
            accept_multiple_files=True,
            key="exam_upload",
        )
        if uploaded_exams:
            if st.button("📤 Upload Papers", key="btn_exam"):
                saved = 0
                for f in uploaded_exams:
                    dest = Path("data/exam_papers") / f.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(f.getvalue())
                    add_exam_paper(subject["id"], f.name, str(dest))
                    saved += 1
                st.success(f"✅ {saved} paper(s) uploaded. Parsing coming soon!")
                st.rerun()

        # Show existing papers
        papers = get_exam_papers(subject["id"])
        if papers:
            st.markdown("**Uploaded papers:**")
            for p in papers:
                st.caption(f"📄 {p['name']}  —  uploaded {p.get('uploaded_at', '')}")
        else:
            st.caption("No exam papers uploaded yet.")


# ────────────────────────────────────────────────────────────────────────────
# MAIN ROUTER
# ────────────────────────────────────────────────────────────────────────────

def main():
    if st.session_state.user is None:
        show_login()
        return

    show_sidebar()
    show_header()

    page = st.session_state.page
    if page == "record_lecture":
        show_record_lecture()
    elif page == "exam_papers_and_syllabus":
        show_exam_papers_and_syllabus()
    else:
        show_dashboard()


if __name__ == "__page__":
    main()

main()
