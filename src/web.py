"""
AURA — Academic Retrieval & Analysis
Flask web frontend (replaces Streamlit for a more robust, deployable UI).

Run:  python src/web.py
Then:  open http://localhost:5000
"""

from __future__ import annotations

import json
import os
import sys
import datetime
import threading
import time
from dataclasses import asdict
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, jsonify, send_file,
)
import numpy as np

from src import config  # noqa: F401  — loads .env
from src.database import (
    init_db, create_user, authenticate_user,
    create_subject, get_subjects, delete_subject,
    create_lecture, update_lecture, get_lectures, get_lecture,
    add_exam_paper, get_exam_papers,
    log_usage, get_usage_stats,
)
from src.transcription import (
    FasterWhisperTranscriber, AudioCapture, TranscriptSegment,
)
from src.summarization import summarize_transcript, _fmt_time

# ── App setup ───────────────────────────────────────────────────────────────

app = Flask(
    __name__,
    template_folder=str(ROOT / "src" / "templates"),
    static_folder=str(ROOT / "src" / "static"),
)
app.secret_key = os.environ.get("FLASK_SECRET", "aura-dev-secret-key-change-me")

init_db()

# Clear all sessions on every fresh app start so users must log in
with app.app_context():
    app.secret_key = os.urandom(24).hex()

# ── Globals for the recording worker (single-user local app) ───────────────

_transcriber: FasterWhisperTranscriber | None = None
_rec_lock = threading.Lock()
_rec_state = {
    "active": False,
    "stop_flag": None,
    "segments": [],
    "elapsed": [0.0],
    "worker": None,
}


def _get_transcriber() -> FasterWhisperTranscriber:
    global _transcriber
    if _transcriber is None:
        _transcriber = FasterWhisperTranscriber(
            model_size="base.en", device="auto", compute_type="int8",
        )
    return _transcriber


# ── Auth helpers ────────────────────────────────────────────────────────────

def _current_user() -> dict | None:
    return session.get("user")


def _require_login(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not _current_user():
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ────────────────────────────────────────────────────────────────────────────
# AUTH ROUTES
# ────────────────────────────────────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    if _current_user():
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not username or not password:
            flash("Please fill in all fields.", "error")
        else:
            user = authenticate_user(username, password)
            if user:
                session["user"] = user
                return redirect(url_for("home"))
            else:
                flash("Invalid username or password.", "error")

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if _current_user():
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm", "")
        if not username or not password:
            flash("Please fill in all fields.", "error")
        elif password != confirm:
            flash("Passwords do not match.", "error")
        else:
            uid = create_user(username, password)
            if uid:
                # Auto-login after signup
                user = authenticate_user(username, password)
                session["user"] = user
                flash(f"Welcome to AURA, {username}!", "success")
                return redirect(url_for("home"))
            else:
                flash("Username already taken.", "error")

    return render_template("signup.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ────────────────────────────────────────────────────────────────────────────
# HOME (subject list)
# ────────────────────────────────────────────────────────────────────────────

@app.route("/")
@_require_login
def home():
    user = _current_user()
    subjects = get_subjects(user["id"])
    stats = get_usage_stats(user["id"])
    return render_template("home.html", user=user, subjects=subjects, stats=stats)


@app.route("/subject/add", methods=["POST"])
@_require_login
def add_subject():
    user = _current_user()
    name = request.form.get("name", "").strip()
    if name:
        result = create_subject(user["id"], name)
        if result:
            flash(f"Subject '{name}' created!", "success")
        else:
            flash("Subject already exists.", "error")
    return redirect(url_for("home"))


@app.route("/subject/<int:subject_id>/delete", methods=["POST"])
@_require_login
def remove_subject(subject_id):
    delete_subject(subject_id)
    flash("Subject deleted.", "success")
    return redirect(url_for("home"))


# ────────────────────────────────────────────────────────────────────────────
# SUBJECT DETAIL (lectures list)
# ────────────────────────────────────────────────────────────────────────────

@app.route("/subject/<int:subject_id>")
@_require_login
def subject_detail(subject_id):
    user = _current_user()
    subjects = get_subjects(user["id"])
    subject = next((s for s in subjects if s["id"] == subject_id), None)
    if not subject:
        flash("Subject not found.", "error")
        return redirect(url_for("home"))

    lectures = get_lectures(subject_id)
    stats = get_usage_stats(user["id"])
    papers = get_exam_papers(subject_id)
    return render_template(
        "subject.html",
        user=user, subject=subject, lectures=lectures,
        stats=stats, papers=papers,
    )


# ────────────────────────────────────────────────────────────────────────────
# RECORD LECTURE page
# ────────────────────────────────────────────────────────────────────────────

@app.route("/subject/<int:subject_id>/record")
@_require_login
def record_page(subject_id):
    user = _current_user()
    subjects = get_subjects(user["id"])
    subject = next((s for s in subjects if s["id"] == subject_id), None)
    if not subject:
        return redirect(url_for("home"))
    stats = get_usage_stats(user["id"])
    return render_template("record.html", user=user, subject=subject, stats=stats)


# ── Recording API (AJAX) ───────────────────────────────────────────────────

def _recording_worker(capture, transcriber, stop_flag, segments_out, elapsed_out):
    """Background thread: capture audio → buffer → transcribe on silence/max."""
    SR = AudioCapture.SAMPLE_RATE
    BLOCK = AudioCapture.BLOCK_SECONDS
    SEG_SEC = 10.0
    SILENCE_THRESH = 0.01
    SILENCE_DUR = 1.5

    buf, buf_dur, elapsed = [], 0.0, 0.0
    silent_chunks, max_silent = 0, int(SILENCE_DUR / BLOCK)

    capture.start()
    try:
        while not stop_flag.is_set():
            chunk = capture.read_chunk(timeout=0.5)
            if chunk is None:
                continue
            clen = len(chunk) / SR
            buf.append(chunk)
            buf_dur += clen
            elapsed += clen
            elapsed_out.clear()
            elapsed_out.append(elapsed)

            rms = float(np.sqrt(np.mean(chunk ** 2)))
            silent_chunks = silent_chunks + 1 if rms < SILENCE_THRESH else 0

            flush = (silent_chunks >= max_silent and buf_dur > 2.0) or buf_dur >= SEG_SEC
            if flush and buf:
                audio = np.concatenate(buf)
                bstart = elapsed - buf_dur
                segs = transcriber.transcribe_array(audio, SR)
                for s in segs:
                    s.start_time = round(bstart + s.start_time, 2)
                    s.end_time = round(bstart + s.end_time, 2)
                    if s.text.strip():
                        segments_out.append(s)
                buf.clear()
                buf_dur = 0.0
                silent_chunks = 0

        # flush remainder
        if buf:
            audio = np.concatenate(buf)
            bstart = elapsed - buf_dur
            segs = transcriber.transcribe_array(audio, SR)
            for s in segs:
                s.start_time = round(bstart + s.start_time, 2)
                s.end_time = round(bstart + s.end_time, 2)
                if s.text.strip():
                    segments_out.append(s)
    finally:
        capture.stop()


@app.route("/api/record/start", methods=["POST"])
@_require_login
def api_start_recording():
    with _rec_lock:
        if _rec_state["active"]:
            return jsonify(ok=False, error="Already recording")

        transcriber = _get_transcriber()
        capture = AudioCapture(device=None)
        stop_flag = threading.Event()
        segments_out = []
        elapsed_out = [0.0]

        worker = threading.Thread(
            target=_recording_worker,
            args=(capture, transcriber, stop_flag, segments_out, elapsed_out),
            daemon=True,
        )
        worker.start()

        _rec_state.update({
            "active": True,
            "stop_flag": stop_flag,
            "segments": segments_out,
            "elapsed": elapsed_out,
            "worker": worker,
            "capture": capture,
        })

    return jsonify(ok=True)


@app.route("/api/record/status")
@_require_login
def api_recording_status():
    with _rec_lock:
        segs = _rec_state["segments"]
        el = _rec_state["elapsed"][0] if _rec_state["elapsed"] else 0.0
        texts = [s.text for s in segs]
    return jsonify(
        active=_rec_state["active"],
        elapsed=round(el, 1),
        segment_count=len(texts),
        texts=texts,
    )


@app.route("/api/record/stop", methods=["POST"])
@_require_login
def api_stop_recording():
    with _rec_lock:
        if not _rec_state["active"]:
            return jsonify(ok=False, error="Not recording")

        _rec_state["stop_flag"].set()

    _rec_state["worker"].join(timeout=15)

    with _rec_lock:
        segments = list(_rec_state["segments"])
        elapsed = _rec_state["elapsed"][0] if _rec_state["elapsed"] else 0.0
        _rec_state["active"] = False

    if not segments:
        return jsonify(ok=False, error="No speech detected")

    # ── Save transcript ──────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "data" / "transcripts"
    out_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = out_dir / f"transcript_{ts}.json"
    with open(transcript_path, "w") as f:
        json.dump({
            "created": ts,
            "duration_seconds": round(elapsed, 2),
            "segments": [asdict(s) for s in segments],
        }, f, indent=2)

    return jsonify(
        ok=True,
        transcript_path=str(transcript_path),
        elapsed=round(elapsed, 2),
        segment_count=len(segments),
    )


@app.route("/api/summarize", methods=["POST"])
@_require_login
def api_summarize():
    user = _current_user()
    data = request.get_json()
    transcript_path = data.get("transcript_path")
    subject_id = data.get("subject_id")
    title = data.get("title", "Untitled Lecture")
    elapsed = data.get("elapsed", 0.0)

    if not transcript_path or not Path(transcript_path).exists():
        return jsonify(ok=False, error="Transcript not found")

    result = summarize_transcript(
        transcript_path=transcript_path,
        output_dir=str(ROOT / "data" / "outputs"),
    )

    # Log usage
    usage = result.get("usage", {})
    if usage:
        log_usage(
            user_id=user["id"],
            action="summarize",
            model=usage.get("model", "gpt-4o-mini"),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    # Save lecture record
    lecture_id = create_lecture(
        subject_id=subject_id,
        title=title,
        duration_seconds=elapsed,
        transcript_path=transcript_path,
        short_summary_path=str(result["short"]),
        long_summary_path=str(result["long"]),
    )

    short_md = Path(result["short"]).read_text(encoding="utf-8") if result.get("short") else ""
    long_md = Path(result["long"]).read_text(encoding="utf-8") if result.get("long") else ""

    return jsonify(
        ok=True,
        lecture_id=lecture_id,
        short_summary=short_md,
        long_summary=long_md,
        usage=usage,
    )


# ────────────────────────────────────────────────────────────────────────────
# VIEW LECTURE SUMMARY
# ────────────────────────────────────────────────────────────────────────────

@app.route("/lecture/<int:lecture_id>")
@_require_login
def view_lecture(lecture_id):
    user = _current_user()
    lec = get_lecture(lecture_id)
    if not lec:
        flash("Lecture not found.", "error")
        return redirect(url_for("home"))

    stats = get_usage_stats(user["id"])

    short_md = ""
    long_md = ""
    transcript_text = ""

    if lec.get("short_summary_path") and Path(lec["short_summary_path"]).exists():
        short_md = Path(lec["short_summary_path"]).read_text(encoding="utf-8")
    if lec.get("long_summary_path") and Path(lec["long_summary_path"]).exists():
        long_md = Path(lec["long_summary_path"]).read_text(encoding="utf-8")
    if lec.get("transcript_path") and Path(lec["transcript_path"]).exists():
        with open(lec["transcript_path"]) as f:
            tdata = json.load(f)
        transcript_text = " ".join(s["text"] for s in tdata.get("segments", []) if s.get("text"))

    return render_template(
        "lecture.html",
        user=user, lecture=lec, stats=stats,
        short_summary=short_md, long_summary=long_md,
        transcript_text=transcript_text,
    )


# ────────────────────────────────────────────────────────────────────────────
# EXAM PAPERS (frontend template — non-functional backend)
# ────────────────────────────────────────────────────────────────────────────

@app.route("/subject/<int:subject_id>/exams", methods=["GET", "POST"])
@_require_login
def exam_papers(subject_id):
    user = _current_user()
    subjects = get_subjects(user["id"])
    subject = next((s for s in subjects if s["id"] == subject_id), None)
    if not subject:
        return redirect(url_for("home"))

    if request.method == "POST":
        files = request.files.getlist("papers")
        saved = 0
        for f in files:
            if f.filename:
                dest = ROOT / "data" / "exam_papers" / f.filename
                dest.parent.mkdir(parents=True, exist_ok=True)
                f.save(str(dest))
                add_exam_paper(subject_id, f.filename, str(dest))
                saved += 1
        if saved:
            flash(f"{saved} paper(s) uploaded. Parsing coming soon!", "success")

    papers = get_exam_papers(subject_id)
    stats = get_usage_stats(user["id"])
    return render_template(
        "exams.html",
        user=user, subject=subject, papers=papers, stats=stats,
    )


# ────────────────────────────────────────────────────────────────────────────
# FILE DOWNLOADS
# ────────────────────────────────────────────────────────────────────────────

@app.route("/download")
@_require_login
def download_file():
    fpath = request.args.get("path", "")
    p = Path(fpath)
    if p.exists():
        return send_file(str(p), as_attachment=True)
    flash("File not found.", "error")
    return redirect(url_for("home"))


# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  🎓 AURA — http://localhost:5000\n")
    app.run(debug=True, port=5000, use_reloader=False)
