"""
Transcriber interface — batch (file upload) and live (mic / system audio) ASR.

Supported backends
──────────────────
1. faster-whisper  (default)  — CTranslate2-optimised Whisper; runs well on
   CPU / Apple Silicon.  Best choice for a macOS-first POC.
2. NVIDIA Parakeet via NeMo   — state-of-the-art accuracy; requires a CUDA
   GPU for practical speed.  Swap in when running on a GPU workstation.

Usage
─────
  # List available microphones / loopback devices
  python -m src.transcription --list-devices

  # Live transcription from default mic
  python -m src.transcription

  # Live transcription from a specific device (e.g. BlackHole loopback)
  python -m src.transcription --device 2

  # Batch-transcribe an uploaded recording
  python -m src.transcription --file data/audio/lecture.m4a

  # Use Parakeet instead of faster-whisper (needs CUDA)
  python -m src.transcription --backend parakeet
"""

from __future__ import annotations

import os
import sys
import json
import queue
import argparse
import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd


# ────────────────────────────────────────────────────────────────────────────
# Data types
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class TranscriptSegment:
    """A single timestamped piece of recognised speech."""
    start_time: float          # seconds from start of recording
    end_time: float
    text: str
    confidence: Optional[float] = None


# ────────────────────────────────────────────────────────────────────────────
# Base transcriber
# ────────────────────────────────────────────────────────────────────────────

class BaseTranscriber(ABC):
    """Common interface every ASR backend must implement."""

    @abstractmethod
    def transcribe_array(
        self, audio: np.ndarray, sample_rate: int
    ) -> list[TranscriptSegment]:
        """Transcribe a numpy float32 mono audio array → timestamped segments."""
        ...

    def transcribe_file(self, path: str | Path) -> list[TranscriptSegment]:
        """Transcribe an audio file.  Default: load → array → transcribe_array."""
        import soundfile as sf

        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # stereo → mono
        return self.transcribe_array(audio, sr)


# ────────────────────────────────────────────────────────────────────────────
# Backend 1 — faster-whisper  (recommended for macOS / CPU)
# ────────────────────────────────────────────────────────────────────────────

class FasterWhisperTranscriber(BaseTranscriber):
    """
    CTranslate2-accelerated Whisper.

    Install
    -------
    pip install faster-whisper

    Model sizes (English-only, fastest → most accurate):
      tiny.en → base.en → small.en → medium.en → large-v3
    """

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "auto",
        compute_type: str = "int8",
    ):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )
        print(f"[ASR] faster-whisper loaded  model={model_size}  device={device}")

    def transcribe_array(
        self, audio: np.ndarray, sample_rate: int
    ) -> list[TranscriptSegment]:
        # faster-whisper expects 16 kHz mono float32
        if sample_rate != 16_000:
            import resampy
            audio = resampy.resample(audio, sample_rate, 16_000)

        segments_iter, _info = self.model.transcribe(
            audio,
            language="en",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        results: list[TranscriptSegment] = []
        for seg in segments_iter:
            results.append(
                TranscriptSegment(
                    start_time=round(seg.start, 2),
                    end_time=round(seg.end, 2),
                    text=seg.text.strip(),
                    confidence=(
                        round(seg.avg_logprob, 4) if seg.avg_logprob else None
                    ),
                )
            )
        return results


# ────────────────────────────────────────────────────────────────────────────
# Backend 2 — NVIDIA Parakeet via NeMo  (requires CUDA)
# ────────────────────────────────────────────────────────────────────────────

class ParakeetTranscriber(BaseTranscriber):
    """
    NVIDIA Parakeet — state-of-the-art open-source ASR models.

    Install
    -------
    pip install nemo_toolkit[asr]

    Available models (HuggingFace / NVIDIA NGC):
      nvidia/parakeet-ctc-1.1b    — CTC decoder, fast batch inference
      nvidia/parakeet-rnnt-1.1b   — RNN-T decoder, good for streaming
      nvidia/parakeet-tdt-1.1b    — TDT decoder, best overall accuracy

    Note: these models are trained on English and run best on CUDA GPUs.
    CPU inference works but is very slow for hour-long audio.
    """

    def __init__(self, model_name: str = "nvidia/parakeet-tdt-1.1b"):
        import torch
        import nemo.collections.asr as nemo_asr

        if not torch.cuda.is_available():
            print(
                "[WARN] No CUDA GPU detected. Parakeet will run on CPU "
                "(very slow for long audio). Consider using --backend faster-whisper."
            )

        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name)
        self.model.eval()
        print(f"[ASR] Parakeet loaded  model={model_name}")

    # NeMo models accept file paths natively, but we also support arrays.
    def transcribe_array(
        self, audio: np.ndarray, sample_rate: int
    ) -> list[TranscriptSegment]:
        import tempfile
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, audio, sample_rate)

        try:
            return self._transcribe_paths([tmp_path], len(audio) / sample_rate)
        finally:
            os.unlink(tmp_path)

    def transcribe_file(self, path: str | Path) -> list[TranscriptSegment]:
        import soundfile as sf

        info = sf.info(str(path))
        duration = info.duration
        return self._transcribe_paths([str(path)], duration)

    # -- internal helper --------------------------------------------------

    def _transcribe_paths(
        self, paths: list[str], duration: float
    ) -> list[TranscriptSegment]:
        """Run NeMo transcribe and normalise the output format."""
        output = self.model.transcribe(paths, timestamps=True)

        segments: list[TranscriptSegment] = []

        # NeMo returns different structures depending on model variant.
        # We handle the common cases gracefully.
        if isinstance(output, (list, tuple)) and len(output) > 0:
            hyp = output[0]

            # Case 1: hypothesis object with word/segment timestamps
            if hasattr(hyp, "timestep") and hyp.timestep:
                for ts in hyp.timestep:
                    segments.append(
                        TranscriptSegment(
                            start_time=round(getattr(ts, "start", 0.0), 2),
                            end_time=round(getattr(ts, "end", 0.0), 2),
                            text=getattr(ts, "text", str(ts)).strip(),
                        )
                    )
            # Case 2: plain string
            else:
                text = hyp if isinstance(hyp, str) else str(hyp)
                segments.append(
                    TranscriptSegment(
                        start_time=0.0,
                        end_time=round(duration, 2),
                        text=text.strip(),
                    )
                )
        return segments


# ────────────────────────────────────────────────────────────────────────────
# Audio capture  (mic or loopback device)
# ────────────────────────────────────────────────────────────────────────────

class AudioCapture:
    """
    Opens an input stream via sounddevice and queues audio blocks.

    For system-audio capture on macOS, set up a loopback driver
    (e.g. BlackHole) and pass its device index here.
    """

    SAMPLE_RATE = 16_000      # 16 kHz mono — standard for ASR
    CHANNELS = 1
    BLOCK_SECONDS = 0.5       # size of each callback block

    def __init__(self, device: int | str | None = None):
        self.device = device
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None

    # -- sounddevice callback ---------------------------------------------

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[MIC] {status}", file=sys.stderr)
        self._queue.put(indata[:, 0].copy())   # keep only mono channel

    # -- public -----------------------------------------------------------

    def start(self):
        blocksize = int(self.SAMPLE_RATE * self.BLOCK_SECONDS)
        self._stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            dtype="float32",
            blocksize=blocksize,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        print(
            f"[MIC] Recording — device={self.device or 'default'}, "
            f"rate={self.SAMPLE_RATE} Hz"
        )

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read_chunk(self, timeout: float = 1.0) -> np.ndarray | None:
        """Return the next audio block from the queue (blocking)."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None


# ────────────────────────────────────────────────────────────────────────────
# Live transcription session
# ────────────────────────────────────────────────────────────────────────────

class LiveTranscriptionSession:
    """
    Orchestrates:  mic capture → buffer → silence detection → ASR → output.

    Strategy (simple & reliable for a POC):
      1. Accumulate audio in a rolling buffer.
      2. Flush buffer to ASR when a silence gap is detected OR the buffer
         reaches ``segment_seconds``.
      3. Print the transcript live and persist to disk when stopped.
    """

    def __init__(
        self,
        transcriber: BaseTranscriber,
        output_dir: str | Path = "data/transcripts",
        segment_seconds: float = 10.0,
        silence_threshold: float = 0.01,
        silence_duration: float = 1.5,
        device: int | str | None = None,
    ):
        self.transcriber = transcriber
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.segment_seconds = segment_seconds
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration

        self.capture = AudioCapture(device=device)

        self._all_segments: list[TranscriptSegment] = []
        self._buffer: list[np.ndarray] = []
        self._buffer_duration: float = 0.0
        self._elapsed: float = 0.0
        self._running = False

    # -- silence detection (simple RMS energy) ----------------------------

    @staticmethod
    def _rms(audio: np.ndarray) -> float:
        return float(np.sqrt(np.mean(audio ** 2)))

    def _is_silent(self, chunk: np.ndarray) -> bool:
        return self._rms(chunk) < self.silence_threshold

    # -- flush & transcribe -----------------------------------------------

    def _flush_buffer(self):
        """Transcribe whatever is in the buffer, then clear it."""
        if not self._buffer:
            return

        audio = np.concatenate(self._buffer)
        buf_start = self._elapsed - self._buffer_duration

        segments = self.transcriber.transcribe_array(
            audio, AudioCapture.SAMPLE_RATE
        )

        # Offset segment timestamps to absolute lecture time
        for seg in segments:
            seg.start_time = round(buf_start + seg.start_time, 2)
            seg.end_time = round(buf_start + seg.end_time, 2)
            if seg.text:
                self._all_segments.append(seg)
                self._print_segment(seg)

        self._buffer.clear()
        self._buffer_duration = 0.0

    # -- display helpers --------------------------------------------------

    def _print_segment(self, seg: TranscriptSegment):
        ts = f"[{self._fmt(seg.start_time)} → {self._fmt(seg.end_time)}]"
        print(f"  {ts}  {seg.text}")

    @staticmethod
    def _fmt(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    # -- main loop --------------------------------------------------------

    def run(self):
        """Start live capture + transcription.  Press Ctrl-C to stop."""
        self._running = True
        self.capture.start()

        silent_chunks = 0
        max_silent = int(self.silence_duration / AudioCapture.BLOCK_SECONDS)

        print("\n╔══════════════════════════════════════════════╗")
        print("║   AURA — Live Transcription Started          ║")
        print("║   Press Ctrl-C to stop                       ║")
        print("╚══════════════════════════════════════════════╝\n")

        try:
            while self._running:
                chunk = self.capture.read_chunk(timeout=1.0)
                if chunk is None:
                    continue

                chunk_dur = len(chunk) / AudioCapture.SAMPLE_RATE
                self._buffer.append(chunk)
                self._buffer_duration += chunk_dur
                self._elapsed += chunk_dur

                # Track consecutive silent blocks
                if self._is_silent(chunk):
                    silent_chunks += 1
                else:
                    silent_chunks = 0

                # Decide when to flush
                should_flush = (
                    (silent_chunks >= max_silent and self._buffer_duration > 2.0)
                    or self._buffer_duration >= self.segment_seconds
                )
                if should_flush:
                    self._flush_buffer()
                    silent_chunks = 0

        except KeyboardInterrupt:
            print("\n[STOP] Finishing up…")

        finally:
            self._flush_buffer()       # transcribe any remaining audio
            self.capture.stop()
            self._running = False

        return self._save()

    # -- persistence ------------------------------------------------------

    def _save(self) -> Path:
        """Write transcript to JSON (machine-readable) + TXT (human-readable)."""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = self.output_dir / f"transcript_{ts}"

        # ── JSON ──
        json_path = base.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(
                {
                    "created": ts,
                    "duration_seconds": round(self._elapsed, 2),
                    "segments": [asdict(s) for s in self._all_segments],
                },
                f,
                indent=2,
            )

        # ── Plain text ──
        txt_path = base.with_suffix(".txt")
        with open(txt_path, "w") as f:
            f.write(f"AURA Transcript — {ts}\n")
            f.write(f"Duration: {self._fmt(self._elapsed)}\n")
            f.write("=" * 60 + "\n\n")
            for seg in self._all_segments:
                f.write(
                    f"[{self._fmt(seg.start_time)} → {self._fmt(seg.end_time)}]\n"
                )
                f.write(f"{seg.text}\n\n")

        print(f"\n✓ Saved:  {json_path}")
        print(f"✓ Saved:  {txt_path}")
        print(f"  Total segments : {len(self._all_segments)}")
        print(f"  Total duration : {self._fmt(self._elapsed)}")
        return json_path


# ────────────────────────────────────────────────────────────────────────────
# Factory
# ────────────────────────────────────────────────────────────────────────────

BACKENDS = {
    "faster-whisper": FasterWhisperTranscriber,
    "parakeet": ParakeetTranscriber,
}


def get_transcriber(backend: str = "faster-whisper", **kwargs) -> BaseTranscriber:
    """Instantiate a transcriber by name."""
    cls = BACKENDS.get(backend)
    if cls is None:
        raise ValueError(
            f"Unknown backend '{backend}'. Choose from: {list(BACKENDS)}"
        )
    return cls(**kwargs)


# ────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AURA — Lecture Transcription (live mic or batch file)"
    )
    parser.add_argument(
        "--backend",
        default="faster-whisper",
        choices=list(BACKENDS),
        help="ASR backend (default: faster-whisper)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name/size.  "
            "faster-whisper: tiny.en | base.en | small.en | medium.en | large-v3.  "
            "parakeet: nvidia/parakeet-ctc-1.1b | parakeet-rnnt-1.1b | parakeet-tdt-1.1b"
        ),
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Path to audio file for batch transcription (omit for live mic)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Audio input device index or name (live mode).  Use --list-devices to see options.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/transcripts",
        help="Directory to save transcripts (default: data/transcripts)",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=10.0,
        help="Max seconds per transcription chunk in live mode (default: 10)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit",
    )
    args = parser.parse_args()

    # ── List devices ────────────────────────────────────────────────────
    if args.list_devices:
        print("\nAvailable audio input devices:\n")
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                tag = " ← default" if i == sd.default.device[0] else ""
                print(
                    f"  [{i}] {d['name']}  "
                    f"(ch={d['max_input_channels']}, "
                    f"rate={int(d['default_samplerate'])}){tag}"
                )
        print()
        return

    # ── Build transcriber ───────────────────────────────────────────────
    model_kwargs: dict = {}
    if args.model:
        key = "model_size" if args.backend == "faster-whisper" else "model_name"
        model_kwargs[key] = args.model
    transcriber = get_transcriber(args.backend, **model_kwargs)

    # ── Batch mode ──────────────────────────────────────────────────────
    if args.file:
        print(f"\n[BATCH] Transcribing: {args.file}\n")
        segments = transcriber.transcribe_file(args.file)

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = out_dir / f"transcript_{ts}"

        json_path = base.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(
                {"segments": [asdict(s) for s in segments]}, f, indent=2
            )

        txt_path = base.with_suffix(".txt")
        with open(txt_path, "w") as f:
            for seg in segments:
                f.write(
                    f"[{seg.start_time:.1f}s → {seg.end_time:.1f}s] {seg.text}\n"
                )

        print(f"✓ {len(segments)} segments → {json_path}")
        return

    # ── Live mode ───────────────────────────────────────────────────────
    device = (
        int(args.device) if args.device and args.device.isdigit() else args.device
    )
    session = LiveTranscriptionSession(
        transcriber=transcriber,
        output_dir=args.output_dir,
        segment_seconds=args.segment_seconds,
        device=device,
    )
    session.run()


if __name__ == "__main__":
    main()
