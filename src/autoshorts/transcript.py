"""Timestamped transcript segments, from either a source's own captions or
local whisper transcription - source-agnostic (the YouTube-specific download orchestration lives
behind MediaSource)."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

import webvtt


@dataclass
class Segment:
    start_sec: float
    end_sec: float
    text: str
    # best-effort role guess ("앵커"/"패널") filled in later by
    # analyze.label_speakers() - empty string until then, since there's no
    # real speaker-diarization signal in a plain transcript to set this at
    # parse time.
    speaker: str = ""


def _vtt_timestamp_to_sec(ts: str) -> float:
    parts = [float(p) for p in ts.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts
    return h * 3600 + m * 60 + s


def _clean_caption_text(raw: str) -> str:
    text = re.sub(r"<[^>]+>", "", raw)
    return html.unescape(text).strip()


def parse_vtt(vtt_path: Path) -> list[Segment]:
    """Parse a vtt file into segments, collapsing YouTube's rolling-caption
    duplication (auto captions repeat trailing words across consecutive cues)."""
    segments: list[Segment] = []
    last_text = ""
    for caption in webvtt.read(str(vtt_path)):
        text = _clean_caption_text(caption.text)
        text = " ".join(line.strip() for line in text.splitlines() if line.strip())
        if not text or text == last_text:
            continue
        if last_text and text.startswith(last_text):
            text = text[len(last_text):].strip()
        if not text:
            continue
        start = _vtt_timestamp_to_sec(caption.start)
        end = _vtt_timestamp_to_sec(caption.end)
        segments.append(Segment(start_sec=start, end_sec=end, text=text))
        last_text = _clean_caption_text(caption.text).splitlines()[-1].strip()
    return segments


def whisper_transcribe(audio_path: Path, lang: str = "ko", model_size: str = "small") -> list[Segment]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    raw_segments, _info = model.transcribe(str(audio_path), language=lang, vad_filter=True)
    return [
        Segment(start_sec=s.start, end_sec=s.end, text=s.text.strip())
        for s in raw_segments
        if s.text.strip()
    ]


def get_transcript_for_source(source, work_dir: Path, lang: str = "ko") -> tuple[list[Segment], str]:
    """Return (segments, source_label) where source_label is 'captions' or
    'whisper' - tries the MediaSource's own captions first, falls back to
    whisper on its extracted audio."""
    vtt_path = source.captions_vtt(work_dir, lang=lang)
    if vtt_path is not None:
        segments = parse_vtt(vtt_path)
        if segments:
            return segments, "captions"

    audio_path = source.extract_audio(work_dir)
    segments = whisper_transcribe(audio_path, lang=lang)
    return segments, "whisper"


def segments_in_range(segments: list[Segment], start_sec: float, end_sec: float) -> list[Segment]:
    return [
        s for s in segments
        if s.end_sec > start_sec and s.start_sec < end_sec and s.text.strip()
    ]


def excerpt_for_range(segments: list[Segment], start_sec: float, end_sec: float) -> str:
    return " ".join(s.text.strip() for s in segments_in_range(segments, start_sec, end_sec))


def format_for_prompt(segments: list[Segment]) -> str:
    """Render segments as "[mm:ss-mm:ss] text" lines for the analysis prompt.
    Segments whose speaker role is known ("앵커"/"패널", see
    analyze.label_speakers) get it as a tag: "[mm:ss-mm:ss] [패널] text"."""

    def hms(sec: float) -> str:
        m, s = divmod(int(sec), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    return "\n".join(
        f"[{hms(s.start_sec)}-{hms(s.end_sec)}] " + (f"[{s.speaker}] " if s.speaker else "") + s.text
        for s in segments
    )
