"""Pick clean cut points for a shorts clip.

Two cuts on a clip's edges sound wrong when they land inside a word or in
the middle of a sentence: transcript (caption) boundaries are only
approximate, and the first/last line of a candidate is often a fragment.
The fix here works on the actual audio: look for the natural pauses
(silences) around the intended start/end and cut inside them, so a clip
begins right before the speaker starts talking and ends right after they
finish, with a short lead-in/tail instead of a hard chop.

Pure helpers (parse_silencedetect, choose_start, choose_end) hold the
decision logic so it can be unit-tested without ffmpeg.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

# ffmpeg silencedetect: anything quieter than this for at least this long
# counts as a pause between words/sentences.
SILENCE_NOISE_DB = -32
SILENCE_MIN_SEC = 0.2

# extra footage fetched around the intended range so there is room to move
# each cut into the nearest pause.
PAD_BEFORE_SEC = 2.0
PAD_AFTER_SEC = 3.5

# how far from the intended start/end a pause may be and still be used
START_SEARCH_BACK = 1.5
START_SEARCH_FWD = 1.0
END_SEARCH_BACK = 1.0
END_SEARCH_FWD = 3.0

# kept inside the pause: audio before the first word / after the last one
START_LEAD_SEC = 0.15
END_TAIL_SEC = 0.35

MIN_CLIP_SEC = 3.0

_SIL_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SIL_END = re.compile(r"silence_end:\s*(-?[\d.]+)")


def parse_silencedetect(stderr: str, total_sec: float) -> list[tuple[float, float]]:
    """(start, end) pairs from ffmpeg's silencedetect log. A silence that
    runs to the end of the file has no silence_end line - it ends at
    total_sec."""
    silences: list[tuple[float, float]] = []
    open_start: float | None = None
    for line in stderr.splitlines():
        m = _SIL_START.search(line)
        if m:
            open_start = max(0.0, float(m.group(1)))
            continue
        m = _SIL_END.search(line)
        if m and open_start is not None:
            silences.append((open_start, float(m.group(1))))
            open_start = None
    if open_start is not None:
        silences.append((open_start, total_sec))
    return silences


def choose_start(silences: list[tuple[float, float]], start: float) -> float:
    """Move `start` to just before speech resumes after the nearest pause.
    Pauses before `start` are preferred over ones after it (cutting later
    would drop the first words of the intended line). Returns `start`
    unchanged when no pause is close enough."""
    best: tuple[float, float] | None = None
    best_score = 0.0
    for s, e in silences:
        if e < start - START_SEARCH_BACK or s > start + START_SEARCH_FWD:
            continue
        if s <= start <= e:
            dist = 0.0
        elif e < start:
            dist = start - e
        else:
            dist = (s - start) * 2  # forward is the worse direction
        if best is None or dist < best_score:
            best, best_score = (s, e), dist
    if best is None:
        return start
    s, e = best
    return max(s, e - START_LEAD_SEC)


def choose_end(silences: list[tuple[float, float]], end: float) -> float:
    """Move `end` to just after the speaker finishes: the first pause that
    starts at/after the intended end (within END_SEARCH_FWD), else the
    latest pause shortly before it. Returns `end` unchanged when none."""
    after = [(s, e) for s, e in silences if end - 0.3 <= s <= end + END_SEARCH_FWD]
    if after:
        s, e = min(after, key=lambda p: p[0])
        return min(e, s + END_TAIL_SEC)
    before = [(s, e) for s, e in silences if end - END_SEARCH_BACK <= s < end - 0.3]
    if before:
        s, e = max(before, key=lambda p: p[0])
        return min(e, s + END_TAIL_SEC)
    return end


def detect_silences(media_path: Path, total_sec: float) -> list[tuple[float, float]]:
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostats", "-i", str(media_path), "-vn",
            "-af", f"silencedetect=noise={SILENCE_NOISE_DB}dB:d={SILENCE_MIN_SEC}",
            "-f", "null", "-",
        ],
        capture_output=True, text=True,
    )
    return parse_silencedetect(proc.stderr, total_sec)


def probe_duration(media_path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(media_path),
        ],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return float(out)


def trim(media_path: Path, start: float, end: float, out_path: Path) -> Path:
    """Accurate re-encoded trim (same near-lossless settings the clips are
    extracted with, so the final render is still the only lossy pass)."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(media_path), "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "15",
            "-c:a", "aac", "-b:a", "256k", str(out_path),
        ],
        check=True, capture_output=True,
    )
    return out_path
