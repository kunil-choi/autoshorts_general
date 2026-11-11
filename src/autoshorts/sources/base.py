"""The MediaSource interface both input modes implement.

Everything downstream (transcript, analyze, render) works against this
interface instead of caring whether the video came from a YouTube link or
an uploaded file - see YouTubeSource / UploadedFileSource for the two
implementations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class MediaSource(Protocol):
    id: str
    title: str

    def duration_sec(self) -> float:
        """Total length of the source video, in seconds."""
        ...

    def captions_vtt(self, work_dir: Path, lang: str = "ko") -> Path | None:
        """Existing (manual/auto) captions as a vtt file under work_dir, or
        None if there are none to fetch - e.g. an uploaded file always
        returns None here, since only a published YouTube video can have
        platform captions."""
        ...

    def extract_audio(self, out_dir: Path) -> Path:
        """Best-available audio for the whole source, for whisper fallback."""
        ...

    def extract_clip(self, start_sec: float, end_sec: float, out_path: Path) -> Path:
        """A single [start_sec, end_sec] segment as a standalone video file,
        at the source's native quality - the raw material clipbuild.py
        hard-cuts together before render.py composites it."""
        ...
