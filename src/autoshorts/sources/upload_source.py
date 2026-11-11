"""MediaSource backed by a video file the worker already has on their own
machine - no download step, so no YouTube bot-block exposure, and no
generation loss from YouTube's own upload transcoding (see the project
README's quality-vs-URL-mode note).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .base import MediaSource


class UploadedFileSource(MediaSource):
    def __init__(self, file_path: Path, title: str | None = None):
        self.path = Path(file_path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.id = self.path.stem
        self.title = title or self.path.stem

    def duration_sec(self) -> float:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "format=duration", "-of", "csv=p=0",
                str(self.path),
            ],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        return float(out)

    def captions_vtt(self, work_dir: Path, lang: str = "ko") -> Path | None:
        # an uploaded file was never published, so there is no platform
        # captions track to fetch - transcript.py always falls back to
        # whisper on extract_audio() for this source.
        return None

    def extract_audio(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{self.id}.mp3"
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(self.path),
                "-vn", "-c:a", "libmp3lame", "-q:a", "4",
                str(out_path),
            ],
            check=True, capture_output=True,
        )
        return out_path

    def extract_clip(self, start_sec: float, end_sec: float, out_path: Path) -> Path:
        """Trim exactly [start_sec, end_sec] from the local file - the
        output starts/ends precisely there, with no extra footage, so
        clipbuild.py can concatenate several of these with a plain hard
        cut. Re-encoding (rather than "-c copy") from an input seek is what
        makes ffmpeg land on the exact requested timestamp instead of the
        nearest keyframe."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(start_sec), "-to", str(end_sec), "-i", str(self.path),
                "-c:v", "libx264", "-preset", "medium", "-crf", "15",
                "-c:a", "aac", "-b:a", "256k",
                str(out_path),
            ],
            check=True, capture_output=True,
        )
        return out_path
