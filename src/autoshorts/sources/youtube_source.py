"""MediaSource backed by a YouTube URL/video id, via yt-dlp.

Runs locally (not on a shared server) so YouTube sees the worker's own
network IP instead of a cloud IP - see the parent project's README for why
that matters. download_audio / download_auto_captions / download_segment wrapped behind MediaSource.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yt_dlp

from .base import MediaSource

_VIDEO_ID_RE = re.compile(r"^[\w-]{11}$")
_URL_VIDEO_ID_PATTERNS = (
    re.compile(r"(?:youtube\.com/watch\?(?:.*&)?v=|youtube\.com/shorts/|youtube\.com/embed/|youtu\.be/)([\w-]{11})"),
)


def _extract_video_id(url_or_id: str) -> str:
    """Pull an 11-char video id out of a YouTube URL, or pass through a bare id."""
    candidate = url_or_id.strip()
    if _VIDEO_ID_RE.match(candidate):
        return candidate
    for pattern in _URL_VIDEO_ID_PATTERNS:
        m = pattern.search(candidate)
        if m:
            return m.group(1)
    raise ValueError(f"couldn't find a YouTube video id in {url_or_id!r}")


def _base_ydl_opts(extra: dict | None = None) -> dict:
    """Common yt-dlp options.

    YouTube aggressively rate-limits/blocks requests from cloud IPs (the
    "Sign in to confirm you're not a bot" error is common on cloud
    runners). Mitigation depends on whether a cookies file
    (YTDLP_COOKIES_FILE, exported from a real, logged-in browser session)
    is available:

    - with cookies: use the web client (first in the list = preferred). The
      android/tv clients don't accept web-session cookies properly and were
      observed returning "Requested format is not available" for every
      video once cookies were added, even though the sign-in check itself
      passed.
    - without cookies: prefer the android/tv player clients, since they
      don't need the same web sign-in check as the default web client.
    """
    cookies_file = os.environ.get("YTDLP_COOKIES_FILE")
    player_client = ["web", "android", "tv"] if cookies_file else ["android", "tv", "web"]
    opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {"youtube": {"player_client": player_client}},
    }
    if cookies_file:
        opts["cookiefile"] = cookies_file
    if extra:
        opts.update(extra)
    return opts


class YouTubeSource(MediaSource):
    def __init__(self, url_or_id: str):
        self.id = _extract_video_id(url_or_id)
        self.url = f"https://www.youtube.com/watch?v={self.id}"
        self._info: dict | None = None

    def _fetch_info(self) -> dict:
        """Fetch title/duration/description only, without resolving/selecting
        formats.

        process=False skips yt-dlp's format-selection step entirely. That
        step is what breaks (with "Requested format is not available") once
        YouTube requires a PO token for the formats list of a
        cookie-authenticated session; skipping it is safe here since we
        only need metadata fields, which the extractor already fills in
        before format selection runs.
        """
        if self._info is None:
            ydl_opts = _base_ydl_opts({"skip_download": True})
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                self._info = ydl.extract_info(self.url, download=False, process=False)
        return self._info

    @property
    def title(self) -> str:
        return self._fetch_info().get("title", "")

    @property
    def description(self) -> str | None:
        return self._fetch_info().get("description")

    @property
    def upload_date(self) -> str | None:
        """YYYYMMDD the source video was published (used as its 촬영/방송
        date). Falls back through the other date fields yt-dlp may fill
        instead (release_date, then a unix timestamp, in KST) since which
        ones are present varies with the player client in use."""
        info = self._fetch_info()
        for key in ("upload_date", "release_date"):
            if info.get(key):
                return str(info[key])
        for key in ("release_timestamp", "timestamp"):
            if info.get(key):
                from datetime import datetime
                from zoneinfo import ZoneInfo
                return datetime.fromtimestamp(float(info[key]), ZoneInfo("Asia/Seoul")).strftime("%Y%m%d")
        return None

    def duration_sec(self) -> float:
        duration = self._fetch_info().get("duration")
        if duration is None:
            raise ValueError(f"no duration available for video {self.id}")
        return float(duration)

    def captions_vtt(self, work_dir: Path, lang: str = "ko") -> Path | None:
        """Try to fetch existing (manual or auto-generated) captions as vtt,
        written under work_dir (the caller's job-specific directory - see
        MediaSource.captions_vtt).

        Returns the path to the .vtt file, or None if no captions were available.
        """
        work_dir.mkdir(parents=True, exist_ok=True)
        out_tmpl = str(work_dir / self.id)
        ydl_opts = _base_ydl_opts(
            {
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": [lang],
                "subtitlesformat": "vtt",
                "outtmpl": out_tmpl,
                # skip_download=True still runs format selection (for filename
                # templating etc.), which fails with "Requested format is not
                # available" under the same PO-token gating as _fetch_info. We
                # don't need any video/audio format here, only subtitles, so
                # ignore that failure instead of aborting the whole call.
                "ignore_no_formats_error": True,
            }
        )
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([self.url])

        candidates = list(work_dir.glob(f"{self.id}*.{lang}.vtt"))
        return candidates[0] if candidates else None

    def extract_audio(self, out_dir: Path) -> Path:
        """Download best-audio only, for transcript generation. Returns the mp3 path."""
        out_dir.mkdir(parents=True, exist_ok=True)
        out_tmpl = str(out_dir / f"{self.id}.%(ext)s")
        ydl_opts = _base_ydl_opts(
            {
                "format": "bestaudio/best",
                "outtmpl": out_tmpl,
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "128",
                    }
                ],
            }
        )
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([self.url])
        return out_dir / f"{self.id}.mp3"

    def extract_clip(self, start_sec: float, end_sec: float, out_path: Path) -> Path:
        """Download exactly the [start_sec, end_sec] section of the video
        (video+audio muxed) - the returned file's content starts and ends
        precisely there, with no extra footage, so clipbuild.py can
        concatenate several of these outputs with a plain hard cut."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ydl_opts = _base_ydl_opts(
            {
                # no ext filter: YouTube serves anything above 1080p (1440p/4K)
                # as VP9/AV1 in a webm container only, never mp4, so restricting
                # to mp4 here would silently cap the source at a lower
                # resolution than what's actually available. merge_output_format
                # remuxes whatever codec comes back into an .mp4 container, and
                # render.py re-encodes to libx264 anyway, so the source
                # container/codec doesn't need to be mp4/h264 up front.
                "format": "bestvideo+bestaudio/best",
                "outtmpl": str(out_path.with_suffix("")) + ".%(ext)s",
                "download_ranges": yt_dlp.utils.download_range_func(None, [(start_sec, end_sec)]),
                "force_keyframes_at_cuts": True,
                # force_keyframes_at_cuts makes yt-dlp re-encode at the cut
                # points (via ffmpeg -ss/-i input seeking) so the output
                # starts/ends exactly on [start_sec, end_sec] instead of
                # snapping to the nearest source keyframe. Left unconfigured,
                # that re-encode falls back to ffmpeg's plain defaults
                # (libx264 crf 23 medium, ~128k aac), which bakes in lossy
                # artifacts before render.py even gets to encode with its own
                # final pass. Pin it to a near-lossless intermediate so the
                # only meaningful quality loss happens once, at the final encode.
                "external_downloader_args": {
                    "ffmpeg_o": [
                        "-c:v", "libx264", "-preset", "medium", "-crf", "15",
                        "-c:a", "aac", "-b:a", "256k",
                    ],
                },
                "merge_output_format": "mp4",
                # each clip's path is keyed by its own start/end, but be
                # defensive: never silently reuse a stale file left over from
                # a previous (e.g. interrupted) run at the same path
                "overwrites": True,
            }
        )
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([self.url])
        produced = out_path.with_suffix(".mp4")
        if produced != out_path and produced.exists():
            produced.rename(out_path)
        return out_path
