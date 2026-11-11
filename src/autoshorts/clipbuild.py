"""Turn one or more transcript ranges the worker picked (step 6) into a
single hard-cut source clip, ready for render.py.

Only straight cuts are supported (no crossfades/transitions) - each range
is extracted from the MediaSource at its native quality via
MediaSource.extract_clip() (which returns exactly that range, no padding),
then concatenated in the order the worker selected them.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from . import boundaries
from .analyze import ClipRange
from .sources.base import MediaSource


def build_clip(
    source: MediaSource, ranges: list[ClipRange], out_path: Path, work_dir: Path | None = None
) -> Path:
    if not ranges:
        raise ValueError("at least one range is required")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = work_dir or out_path.parent / f"_{out_path.stem}_parts"
    work_dir.mkdir(parents=True, exist_ok=True)

    part_paths = [
        source.extract_clip(r.start_sec, r.end_sec, work_dir / f"part{i}.mp4")
        for i, r in enumerate(ranges)
    ]

    if len(part_paths) == 1:
        shutil.move(str(part_paths[0]), str(out_path))
        return out_path

    concat_list = work_dir / "concat.txt"
    concat_list.write_text(
        "\n".join(f"file '{p.resolve().as_posix()}'" for p in part_paths), encoding="utf-8"
    )
    # every part came out of the same extract_clip() encode settings
    # (libx264 crf 15 / aac 256k), so a stream-copy concat is safe and
    # avoids yet another re-encode generation before render.py's own pass.
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(out_path),
        ],
        check=True, capture_output=True,
    )
    return out_path


def _snap_range_to_pauses(
    source: MediaSource, r: ClipRange, part_path: Path,
    bound: tuple[float | None, float | None] = (None, None),
) -> tuple[Path, ClipRange]:
    """Extract r with a little padding, then cut it at the nearest natural
    pauses (see boundaries.py) so it neither starts nor ends mid-word.
    `bound` = (lo, hi) absolute times the cut must stay inside - used for the
    edges next to a deleted line, so snapping can't pull deleted speech back
    in. Falls back to the plain, exact range if anything about that fails."""
    padded_start = max(0.0, r.start_sec - boundaries.PAD_BEFORE_SEC)
    padded_end = r.end_sec + boundaries.PAD_AFTER_SEC
    padded_path = part_path.with_name(part_path.stem + "_padded.mp4")
    try:
        source.extract_clip(padded_start, padded_end, padded_path)
        total = boundaries.probe_duration(padded_path)
        silences = boundaries.detect_silences(padded_path, total)
        cut_start = boundaries.choose_start(silences, r.start_sec - padded_start)
        cut_end = min(boundaries.choose_end(silences, r.end_sec - padded_start), total)
        lo, hi = bound
        if lo is not None:
            cut_start = max(cut_start, lo - padded_start)
        if hi is not None:
            cut_end = min(cut_end, hi - padded_start)
        if cut_end - cut_start < boundaries.MIN_CLIP_SEC:
            raise ValueError("snapped clip too short")
        boundaries.trim(padded_path, cut_start, cut_end, part_path)
        return part_path, ClipRange(padded_start + cut_start, padded_start + cut_end)
    except Exception as e:  # noqa: BLE001 - snapping is an improvement, never a reason to fail the clip
        print(f"[clipbuild] 문장 경계 보정 실패, 원래 구간을 그대로 사용합니다: {e}")
        return source.extract_clip(r.start_sec, r.end_sec, part_path), r
    finally:
        padded_path.unlink(missing_ok=True)


def build_clip_snapped(
    source: MediaSource, ranges: list[ClipRange], out_path: Path, work_dir: Path | None = None,
    bounds: list[tuple[float | None, float | None]] | None = None,
) -> tuple[Path, list[ClipRange]]:
    """build_clip, but each range's start/end is first moved to the nearest
    audio pause. Returns (clip_path, ranges_actually_used): captions must be
    built from the returned ranges, not the requested ones, so they stay in
    sync with the (slightly shifted) footage."""
    if not ranges:
        raise ValueError("at least one range is required")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = work_dir or out_path.parent / f"_{out_path.stem}_parts"
    work_dir.mkdir(parents=True, exist_ok=True)

    parts: list[Path] = []
    used: list[ClipRange] = []
    for i, r in enumerate(ranges):
        bound = bounds[i] if bounds and i < len(bounds) else (None, None)
        path, used_range = _snap_range_to_pauses(source, r, work_dir / f"part{i}.mp4", bound)
        parts.append(path)
        used.append(used_range)

    if len(parts) == 1:
        shutil.move(str(parts[0]), str(out_path))
        return out_path, used

    concat_list = work_dir / "concat.txt"
    concat_list.write_text(
        "\n".join(f"file '{p.resolve().as_posix()}'" for p in parts), encoding="utf-8"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(out_path)],
        check=True, capture_output=True,
    )
    return out_path, used
