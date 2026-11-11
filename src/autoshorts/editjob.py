"""Persistent state + pure editing logic for the staged auto-generation flow
("자동 생성"): analysis picks candidate blocks first, then a worker edits
each block's caption lines (fix typos, delete lines - which also cuts that
footage out), watches a confirmation render, and only then uploads.

Everything lives on disk under work/edit_jobs/<job_id>/ so a job survives
app restarts (editing can take hours and the app may be rebuilt meanwhile):

    job.json        job + block state (small, rewritten on every change)
    segments.json   the full transcript with speaker roles
    blocks/<n>/     per-block media (clip.mp4, final.mp4)

A block edits a span of transcript segments [first, last] (indexes into
segments.json). `deleted` lists segment indexes inside the span that are cut
out, `texts` holds caption text edits by segment index. Deleting a line cuts
its footage (block_ranges); editing text only changes the caption.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from .analyze import Candidate, ClipRange, TIER_LABELS
from .transcript import Segment

_lock = threading.RLock()

# transcript lines shown before/after a block's span so the worker can pull
# neighbouring lines in
CONTEXT_LINES = 8
# ranges shorter than this (seconds) are dropped as slivers
MIN_RANGE_SEC = 0.3
KEEP_DAYS_AFTER_FINISH = 7

_ID_RE = re.compile(r"^[0-9a-f]{6,32}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jobs_root(work_dir: Path) -> Path:
    return work_dir / "edit_jobs"


def job_dir(work_dir: Path, job_id: str) -> Path:
    if not _ID_RE.match(job_id):
        raise ValueError(f"invalid job id: {job_id!r}")
    return jobs_root(work_dir) / job_id


def block_dir(work_dir: Path, job_id: str, index: int) -> Path:
    return job_dir(work_dir, job_id) / "blocks" / str(int(index))


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------- jobs

def create_job(
    work_dir: Path, *, source_type: str, url: str = "", source_path: str = "",
    clip_length_key: str, tiers: list[str], tier_counts: dict[str, int], keyword: str = "",
) -> dict:
    """source_type: "youtube"(url) 또는 "file"(source_path = 서버가 저장해 둔 원본 영상 파일)."""
    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "source_type": source_type,
        "url": url,
        "source_path": source_path,
        "clip_length_key": clip_length_key,
        "tiers": tiers,
        "tier_counts": tier_counts,
        "keyword": keyword,
        "video_title": "",
        "status": "analyzing",  # analyzing | editing | error
        "message": "",
        "created_at": _now(),
        "finished_at": None,
        "blocks": [],
    }
    with _lock:
        _write_json(job_dir(work_dir, job_id) / "job.json", job)
    return job


def load_job(work_dir: Path, job_id: str) -> dict | None:
    try:
        path = job_dir(work_dir, job_id) / "job.json"
    except ValueError:
        return None
    with _lock:
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None


def save_job(work_dir: Path, job: dict) -> None:
    with _lock:
        _write_json(job_dir(work_dir, job["id"]) / "job.json", job)


def update_job(work_dir: Path, job_id: str, mutator: Callable[[dict], None]) -> dict:
    """Load, apply mutator(job) and save under one lock - background render/
    upload tasks and web requests all go through this so they never clobber
    each other's changes."""
    with _lock:
        job = load_job(work_dir, job_id)
        if job is None:
            raise KeyError(job_id)
        mutator(job)
        save_job(work_dir, job)
        return job


def save_segments(work_dir: Path, job_id: str, segments: list[Segment]) -> None:
    data = [
        {"start": s.start_sec, "end": s.end_sec, "text": s.text, "speaker": s.speaker}
        for s in segments
    ]
    with _lock:
        _write_json(job_dir(work_dir, job_id) / "segments.json", data)


def load_segments(work_dir: Path, job_id: str) -> list[Segment]:
    path = job_dir(work_dir, job_id) / "segments.json"
    with _lock:
        if not path.is_file():
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
    return [Segment(d["start"], d["end"], d["text"], d.get("speaker", "")) for d in raw]


def list_jobs(work_dir: Path) -> list[dict]:
    """Job summaries, newest first."""
    root = jobs_root(work_dir)
    if not root.is_dir():
        return []
    jobs = []
    for d in root.iterdir():
        job = load_job(work_dir, d.name) if d.is_dir() else None
        if job:
            jobs.append(job)
    jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    return [summarize_job(j) for j in jobs]


def summarize_job(job: dict) -> dict:
    enabled = [b for b in job.get("blocks", []) if b.get("enabled", True)]
    uploaded = sum(1 for b in enabled if b.get("upload", {}).get("status") == "done")
    rendered = sum(1 for b in enabled if b.get("render", {}).get("status") == "ready")
    return {
        "id": job["id"],
        "source_type": job.get("source_type", "youtube"),
        "video_title": job.get("video_title", ""),
        "status": job.get("status", ""),
        "message": job.get("message", ""),
        "created_at": job.get("created_at", ""),
        "stage": stage_label(job),
        "blocks_total": len(enabled),
        "blocks_rendered": rendered,
        "blocks_uploaded": uploaded,
    }


def stage_label(job: dict) -> str:
    status = job.get("status")
    if status == "analyzing":
        return "후보 분석 중"
    if status == "error":
        return "오류"
    blocks = [b for b in job.get("blocks", []) if b.get("enabled", True)]
    if blocks and all(b.get("upload", {}).get("status") == "done" for b in blocks):
        return "업로드 완료"
    if any(b.get("upload", {}).get("status") == "done" for b in blocks):
        return "일부 업로드됨"
    if any(b.get("render", {}).get("status") in ("ready", "rendering") for b in blocks):
        return "확인 영상 단계"
    return "자막 편집 대기"


def delete_job(work_dir: Path, job_id: str) -> None:
    with _lock:
        shutil.rmtree(job_dir(work_dir, job_id), ignore_errors=True)


def cleanup_old(work_dir: Path, now: datetime | None = None, days: int = KEEP_DAYS_AFTER_FINISH) -> list[str]:
    """Delete finished jobs (everything uploaded) whose finish is more than
    `days` old - their intermediate videos are big. Unfinished jobs are
    never touched. Returns the removed job ids."""
    now = now or datetime.now(timezone.utc)
    removed = []
    root = jobs_root(work_dir)
    if not root.is_dir():
        return removed
    for d in list(root.iterdir()):
        job = load_job(work_dir, d.name) if d.is_dir() else None
        if not job or not job.get("finished_at"):
            continue
        try:
            finished = datetime.fromisoformat(job["finished_at"])
        except ValueError:
            continue
        if now - finished > timedelta(days=days):
            delete_job(work_dir, job["id"])
            removed.append(job["id"])
    return removed


def cleanup_old_dirs(root: Path, days: int = KEEP_DAYS_AFTER_FINISH, now: float | None = None) -> list[str]:
    """Delete sub-folders of `root` not modified for `days` (e.g. the old
    one-click jobs' work folders: everything in them is already uploaded)."""
    import time
    if not root.is_dir():
        return []
    cutoff = (now if now is not None else time.time()) - days * 86400
    removed = []
    for d in list(root.iterdir()):
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                removed.append(d.name)
        except OSError:
            continue
    return removed


def recover_interrupted(work_dir: Path) -> None:
    """Called at app start: whatever was running when the app last closed is
    not running any more. Analyzing jobs become errors (delete + restart),
    renders go back to "none" (just press again), uploads become errors with
    a hint to check YouTube Studio - the video may or may not have gone up."""
    root = jobs_root(work_dir)
    if not root.is_dir():
        return
    for d in list(root.iterdir()):
        job = load_job(work_dir, d.name) if d.is_dir() else None
        if not job:
            continue

        def fix(j: dict) -> None:
            if j.get("status") == "analyzing":
                j["status"] = "error"
                j["message"] = "분석 중에 앱이 꺼졌습니다. 이 작업을 삭제하고 다시 시작해 주세요."
            for b in j.get("blocks", []):
                if b.get("render", {}).get("status") == "rendering":
                    b["render"].update(status="none", message="")
                if b.get("upload", {}).get("status") == "uploading":
                    b["upload"].update(
                        status="error",
                        message="업로드 중에 앱이 꺼졌습니다. 유튜브 스튜디오에 올라갔는지 먼저 확인해 주세요.",
                    )

        update_job(work_dir, job["id"], fix)


def mark_finished_if_done(job: dict) -> None:
    """Sets finished_at once every enabled block is uploaded (clears it if
    that stops being true)."""
    blocks = [b for b in job.get("blocks", []) if b.get("enabled", True)]
    done = bool(blocks) and all(b.get("upload", {}).get("status") == "done" for b in blocks)
    if done and not job.get("finished_at"):
        job["finished_at"] = _now()
    elif not done:
        job["finished_at"] = None


# ---------------------------------------------------------------- blocks

def _span_of(c: Candidate, segments: list[Segment]) -> tuple[int, int] | None:
    """First/last transcript line of a candidate. A line counts only if at
    least half of it lies inside the candidate's range: neighbouring lines
    routinely overlap by a second or two (rolling captions), and those
    must not be pulled in."""
    start, end = c.ranges[0].start_sec, c.ranges[-1].end_sec
    idxs = []
    for i, s in enumerate(segments):
        overlap = min(s.end_sec, end) - max(s.start_sec, start)
        dur = s.end_sec - s.start_sec
        if overlap > 0 and (dur <= 0 or overlap >= dur / 2):
            idxs.append(i)
    return (idxs[0], idxs[-1]) if idxs else None


def title_from_thumbnail(thumbnail_text: str, fallback: str = "") -> str:
    """업로드 제목 = 썸네일 문구의 모든 줄을 공백으로 이어 붙인 것 (2줄이면 2줄 다)."""
    joined = " ".join(p.strip() for p in (thumbnail_text or "").splitlines() if p.strip())
    return joined or fallback


def build_blocks(
    candidates: list[Candidate],
    segments: list[Segment],
    *,
    guest_label: str,
    shoot_date: str,
    date_kind: str,
) -> list[dict]:
    blocks: list[dict] = []
    for c in candidates:
        span = _span_of(c, segments)
        if span is None:
            continue
        first, last = span
        thumb = c.thumbnail_text or ""
        title = title_from_thumbnail(thumb, c.title)
        blocks.append({
            "index": len(blocks),
            "enabled": True,
            "tier": c.tier,
            "tier_label": TIER_LABELS.get(c.tier, c.tier),
            "title": title,
            "thumbnail_text": thumb,
            "summary": c.summary,
            "reason": c.reason,
            "guest_label": guest_label,
            "shoot_date": shoot_date,
            "date_kind": date_kind,
            "first": first,
            "last": last,
            "deleted": [],
            "texts": {},
            "render": {"status": "none", "hash": "", "message": "", "version": 0},
            "upload": {"status": "none", "video_id": "", "url": "", "message": ""},
        })
    return blocks


def kept_indices(block: dict) -> list[int]:
    deleted = set(block.get("deleted", []))
    return [i for i in range(block["first"], block["last"] + 1) if i not in deleted]


def _runs(indices: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for i in indices:
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


def block_ranges(
    block: dict, segments: list[Segment],
) -> tuple[list[ClipRange], list[tuple[float | None, float | None]]]:
    """The footage ranges a block keeps, plus per-range (lo, hi) limits the
    audio-pause snapping must not cross (the edges that border a deleted
    line - otherwise snapping could pull the deleted words back in).

    Consecutive kept lines form one range; every deleted run splits the
    block into separate ranges. Transcript lines overlap their neighbours by
    a second or two, so a range's edge next to a deleted run is clamped to
    that run.
    """
    deleted = set(block.get("deleted", []))
    first, last = block["first"], block["last"]
    ranges: list[ClipRange] = []
    bounds: list[tuple[float | None, float | None]] = []
    for run in _runs(kept_indices(block)):
        a, b = run[0], run[-1]
        start, end = segments[a].start_sec, segments[b].end_sec
        lo = hi = None
        if a - 1 >= first and (a - 1) in deleted:
            j = a - 1
            lo = segments[j].end_sec
            while j - 1 >= first and (j - 1) in deleted:
                j -= 1
                lo = max(lo, segments[j].end_sec)
            start = max(start, lo)
        if b + 1 <= last and (b + 1) in deleted:
            j = b + 1
            hi = segments[j].start_sec
            while j + 1 <= last and (j + 1) in deleted:
                j += 1
                hi = min(hi, segments[j].start_sec)
            end = min(end, hi)
        if end - start < MIN_RANGE_SEC:
            continue
        ranges.append(ClipRange(start_sec=start, end_sec=end))
        bounds.append((lo, hi))
    return ranges, bounds


def block_duration(block: dict, segments: list[Segment]) -> float:
    ranges, _ = block_ranges(block, segments)
    return sum(r.end_sec - r.start_sec for r in ranges)


def block_cue_segments(block: dict, segments: list[Segment]) -> list[Segment]:
    """Segments that may produce captions: only the kept lines, with the
    worker's text edits applied (deleted/outside lines must not sneak a
    caption in through the neighbour overlap)."""
    texts = block.get("texts", {})
    out = []
    for i in kept_indices(block):
        seg = segments[i]
        text = texts.get(str(i), seg.text)
        out.append(Segment(seg.start_sec, seg.end_sec, text, seg.speaker))
    return out


def _sha(obj) -> str:
    return hashlib.sha1(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def clip_hash(block: dict, segments: list[Segment]) -> str:
    ranges, _ = block_ranges(block, segments)
    return _sha([(round(r.start_sec, 3), round(r.end_sec, 3)) for r in ranges])


def render_hash(block: dict, segments: list[Segment]) -> str:
    """Identifies everything that goes into the confirmation render - when it
    differs from the hash stored at render time, the video is stale."""
    texts = block.get("texts", {})
    return _sha({
        "clip": clip_hash(block, segments),
        "texts": [(i, texts.get(str(i), segments[i].text)) for i in kept_indices(block)],
        "thumb": block.get("thumbnail_text", ""),
        "guest": block.get("guest_label", ""),
        "date": block.get("shoot_date", ""),
        "kind": block.get("date_kind", ""),
    })


def render_state(block: dict, segments: list[Segment]) -> str:
    """none | rendering | ready | stale | error"""
    status = block.get("render", {}).get("status", "none")
    if status == "ready" and block["render"].get("hash") != render_hash(block, segments):
        return "stale"
    return status


def apply_block_edit(block: dict, segments: list[Segment], payload: dict) -> None:
    """Validate a worker's edit and apply it to the block in place. Raises
    ValueError with a Korean message for anything invalid (nothing is
    changed in that case)."""
    n = len(segments)
    first = int(payload.get("first", block["first"]))
    last = int(payload.get("last", block["last"]))
    if not (0 <= first <= last < n):
        raise ValueError("구간 범위가 올바르지 않습니다.")
    deleted = sorted({int(i) for i in payload.get("deleted", block.get("deleted", []))
                      if first <= int(i) <= last})
    if len(deleted) >= last - first + 1:
        raise ValueError("모든 줄을 삭제할 수는 없습니다. 최소 한 줄은 남겨야 합니다.")
    texts: dict[str, str] = {}
    for key, value in (payload.get("texts", block.get("texts", {})) or {}).items():
        i = int(key)
        if first <= i <= last and i not in deleted and isinstance(value, str):
            value = value.replace("\n", " ").strip()
            if value != segments[i].text.strip():
                texts[str(i)] = value
    new_block = dict(block)
    new_block.update(first=first, last=last, deleted=deleted, texts=texts)
    if not block_ranges(new_block, segments)[0]:
        raise ValueError("남은 영상 길이가 너무 짧습니다.")

    block.update(first=first, last=last, deleted=deleted, texts=texts)
    for key in ("thumbnail_text", "title", "guest_label", "shoot_date"):
        if key in payload and isinstance(payload[key], str):
            block[key] = payload[key].strip("\r") if key == "thumbnail_text" else payload[key].strip()
    if payload.get("date_kind") in ("촬영", "방송"):
        block["date_kind"] = payload["date_kind"]
    if "enabled" in payload:
        block["enabled"] = bool(payload["enabled"])
